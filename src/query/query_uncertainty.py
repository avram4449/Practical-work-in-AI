import math
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.tensor import to_numpy

from .batchbald_redux.batchbald import get_batchbald_batch

### IMPLEMENTATION
# Add name of uncerainty method here and add computation _get_xxx_function.
NAMES = ["bald", "entropy", "random", "batchbald", "variationratios"]


def bald_bernoulli(
    logits_mc: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    agg: str = "mean",
    topk: Optional[int] = None,
):
    """
    Bernoulli (multi-label) BALD with flexible aggregation.
    logits_mc: [B,K,C] or [B,1,C]
    weights: optional per-class weights [C] (will be normalized)
    agg: "mean" | "weighted" | "topk_mean" | "max"
    topk: used when agg == "topk_mean" (e.g., 3)
    Returns: [B] scores
    """
    if logits_mc.dim() == 2:
        logits_mc = logits_mc.unsqueeze(1) 
    probs = torch.sigmoid(logits_mc)              
    mean_p = probs.mean(dim=1)                    
    eps = 1e-8
    H_mean = - (mean_p * torch.log(mean_p + eps) + (1 - mean_p) * torch.log(1 - mean_p + eps))  
    H_each = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))      
    E_H = H_each.mean(dim=1)                      
    MI = H_mean - E_H                             

    if agg == "max":
        return MI.max(dim=1).values 

    if agg == "topk_mean":
        k = topk if topk is not None else max(1, MI.shape[1] // 4)
        top_vals, _ = MI.topk(k=min(k, MI.shape[1]), dim=1)
        return top_vals.mean(dim=1)

    if agg == "weighted" and weights is not None:
        w = weights
        if w.dim() == 1:
            w = w / (w.sum() + eps)
            w = w.view(1, -1)       
        return (MI * w).sum(dim=1)  

    return MI.mean(dim=1)            


def get_acq_function(cfg, pt_model) -> Callable[[torch.Tensor], torch.Tensor]:
    name = str(cfg.query.name).split("_")[0]
    multilabel = bool(getattr(cfg.data, "multilabel", False))
    if name == "bald":
        k_acq = int(getattr(cfg.query, "k_acq", getattr(cfg.model, "k", 1)))
        agg = str(getattr(cfg.query, "bald_agg", "mean"))
        topk = getattr(cfg.query, "bald_topk", None)
        weight_mode = str(getattr(cfg.query, "bald_weight_mode", "none"))  

        weights = None
        if multilabel and weight_mode != "none":
            pw = getattr(pt_model, "pos_weight", None)
            if pw is not None:
                with torch.no_grad():
                    if weight_mode == "pos_weight":
                        w = pw.clone().float()
                    elif weight_mode == "sqrt_pos_weight":
                        w = pw.clone().float().sqrt()
                    elif weight_mode == "inv_pos_weight":
                        w = (1.0 / (pw.clone().float().clamp(min=1e-6)))
                    else:
                        w = None
                    if w is not None:
                        weights = w

        def _acq_bald(x: torch.Tensor):
            """Multi-label BALD with flexible aggregation."""
            with torch.no_grad():
                out = pt_model(x, agg=False, k=k_acq)  
                if multilabel:
                    w_dev = None if weights is None else weights.to(out.device)
                    scores = bald_bernoulli(out, weights=w_dev, agg=agg, topk=topk)
                else:
                    scores = mutual_bald(out)  
            return scores

        return _acq_bald
    elif name == "entropy":
        return _get_bay_entropy_fct(pt_model)
    elif name == "random":
        return _get_random_fct()
    elif name == "batchbald":
        return get_bay_logits(pt_model)
    elif name == "variationratios":
        return _get_var_ratios(pt_model)
    else:
        raise NotImplementedError


def get_post_acq_function(
    cfg: DictConfig, device="cuda:0"
) -> Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    names = str(cfg.query.name).split("_")
    if cfg.query.name == "batchbald":
        num_samples = 40000 

        def post_acq_function(logprob_n_k_c: np.ndarray, acq_size: int):
            """BatchBALD acquisition function using logits with iterative conditional mutual information."""
            assert (
                len(logprob_n_k_c.shape) == 3
            logprob_n_k_c = torch.from_numpy(logprob_n_k_c).to(
                device=device, dtype=torch.double
            )
            with torch.no_grad():
                out = get_batchbald_batch(
                    logprob_n_k_c,
                    batch_size=acq_size,
                    num_samples=num_samples,
                    dtype=torch.double,
                    device=device,
                )
            indices = np.array(out.indices)
            scores = np.array(out.scores)
            return indices, scores

        return post_acq_function
    else:

        def post_acq_function(acq_scores: np.ndarray, acq_size: int):
            """Acquires based on ranking. Highest ranks are acquired first."""
            assert len(acq_scores.shape) == 1  
            acq_ind = np.arange(len(acq_scores))
            inds = np.argsort(acq_scores)[::-1]
            inds = inds[:acq_size]
            acq_list = acq_scores[inds]
            acq_ind = acq_ind[inds]
            return inds, acq_list

        return post_acq_function


###


def query_sampler(
    dataloader: DataLoader,
    acq_function,
    post_acq_function,
    acq_size: int = 64,
    device="cuda:0",
):
    """Returns the queries (acquisition values and indices) given the data pool and the acquisition function.
    The Acquisition Function Returns Numpy arrays!"""
    acq_list = None
    counts = 0
    for i, batch in enumerate(dataloader):
        acq_values = acq_from_batch(batch, acq_function, device=device)
        if acq_values is None or len(acq_values) == 0:
            continue  
        if acq_list is None:
            shape = acq_values.shape
            new_shape = (len(dataloader) * dataloader.batch_size, *shape[1:])
            acq_list = np.zeros(new_shape)
        acq_list[counts : counts + len(acq_values)] = acq_values
        counts += len(acq_values)
    if acq_list is None or counts == 0:
        raise ValueError("No acquisition values were computed. Check your dataloader and acquisition function.")
    acq_list = acq_list[:counts]
    acq_ind, acq_scores = post_acq_function(acq_list, acq_size)

    return acq_ind, acq_scores


def _get_bay_entropy_fct(pt_model: torch.nn.Module):
    def acq_bay_entropy(x: torch.Tensor):
        """Returns the Entropy of predictions of the bayesian model"""
        with torch.no_grad():
            out = pt_model(x, agg=False)  
            ent = pred_entropy(out)
        return ent

    return acq_bay_entropy


def _get_exp_entropy_fct(pt_model: torch.nn.Module):
    def acq_exp_entropy(x: torch.Tensor):
        """Returns the expected entropoy of some probabilistic model."""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            ex_ent = exp_entropy(out)
        return ex_ent

    return acq_exp_entropy


def _get_bald_fct(pt_model: torch.nn.Module, multilabel: bool = False, k: int = 1):
    def acq_bald(x: torch.Tensor):
        with torch.no_grad():
            out = pt_model(x, agg=False, k=k) 
            if multilabel:
                scores = bald_bernoulli(out)
            else:
                scores = mutual_bald(out)  
        return scores

    return acq_bald


def get_bay_logits(pt_model: torch.nn.Module):
    def acq_logits(x: torch.Tensor):
        """Returns the NxKxC logprobs needed for BatchBALD"""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = torch.log_softmax(out, dim=2)
        return out

    return acq_logits


def _get_var_ratios(pt_model: torch.nn.Module):
    def acq_var_ratios(x: torch.Tensor):
        """Returns the variation ratio values."""
        with torch.no_grad():
            out = pt_model(x, agg=False)
            out = var_ratios(out)
        return out

    return acq_var_ratios


def _get_random_fct():
    def acq_random(x: torch.Tensor, c: float = 0.0001):
        """Returns random values over the interval [0, c)"""
        out = torch.rand(x.shape[0], device=x.device) * c
        return out

    return acq_random


def pred_entropy(logits):
    if logits.dim() == 2:
        logits = logits.unsqueeze(1) 
    out = F.log_softmax(logits, dim=2)
    p = out.exp()
    entropy = -(p * out).sum(dim=2).mean(dim=1)
    return entropy


def var_ratios(logits: torch.Tensor):
    k = logits.shape[1]
    out = F.log_softmax(logits, dim=2) 
    out = torch.logsumexp(out, dim=1) - math.log(k) 
    out = 1 - torch.exp(out.max(dim=-1).values) 
    return out


def exp_entropy(logits: torch.Tensor):
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    out = F.log_softmax(logits, dim=2)
    p = out.exp()
    entropy = -(p * out).sum(dim=2)
    return entropy.mean(dim=1)


def mutual_bald(logits: torch.Tensor):
    return pred_entropy(logits) - exp_entropy(logits)


def acq_from_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    function: Callable[[torch.Tensor], torch.Tensor],
    device="cuda:0",
) -> np.ndarray:
    """Compute function from batch inputs.

    Args:
        batch (Tuple[torch.Tensor, torch.Tensor]): [inputs, labels]
        function (Callable[[torch.Tensor], torch.Tensor]): function where outputs are desired.
        device (str, optional): device for computation. Defaults to "cuda:0".

    Returns:
        np.ndarray: outputs of function for batch inputs.
    """
    x, y = batch
    x = x.to(device)
    out = function(x)
    out = to_numpy(out)
    return out
