import math
from abc import abstractclassmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Tuple, Union
import inspect
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAccuracy, MultilabelAUROC
from torch.nn import functional as F
from loguru import logger
from .callbacks.ema_callback import EMAWeightUpdate
from .utils import exclude_from_wt_decay, freeze_layers, load_from_ssl_checkpoint
from pytorch_lightning.loggers import TensorBoardLogger

class AbstractClassifier(pl.LightningModule):
    def __init__(self, *, num_classes: int = 12, **kwargs):
        super().__init__()
        # Metrics
        self.num_classes = num_classes
        self.acc_train = MultilabelAccuracy(num_labels=num_classes)
        self.acc_val = MultilabelAccuracy(num_labels=num_classes)
        self.acc_test = MultilabelAccuracy(num_labels=num_classes)
        self.auc_val = MultilabelAUROC(num_labels=num_classes)
        self.auc_test = MultilabelAUROC(num_labels=num_classes)

        # Stable pos_weight buffer (always present, avoids missing key warnings)
        pw = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("pos_weight", pw)
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.train_iters_per_epoch = None
        self.ema_model = None
        self.eman = kwargs.get('eman', True)

    def forward(
        self, x: torch.Tensor, k: int = None, agg: bool = True, ema: bool = False
    ) -> torch.Tensor:
        """Forward Pass for the model.

        Args:
            x (torch.Tensor): input
            k (int, optional): #MC samples. Defaults to None.
            agg (bool, optional): return logprobs if True, otherwise logits. Defaults to True.
            ema (bool, optional): select EMA model if True. Defaults to False.

        Returns:
            torch.Tensor: logprobs or logits
        """
        model_forward = self.select_forward_model(ema=ema)

        if k is None:
            # Use full MC only when not in training (e.g. acquisition / eval uncertainty)
            k = getattr(self, 'k', 1)
            if self.training:
                k = 1  # keep training fast

        sig = inspect.signature(model_forward.forward if hasattr(model_forward, 'forward') else model_forward)

        # If the underlying model supports internal k we can pass it, else loop
        if 'k' in sig.parameters:
            out = model_forward(x, k)
            # Expect shape [B,K,C] when k>1, else [B,C]
        else:
            if k > 1:
                outs = []
                # Ensure dropout active if caller set model to train()/custom mode
                for _ in range(k):
                    outs.append(model_forward(x))
                out = torch.stack(outs, dim=1)  # [B,K,C]
            else:
                out = model_forward(x)  # [B,C]

        if agg:
            # If we produced MC samples, aggregate by mean (logits)
            if out.dim() == 3:
                return out.mean(dim=1)
            return out
        else:
            # For downstream uncertainty code we always want [B,K,C]
            if out.dim() == 2:
                out = out.unsqueeze(1)  # [B,1,C]
            return out

    def mc_nll(self, logits: torch.Tensor) -> torch.Tensor:
        """Computs mean logits as required for predictive entropy

        Args:
            logits (torch.Tensor): logits

        Returns:
            torch.Tensor: NLL
        """
        out = torch.log_softmax(logits, dim=-1)
        if len(logits.shape) > 2:
            k = out.shape[1]
            out = torch.logsumexp(out, dim=1) - math.log(k)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        logits = self(x)
        loss = self.loss_fct(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        self.acc_train.update(preds, y.int())
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def select_forward_model(self, ema: bool = False) -> torch.nn.Module:
        """Selects exponential moving avg or normal model according to training state.
        During training select select model except when ema is True.
        During validation select ema model when defined.

        Args:
            ema (bool, optional): _description_. Defaults to False.

        Returns:
            torch.nn.Module: Model for execution of forward pass.
        """
        if ema and self.ema_model is not None:
            return self.ema_model
        elif self.training:
            return self.model
        else:
            if self.ema_model is not None:
                return self.ema_model
            else:
                return self.model

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get Features of underlying model.

        Args:
            x (torch.Tensor): input Nx...

        Returns:
            torch.Tensor: Features NxD
        """
        model_forward = self.select_forward_model()
        return model_forward.get_features(x).flatten(start_dim=1)

    def init_ema_model(self, use_ema: bool = False):
        """Initialize EMA model if use_ema is True.

        Args:
            use_ema (bool, optional): whether EMA IS USED. Defaults to False.
        """
        if use_ema:
            self.ema_model = deepcopy(self.model)
            for param in self.ema_model.parameters():
                param.requires_grad = False
            self.ema_weight_update = EMAWeightUpdate(eman=self.eman)

    def _compute_pos_weight_from_labelled(self, dm: pl.LightningDataModule):
        # Use ONLY labelled subset
        ds = dm.train_set  # ActiveLearningDataset
        if not hasattr(ds, "labelled"):
            return
        labelled_mask = ds.labelled
        if labelled_mask.sum() == 0:
            return
        ys = []
        for idx, is_lab in enumerate(labelled_mask):
            if not is_lab:
                continue
            _, y = ds._dataset[idx]  # underlying dataset item
            ys.append(torch.as_tensor(y).unsqueeze(0))
        if not ys:
            return
        y_all = torch.cat(ys, dim=0).float()  # [L,C] floats 0/1
        freq = y_all.mean(0).clamp(min=1e-6, max=1 - 1e-6)
        new_pw = (1 - freq) / freq
        new_pw = new_pw.clamp(max=10.0)  # <-- Cap pos_weight to 10.0
        if new_pw.shape != self.pos_weight.shape:
            # shape mismatch (should not happen) -> resize buffer
            self.register_buffer("pos_weight", new_pw.clone())
        else:
            self.pos_weight.copy_(new_pw)
        self.loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        logger.info(f"pos_weight updated: {new_pw.cpu().numpy()}  freq={freq.cpu().numpy()}")

    def setup_data_params(self, dm: pl.LightningDataModule):
        tl = dm.train_dataloader()
        if isinstance(tl, (list, tuple)):
            self.train_iters_per_epoch = max(len(x) for x in tl)
        else:
            self.train_iters_per_epoch = len(tl)
        if getattr(self.hparams, "data", {}).get("multilabel", True):
            self._compute_pos_weight_from_labelled(dm)

    def step(
        self, batch: Tuple[torch.tensor, torch.tensor], k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch  # y shape [B,C] multi-label 0/1
        logits = self.forward(x)
        loss = self.loss_fct(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).float()
        return loss, logits, preds, y

    def validation_step(
        self, batch, batch_idx, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): data from dataloader.
            batch_idx (int): batch counter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logprob, labels
        """
        mode = "val"
        loss, logits, preds, y = self.step(batch)
        y_int = y.int()  # <-- Ensure integer type
        self.acc_val.update(preds, y_int)
        self.auc_val.update(torch.sigmoid(logits), y_int)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{mode}/acc", self.acc_val.compute(), on_step=False, on_epoch=True)
        return logits, y

    def test_step(
        self, batch, batch_idx, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): data from dataloader.
            batch_idx (int): batch counter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logprob, labels
        """
        mode = "test"
        loss, logits, preds, y = self.step(batch)
        y_int = y.int()  # <-- Ensure integer type
        self.acc_test.update(preds, y_int)
        self.auc_test.update(torch.sigmoid(logits), y_int)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        return logits, y

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update the EMA model after every batch.

        Args:
            outputs (Any): outputs of model
            batch (Any): data from dataloader
            batch_idx (int): Batch counter
        """
        if self.ema_model is not None:
            self.ema_weight_update.on_train_batch_end(
                self.trainer, self, outputs, batch, batch_idx
            )

    def on_train_epoch_start(self) -> None:
        """Reset training accuracy internal state and set encoder when frozen in eval mode as well as EMA model."""
        self.acc_train.reset()
        if self.hparams.model.freeze_encoder:
            self.model.resnet.eval()
        # When ema model is used during training, correct buffers should be used
        # e.g. eman fixmatch should use eman-batchnorm for teacher!
        if self.ema_model is not None:
            self.ema_model.eval()

    def on_fit_start(self) -> None:
        """Initialize metrics for the tensorboard_logger.
        Either self.loggers is a list with tb_logger as first,
        or only tb_logger is present."""
        metric_placeholder = {"val/acc": 0.0, "test/acc": 0.0}
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

    def on_validation_epoch_start(self) -> None:
        """Reset validation accuracy internal state."""
        self.acc_val.reset()

    def on_test_epoch_start(self) -> None:
        """Rest test accuracy internal state."""
        self.acc_test.reset()

    def on_train_epoch_end(self) -> None:
        self.log("train/acc", self.acc_train.compute(), on_step=False, on_epoch=True)
        # Add train loss average from history
        losses = self.trainer.callback_metrics.get("train/loss_epoch", None)
        if losses is not None:
            self.log("train/loss", losses, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/acc", self.acc_val.compute(), on_step=False, on_epoch=True)
        losses = self.trainer.callback_metrics.get("val/loss", None)
        if losses is not None:
            self.log("val/loss", losses, on_step=False, on_epoch=True)
        auc_val = self.auc_val.compute()
        if auc_val.ndim == 0:  # scalar
            self.log("val/auc_macro", auc_val, on_step=False, on_epoch=True)
        else:
            for i, auc in enumerate(auc_val):
                self.log(f"val/auc_class_{i}", auc, on_step=False, on_epoch=True)
        self.auc_val.reset()

    def on_test_epoch_end(self) -> None:
        mode = "test"
        self.log(f"{mode}/acc", self.acc_test.compute(), on_step=False, on_epoch=True)
        auc_test = self.auc_test.compute()
        print("AUC test shape:", getattr(auc_test, 'shape', 'scalar'), "values:", auc_test)
        if auc_test.ndim == 0:  # scalar
            self.log(f"{mode}/auc_macro", auc_test, on_step=False, on_epoch=True)
        else:
            for i, auc in enumerate(auc_test):
                print(f"Logging test/auc_class_{i}: {auc}")
                self.log(f"test/auc_class_{i}", auc, on_step=False, on_epoch=True)
        self.auc_test.reset()

    def configure_optimizers(self):
        params = (
            list(self.model.parameters())
            if not self.hparams.model.freeze_encoder
            else list(self.model.head.parameters())
        )

        optimizer_name = self.hparams.optim.optimizer.name
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(params)
        elif optimizer_name == "sgd":
            momentum = self.hparams.optim.optimizer.momentum
            nesterov = self.hparams.optim.optimizer.nesterov
            optimizer = torch.optim.SGD(params, momentum=momentum, nesterov=nesterov)
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer_name}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=0.01,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1,
        }

    def visualize_inputs(self, inputs: torch.Tensor, name: str):
        """Saves images of inputs to logger[0] (Tensorboard) with info about epoch etc.

        Args:
            inputs (torch.Tensor): image data [3xHxW]
            name (str): name under which to display
        """
        num_imgs = 64
        num_rows = 8
        grid = (
            torchvision.utils.make_grid(
                inputs[:num_imgs], nrow=num_rows, normalize=True
            )
            .cpu()
            .detach()
        )
        if len(self.loggers) > 0:
            self.loggers[0].experiment.add_image(
                name,
                grid,
                self.current_epoch,
            )

    def load_from_ssl_checkpoint(self):
        """Loads a Self-Supervised Resnet from a Checkpoint obtained from PL Bolts"""
        ### IMPLEMENTATION ###
        # Code needs to be changed here to allow different ssl pretrained models to be loaded.
        load_from_ssl_checkpoint(self.model, path=self.hparams.model.load_pretrained)

    def wrap_dm(self, dm: pl.LightningDataModule) -> pl.LightningDataModule:
        """Prepare Datamodule for use. This here is a placeholder.

        Args:
            dm (pl.LightningDataModule): Datamodule to be used for training

        Returns:
            pl.LightningDataModule: Input datamodule without any changes
        """
        return dm

    # Optional: strict loading helper if you need manual loads elsewhere
    def load_only_state_dict(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        missing, unexpected = self.load_state_dict(ckpt["state_dict"], strict=False)
        if "pos_weight" in missing:
            logger.warning("pos_weight missing in checkpoint; keeping current value.")
        if len(unexpected) > 0:
            logger.warning(f"Unexpected keys: {unexpected}")
