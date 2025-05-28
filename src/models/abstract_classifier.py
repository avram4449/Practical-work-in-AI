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
import torchvision
from loguru import logger
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import Accuracy
from torchmetrics.classification import MultilabelAUROC

from .callbacks.ema_callback import EMAWeightUpdate
from .utils import exclude_from_wt_decay, freeze_layers, load_from_ssl_checkpoint


class AbstractClassifier(pl.LightningModule):
    def __init__(self, eman: bool = True):
        """Abstract Classifier carrying the logic for Bayesian Models with MC Dropout and logging for base values.
        Dropout is per default used always, also during validation due to make use of its Bayesian properties.

        Args:
            eman (bool, optional): Whether or not to use Exponentially Moving AVerage Norm model. Defaults to True.
        """
        super().__init__()

        # general model
        self.train_iters_per_epoch = None

        self.ema_model = None
        self.eman = eman

        self.acc_train = Accuracy(task="multilabel", num_labels=12)
        self.acc_val = Accuracy(task="multilabel", num_labels=12)
        self.acc_test = Accuracy(task="multilabel", num_labels=12)
        self.auc_val = MultilabelAUROC(num_labels=12)
        self.auc_test = MultilabelAUROC(num_labels=12)
        self.loss_fct = nn.BCEWithLogitsLoss()

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
            k = getattr(self, 'k', 1)

        sig = inspect.signature(model_forward.forward if hasattr(model_forward, 'forward') else model_forward)
        if 'k' in sig.parameters:
            out = model_forward(x, k)
        else:
            out = model_forward(x)
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
        loss, logprob, preds, y = self.step(batch)
        self.acc_train.update(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.acc_train.compute(), on_step=False, on_epoch=True)
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

    def step(
        self, batch: Tuple[torch.tensor, torch.tensor], k: int = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard supervised step using provided loss function.

        Args:
            batch (Tuple[torch.tensor, torch.tensor]): Data from Dataloader.
            k (int, optional): #MC sampels. Defaults to None.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: loss, logprob, predictions, labels
        """
        x, y = batch
        logprob = self.forward(x)
        loss = self.loss_fct(logprob, y)
        preds = (torch.sigmoid(logprob) > 0.5).float()
        return loss, logprob, preds, y

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): data from dataloader.
            batch_idx (int): batch counter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logprob, labels
        """
        mode = "val"
        loss, logprob, preds, y = self.step(batch)
        self.acc_val.update(preds, y)
        self.auc_val.update(torch.sigmoid(logprob), y.long())
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.log(f"{mode}/acc", self.acc_val.compute(), on_step=False, on_epoch=True)
        if batch_idx == 0 and self.current_epoch == 0:
            if len(batch[0].shape) == 4:
                self.visualize_inputs(batch[0], name=f"{mode}/data")
        return logprob, y

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Supervised test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): data from dataloader.
            batch_idx (int): batch counter

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: logprob, labels
        """
        mode = "test"
        loss, logprob, preds, y = self.step(batch)
        self.log(f"{mode}/loss", loss, on_step=False, on_epoch=True)
        self.acc_test.update(preds, y)
        self.auc_test.update(torch.sigmoid(logprob), y.long())
        if batch_idx == 0:
            if len(batch[0].shape) == 4:
                self.visualize_inputs(batch[0], name=f"{mode}/data")
        return logprob, y

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
        """Log values during test to disk."""
        mode = "test"
        self.log(f"{mode}/acc", self.acc_test.compute(), on_step=False, on_epoch=True)
        auc_test = self.auc_test.compute()
        if auc_test.ndim == 0:  # scalar
            self.log(f"{mode}/auc_macro", auc_test, on_step=False, on_epoch=True)
        else:
            for i, auc in enumerate(auc_test):
                self.log(f"test/auc_class_{i}", auc, on_step=False, on_epoch=True)
        self.auc_test.reset()

    def setup_data_params(self, dm: pl.LightningDataModule):
        """Create internal parameter with the amount of training iterations per epoch.
        Set up weighted loss values if specified in config.

        Args:
            dm (pl.LightningDataModule): DataModule
        """
        train_loader = dm.train_dataloader()
        if isinstance(train_loader, (tuple, list)):
            self.train_iters_per_epoch = max([len(loader) for loader in train_loader])
        else:
            self.train_iters_per_epoch = len(train_loader)

        # This implementation is correct and uses the amount of labels from the train_loader.
        # Therefore if Resampling is used, more samples are used.
        weighted_loss = False
        if "weighted_loss" in self.hparams.model:
            weighted_loss: bool = self.hparams.model.weighted_loss
        if weighted_loss:
            logger.info("Initializing Weighted Loss")
            if hasattr(dm.train_set, "targets"):
                classes: np.ndarray = dm.train_set.targets
            else:
                # workaround for FixMatch trainings with multiple dataloaders

                if isinstance(train_loader, (tuple, list)):
                    # train_loader 0 is the labeled loader
                    train_loader = train_loader[0]
                classes = []
                for x, y in train_loader:
                    classes.append(y.numpy())
                classes = np.concatenate(classes)

            classes, class_weights = np.unique(classes, return_counts=True)
            # computation identical to sklearn balanced class weights
            # https://github.com/scikit-learn/scikit-learn/blob/36958fb24/sklearn/utils/class_weight.py#L10
            class_weights = torch.tensor(
                np.sum(class_weights) / (len(classes) * class_weights),
                dtype=torch.float,
            )
            self.loss_fct = nn.NLLLoss(weight=class_weights)

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

    def load_only_state_dict(self, path: str):
        """Load the state from checkpoint path.

        Args:
            path (str): Path to torch loadable file.
        """
        ckpt = torch.load(path)
        logger.debug("Loading Model from Path: {}".format(path))
        logger.info("Loading checkpoint from Epoch: {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)

    def get_best_ckpt(
        self, experiment_path: Union[str, Path], use_last: bool = True
    ) -> Path:
        """Return the path to the best checkpoint

        Args:
            experiment_path (Union[str, Path]): path to base experiment

        Returns:
            Path: Best checkpoint path
        """
        model_ckpt_path = Path(experiment_path) / "checkpoints"
        ckpts = [ckpt for ckpt in model_ckpt_path.iterdir() if ckpt.suffix == ".ckpt"]
        # print(ckpts)
        if "last.ckpt" in [ckpt.name for ckpt in ckpts] and use_last:
            model_ckpt = model_ckpt_path / "last.ckpt"
        else:
            ckpts_f = [ckpt for ckpt in ckpts if "last.ckpt" not in ckpt.name]
            ckpts_f.sort(key=lambda x: x.name.split("=")[1].split("-")[0])
            if len(ckpts_f) == 0:
                raise FileNotFoundError(
                    "Path {} has no checkpoints ".format(model_ckpt_path)
                )
            model_ckpt = ckpts_f[-1]
        return model_ckpt
