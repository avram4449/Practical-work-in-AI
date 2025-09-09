import gc
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import TQDMProgressBar

from data.data import TorchVisionDM
from models.bayesian import BayesianModule
from models.callbacks.metrics_callback import (
    ImbClassMetricCallback,
    ISIC2016MetricCallback,
)
from query import QuerySampler
from utils.io import save_json
from utils.log_utils import log_git


class ActiveTrainingLoop(object):
    def __init__(
        self,
        cfg: DictConfig,
        datamodule: TorchVisionDM,
        count: Union[None, int] = None,
        active: bool = True,
        base_dir: str = os.getcwd(), 
        loggers: str = True,
    ):
        self.cfg = cfg
        self.datamodule = deepcopy(datamodule)
        self.count = count
        self.active = active
        self.device = "cuda:0"
        self.base_dir = Path(base_dir)  
        self._save_dict = dict()
        self._init_model()
        self.ckpt_callback = self._init_ckpt_callback()
        self.callbacks = self._init_callbacks()
        self.loggers = False
        if loggers:
            self.loggers = self._init_loggers()

    def _init_ckpt_callback(self) -> pl.callbacks.ModelCheckpoint:
        ckpt_path = os.path.join(self.log_dir, "checkpoints")
        ### Implementation ###
        # Datasets with specific main performance other than accuracy
        # add changes here.
        if self.datamodule.val_dataloader() is not None:
            if self.cfg.data.name == "isic2016":
                monitor = "val/auroc"
                mode = "max"
            elif "isic" in self.cfg.data.name:
                monitor = "val/w_acc"
                mode = "max"
            elif self.cfg.data.name == "miotcd":
                monitor = "val/w_acc"
                mode = "max"
            else:
                monitor = "val/acc"
                mode = "max"
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path,
                monitor=monitor,
                mode=mode,
                save_last=self.cfg.trainer.save_last,
            )
        else:
            ckpt_callback = pl.callbacks.ModelCheckpoint(
                dirpath=ckpt_path, monitor="train/acc", mode="max"
            )
        return ckpt_callback

    def _init_callbacks(self):
        lr_monitor = pl.callbacks.LearningRateMonitor()
        callbacks = [lr_monitor]
        callbacks.append(self.ckpt_callback)
        if self.cfg.trainer.early_stop and self.datamodule.val_dataloader is not None:
            early_stop = pl.callbacks.EarlyStopping(
                monitor=self.cfg.trainer.get("early_stop_monitor", "val/acc"),
                mode=self.cfg.trainer.get("early_stop_mode", "max"),
                patience=self.cfg.trainer.get("early_stop_patience", 15),
                min_delta=self.cfg.trainer.get("early_stop_min_delta", 0.0001),
                verbose=True
            )
            callbacks.append(early_stop)
        if self.cfg.data.name == "isic2016":
            callbacks.append(ISIC2016MetricCallback())
        if self.cfg.data.name in ["isic2019", "miotcd"]:
            callbacks.append(
                ImbClassMetricCallback(num_classes=self.cfg.data.num_classes)
            )
        # add progress bar
        callbacks.append(
            TQDMProgressBar(refresh_rate=self.cfg.trainer.progress_bar_refresh_rate)
        )
        return callbacks

    @property
    def log_dir(self) -> Path:
        log_dir = Path(self.base_dir)
        if self.count is not None:
            log_dir = log_dir / self.version
        return log_dir

    @property
    def version(self) -> str:
        if self.count is not None:
            return "loop-{}".format(self.count)
        return self.cfg.trainer.experiment_id

    @property
    def name(self) -> str:
        name = self.cfg.trainer.experiment_name
        if self.count is not None:
            name = os.path.join(name, self.cfg.trainer.experiment_id)
        return name

    @property
    def data_ckpt_path(self) -> Path:
        return self.log_dir / "data_ckpt"

    @staticmethod
    def obtain_meta_data(repo_path: str, repo_name: str = "repo-name"):
        # based on: https://github.com/MIC-DKFZ/nnDetection/blob/6ac7dac6fd9ffd85b74682a2f565e0028305c2c0/scripts/train.py#L187-L226
        meta_data = {}
        meta_data["date"] = str(datetime.now())
        meta_data["git"] = log_git(repo_path)
        return meta_data

    def _init_loggers(self):
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=self.name,
            version=self.version,
        )
        # add csv logger for important values!
        csv_logger = pl.loggers.CSVLogger(
            save_dir=self.cfg.trainer.experiments_root,
            name=self.name,
            version=self.version,
        )
        return [tb_logger, csv_logger]

    def _init_model(self):
        self.model = BayesianModule(self.cfg)

    def _init_trainer(self):
        self.trainer = pl.Trainer(
            accelerator='gpu' if self.cfg.trainer.n_gpus > 0 else None,
            devices=self.cfg.trainer.n_gpus if self.cfg.trainer.n_gpus > 0 else None,
            logger=self.loggers,
            max_epochs=self.cfg.trainer.max_epochs,
            min_epochs=self.cfg.trainer.min_epochs,
            fast_dev_run=self.cfg.trainer.fast_dev_run,
            callbacks=self.callbacks,
            check_val_every_n_epoch=self.cfg.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.cfg.trainer.gradient_clip_val,
            precision=self.cfg.trainer.precision,
            benchmark=self.cfg.trainer.deterministic is False,
            deterministic=self.cfg.trainer.deterministic,
            profiler=self.cfg.trainer.profiler,
            # enable_progress_bar=self.cfg.trainer.enable_progress_bar,
        )

    def _fit(self):
        dm = self.model.wrap_dm(self.datamodule)
        self.model.setup_data_params(dm)

        self.trainer.fit(model=self.model, datamodule=dm)

        if not self.cfg.trainer.fast_dev_run and self.cfg.trainer.load_best_ckpt:
            best_path = self.ckpt_callback.best_model_path
            logger.info(f"Final Model from: {best_path}")
            self.model = self.model.load_from_checkpoint(best_path, strict=True)
            self.model.setup_data_params(dm)
            logger.info("Recomputed pos_weight after checkpoint restore.")
        else:
            logger.info("Using last model (no best ckpt reload).")

        gc.collect()
        torch.cuda.empty_cache()

    def _test(self):
        self.trainer.test(model=self.model, datamodule=self.datamodule)

    def active_callback(self):
        """Execute active learning logic. -- not included in main.
        Returns the queries to the oracle."""
        self.model = self.model.to(self.device)
        query_sampler = QuerySampler(
            self.cfg, self.model, count=self.count, device=self.device
        )
        query_sampler.setup()
        stored = query_sampler.active_callback(self.datamodule)
        return stored

    def final_callback(self):
        pass

    def _setup_log_struct(self):
        """Save Meta data to a json file."""
        meta_data = self.obtain_meta_data(
            os.path.dirname(os.path.abspath(__file__)), repo_name="realistic-al"
        )
        if os.path.exists(self.log_dir) is False:
            os.makedirs(self.log_dir)
        save_meta = self.log_dir / "meta.json"
        save_json(meta_data, save_meta)

    def log_save_dict(self):
        """
        Save the internal _save_dict to a JSON file in the log directory.
        Extend this method to save any additional state or metrics as needed.
        """
        save_path = self.log_dir / "save_dict.json"
        save_json(self._save_dict, save_path)

    @staticmethod
    def save_test_predictions(model, datamodule, log_dir):
        model.eval()
        test_loader = datamodule.test_dataloader()
        preds = []
        trues = []
        indices = []
        smiles_list = getattr(datamodule.test_dataset, "smiles", None)
        ids_list = getattr(datamodule.test_dataset, "ID", None)
        for i, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                x = x.to(model.device)
                logits = model(x)
                probas = torch.sigmoid(logits).cpu().numpy()
                preds.append(probas)
                trues.append(y.cpu().numpy())
                indices.extend(range(i * test_loader.batch_size, i * test_loader.batch_size + len(x)))
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        df = pd.DataFrame(preds, columns=[f"pred_target_{i}" for i in range(preds.shape[1])])
        for i in range(trues.shape[1]):
            df[f"true_target_{i}"] = trues[:, i]
        df["index"] = indices
        if smiles_list is not None:
            df["smiles"] = smiles_list
        if ids_list is not None:
            df["ID"] = ids_list
        df.to_csv(os.path.join(log_dir, "test_predictions.csv"), index=False)

    def main(self):
        """Executing logic of the Trainer.
        setup_..., init_..., fit, test, final_callback"""
        self._setup_log_struct()
        if self.active:
            self.datamodule.train_set.save_checkpoint(self.data_ckpt_path)
        self._init_trainer()
        self._fit()
        if self.trainer.interrupted:
            return
        if self.cfg.trainer.run_test:
            self._test()
            ActiveTrainingLoop.save_test_predictions(self.model, self.datamodule, self.log_dir)
        self.final_callback()
