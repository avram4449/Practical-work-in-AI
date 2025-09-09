import os

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import utils
from data.base_datamodule import BaseDataModule
from data.data import TorchVisionDM
from trainer import ActiveTrainingLoop
from utils import config_utils
from utils.log_utils import setup_logger
import torch
torch.set_float32_matmul_precision('medium')

@hydra.main(config_path="./config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    setup_logger()
    logger.info("Start logging")
    config_utils.print_config(cfg)
    train(cfg)


def get_torchvision_dm(cfg: DictConfig, active_dataset: bool = True) -> TorchVisionDM:
    """Initialize TorchVisionDM from config.

    Args:
        config (DictConfig): Config obtained
        active_dataset (bool, optional): . Defaults to True.

    Returns:
        TorchVisionDM: DataModule used for training.
    """
    balanced_sampling = False
    if "balanced_sampling" in cfg.data:
        balanced_sampling = cfg.data.balanced_sampling

    imbalance = None
    if "imbalance" in cfg.data:
        imbalance = cfg.data.imbalance
    val_size = None
    if "val_size" in cfg.data:
        val_size = cfg.data.val_size

    datamodule = TorchVisionDM(
        data_root=cfg.trainer.data_root,
        batch_size=cfg.trainer.batch_size,
        dataset=cfg.data.name,
        min_train=cfg.active.min_train,
        random_split=cfg.active.random_split,
        num_classes=cfg.data.num_classes,
        mean=cfg.data.mean,
        std=cfg.data.std,
        transform_train=cfg.data.transform_train,
        transform_test=cfg.data.transform_test,
        shape=cfg.data.shape,
        num_workers=cfg.trainer.num_workers,
        seed=cfg.trainer.seed,
        active=active_dataset,
        persistent_workers=cfg.trainer.persistent_workers,
        imbalance=imbalance,
        timeout=cfg.trainer.timeout,
        val_size=val_size,
        balanced_sampling=balanced_sampling,
    )

    return datamodule


def label_active_dm(
    cfg: DictConfig,
    num_labelled: int,
    balanced: bool,
    datamodule: BaseDataModule,
    balanced_per_cls: int = 5,
):
    """Label the Dataset according to rules."""
    cfg.data.num_classes = cfg.data.num_classes

    if cfg.data.name == "tox21":
        if datamodule.train_set.n_labelled > 0:
            return

        use_ml_balance = getattr(cfg.active, "multilabel_balanced_seed", True) and balanced
        if use_ml_balance:
            _label_multilabel_balanced(
                datamodule.train_set,
                num_classes=cfg.data.num_classes,
                budget=num_labelled,
                rng_seed=cfg.trainer.seed,
            )
        else:
            datamodule.train_set.label_randomly(num_labelled)

        assert datamodule.train_set.n_labelled > 0, "No initial labels assigned for Tox21."
        return

    if cfg.data.name in ["isic2019", "miotcd", "isic2016"] and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif datamodule.imbalance and balanced:
        label_balance = cfg.data.num_classes * balanced_per_cls
        datamodule.train_set.label_balanced(
            n_per_class=label_balance // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
        label_random = num_labelled - label_balance
        if label_random > 0:
            datamodule.train_set.label_randomly(label_random)
    elif balanced:
        datamodule.train_set.label_balanced(
            n_per_class=num_labelled // cfg.data.num_classes,
            num_classes=cfg.data.num_classes,
        )
    else:
        datamodule.train_set.label_randomly(num_labelled)


@logger.catch
def train(cfg: DictConfig):
    """Run standard training.

    Args:
        cfg (DictConfig): config from main
    """
    active_dataset = cfg.active.num_labelled is not None
    logger.info("Set seed")
    utils.set_seed(cfg.trainer.seed)

    datamodule = get_torchvision_dm(cfg, active_dataset)
    if cfg.active.num_labelled and cfg.data.name != "tox21":
        label_active_dm(cfg, cfg.active.num_labelled, cfg.active.balanced, datamodule)

    training_loop = ActiveTrainingLoop(
        cfg, datamodule, active=False, base_dir=os.getcwd()
    )
    training_loop.main()


if __name__ == "__main__":
    main()

def _label_multilabel_balanced(active_ds, num_classes: int, budget: int, rng_seed: int = 0):
    """
    Heuristic: cover positives per class first, then fill remainder randomly.
    """
    import numpy as np
    rng = np.random.default_rng(rng_seed)

    # Pool indices (unlabelled)
    pool_indices = (~active_ds.labelled).nonzero()[0]
    # Extract labels for pool
    lbl_matrix = []
    for idx in pool_indices:
        x, y = active_ds._dataset[idx]
        lbl_matrix.append(y.numpy() if hasattr(y, "numpy") else np.array(y))
    lbl_matrix = np.stack(lbl_matrix)  # [P, C]

    picked = set()
    # Pass 1: one positive per class where possible
    for c in range(num_classes):
        pos_idx = np.where(lbl_matrix[:, c] == 1)[0]
        pos_idx = [pool_indices[p] for p in pos_idx if pool_indices[p] not in picked]
        if pos_idx:
            choice = rng.choice(pos_idx)
            # convert oracle index to pool-relative index
            pool_rel = active_ds._oracle_to_pool_index(choice)[0]
            active_ds.label(pool_rel)
            picked.add(choice)
            if len(picked) >= budget:
                return

    # Pass 2: fill remaining randomly
    remaining_pool_oracle = [i for i in pool_indices if i not in picked]
    rng.shuffle(remaining_pool_oracle)
    for oracle_idx in remaining_pool_oracle:
        pool_rel = active_ds._oracle_to_pool_index(oracle_idx)[0]
        active_ds.label(pool_rel)
        picked.add(oracle_idx)
        if len(picked) >= budget:
            break
