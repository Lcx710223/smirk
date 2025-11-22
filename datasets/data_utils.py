import torch
from datasets.lrs3_dataset import get_datasets_LRS3
from datasets.mead_dataset import get_datasets_MEAD
from datasets.mead_sides_dataset import get_datasets_MEAD_sides
from datasets.ffhq_dataset import get_datasets_FFHQ
from datasets.celeba_dataset import get_datasets_CelebA
from datasets.mixed_dataset_sampler import MixedDatasetBatchSampler
import os

import torch
from torch.utils.data import DataLoader

from datasets.me79_dataset import Me79Dataset

def get_datasets_ME79(config):
    """
    Return (train_ds, val_ds, test_ds) for the personal ME79 dataset.
    Uses config.dataset.ME79.root which should point to /content/ME79.
    Optional config.dataset.ME79.{train_list,val_list,test_list} may be provided.
    """
    ds_cfg = config.dataset.ME79
    root = ds_cfg.root
    train_list = getattr(ds_cfg, "train_list", None)
    val_list = getattr(ds_cfg, "val_list", None)
    test_list = getattr(ds_cfg, "test_list", None)

    train_ds = Me79Dataset(root=root, config=config, list_file=train_list, test=False)
    val_ds = Me79Dataset(root=root, config=config, list_file=val_list, test=True)
    test_ds = Me79Dataset(root=root, config=config, list_file=test_list, test=True) if test_list else None
    return train_ds, val_ds, test_ds


def load_dataloaders(config):
    """
    For your request: only ME79 is used for training/validation (no mixed sampling).
    Returns train_loader, val_loader.
    """
    train_ds, val_ds, test_ds = get_datasets_ME79(config)

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader