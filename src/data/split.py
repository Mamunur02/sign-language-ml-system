from torch.utils.data import random_split
from src.utils.config import TrainConfig
from src.data.dataset import ImageFolderDataset
from src.data.transforms import build_train_transforms, build_val_transforms
import torch


def make_splits(val_fraction: float = 0.1):
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)

    full_ds = ImageFolderDataset(
        cfg.dataset_dir,
        transform=build_train_transforms(cfg.image_size),
    )

    n_val = int(len(full_ds) * val_fraction)
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # swap transforms for val
    val_ds.dataset.transform = build_val_transforms(cfg.image_size)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = make_splits()
    print("Train size:", len(train_ds))
    print("Val size:", len(val_ds))
