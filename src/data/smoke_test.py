from torch.utils.data import DataLoader

from src.utils.config import TrainConfig
from src.data.dataset import ImageFolderDataset
from src.data.transforms import build_train_transforms


def main():
    cfg = TrainConfig()
    cfg.ensure_dirs()

    ds = ImageFolderDataset(
        cfg.dataset_dir,
        transform=build_train_transforms(cfg.image_size),
    )

    print("Dataset size:", len(ds))
    print("Num classes:", len(ds.class_to_idx))
    print("First 10 classes:", list(ds.class_to_idx.keys())[:10])

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    x, y = next(iter(dl))
    print("Batch x shape:", tuple(x.shape))
    print("Batch y shape:", tuple(y.shape))
    print("y min/max:", int(y.min()), int(y.max()))


if __name__ == "__main__":
    main()
