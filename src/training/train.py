from __future__ import annotations
import os
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.utils.config import TrainConfig
from src.data.dataset import ImageFolderDataset
from src.data.transforms import build_train_transforms, build_val_transforms
from src.models.cnn import SimpleCNN


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainConfig,
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "num_classes": cfg.num_classes,
            "image_size": cfg.image_size,
            # Optional: helpful metadata for inference
            "class_to_idx": getattr(getattr(cfg, "class_to_idx", None), "value", None),
        },
        path,
    )

def try_resume(
    ckpt_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """
    Returns:
      start_epoch: int  (next epoch to run)
      best_val_acc: float
    """
    if not ckpt_path.exists():
        return 1, -1.0

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    last_epoch = int(ckpt.get("epoch", 0))
    best_val_acc = float(ckpt.get("best_val_acc", -1.0))

    # next epoch to run
    return last_epoch + 1, best_val_acc

def make_loaders(cfg: TrainConfig, val_fraction: float = 0.1):
    full_ds = ImageFolderDataset(
        cfg.dataset_dir,
        transform=build_train_transforms(cfg.image_size),
    )

    n_val = int(len(full_ds) * val_fraction)
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # swap transforms for validation split
    val_ds.dataset.transform = build_val_transforms(cfg.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    cfg = TrainConfig()
    cfg.ensure_dirs()

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = make_loaders(cfg, val_fraction=0.1)

    model = SimpleCNN(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    ckpt_dir = cfg.processed_data_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    last_ckpt_path = ckpt_dir / "last.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    start_epoch, best_val_acc = try_resume(last_ckpt_path, model, optimizer, device)
    if start_epoch > 1:
        print(f"Resuming from {last_ckpt_path} at epoch {start_epoch} (best_val_acc={best_val_acc:.4f})")
    else:
        print("Starting fresh training run.")

    for epoch in range(start_epoch, cfg.num_epochs + 1):

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Always save "last" so we can resume after interruptions
        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            epoch=epoch,
            cfg=cfg,
            best_val_acc=best_val_acc,
        )

        # Save "best" when validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch=epoch,
                cfg=cfg,
                best_val_acc=best_val_acc,
            )
            print(f"Saved best checkpoint to: {best_ckpt_path} (val_acc={best_val_acc:.4f})")


if __name__ == "__main__":
    main()
