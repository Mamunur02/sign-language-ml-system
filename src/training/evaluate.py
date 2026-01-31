from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data

from src.data.dataset import ImageFolderDataset
from src.data.transforms import build_val_transforms
from src.models.cnn import SimpleCNN


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(x.size(0))

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "n": total,
    }


def main():
    # Usage:
    # python -m src.training.evaluate --run_dir data/processed/runs/<id> --split_dir <path>
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to a run directory (contains checkpoints/)")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Checkpoint filename inside run_dir/checkpoints/")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (class folders)")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build dataset (full data for now; later we’ll evaluate on a saved val split)
    ds = ImageFolderDataset(args.data_dir, transform=build_val_transforms(args.image_size))
    val_idx_path = run_dir / "val_indices.json"
    if val_idx_path.exists():
        with open(val_idx_path, "r") as f:
            val_indices = json.load(f)
        ds = torch.utils.data.Subset(ds, val_indices)
        print(f"Evaluating on saved val split: n={len(val_indices)}")
    else:
        print("No val_indices.json found; evaluating on full dataset.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Load checkpoint metadata
    ckpt = torch.load(ckpt_path, map_location=device)
    base_ds = getattr(ds, "dataset", ds)  # unwrap Subset if needed
    class_to_idx = getattr(base_ds, "class_to_idx", None)
    fallback_num_classes = len(class_to_idx) if class_to_idx is not None else None

    num_classes = ckpt.get("num_classes", fallback_num_classes)
    if num_classes is None:
        raise ValueError("Could not infer num_classes (missing in checkpoint and dataset has no class_to_idx).")
    num_classes = int(num_classes)

    model = SimpleCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    results = evaluate(model, loader, device)

    out_path = run_dir / "eval.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "data_dir": str(Path(args.data_dir).resolve()),
                **results,
            },
            f,
            indent=2,
        )

    print("Saved:", out_path)
    print("Eval:", results)


if __name__ == "__main__":
    main()
