from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data

from src.data.dataset import ImageFolderDataset
from src.data.transforms import build_val_transforms
from src.models.cnn import SimpleCNN

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """
    Returns:
      results_dict: {"loss": float, "acc": float, "n": int}
      y_true: (N,) cpu tensor of true labels
      y_pred: (N,) cpu tensor of predicted labels
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_true = []
    all_pred = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)

        correct += int((preds == y).sum().item())
        total += int(x.size(0))

        all_true.append(y.detach().cpu())
        all_pred.append(preds.detach().cpu())

    y_true = torch.cat(all_true) if all_true else torch.empty(0, dtype=torch.long)
    y_pred = torch.cat(all_pred) if all_pred else torch.empty(0, dtype=torch.long)

    results = {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "n": total,
    }
    return results, y_true, y_pred


def save_reports(
    run_dir: Path,
    ds,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    """
    Saves confusion matrix + classification report into run_dir/reports/.
    Works when ds is a Dataset or a torch.utils.data.Subset wrapping a Dataset.
    """
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # unwrap Subset if needed to access class_to_idx
    base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    if not hasattr(base_ds, "class_to_idx"):
        raise ValueError("Dataset has no class_to_idx; cannot build class names for report.")

    idx_to_class = {v: k for k, v in base_ds.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    yt = y_true.numpy()
    yp = y_pred.numpy()

    cm = confusion_matrix(yt, yp)
    report = classification_report(
        yt,
        yp,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    with open(report_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm.tolist(), f, indent=2)

    with open(report_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = range(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    fig.savefig(report_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved reports to: {report_dir}")


def main():
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

    # Build dataset
    ds = ImageFolderDataset(args.data_dir, transform=build_val_transforms(args.image_size))

    # Use saved val split if available
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

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)

    # Infer num_classes safely
    base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
    fallback_num_classes = len(base_ds.class_to_idx) if hasattr(base_ds, "class_to_idx") else None

    if isinstance(ckpt, dict):
        num_classes = ckpt.get("num_classes", fallback_num_classes)
    else:
        num_classes = fallback_num_classes

    if num_classes is None:
        raise ValueError("Could not infer num_classes (missing in checkpoint and dataset has no class_to_idx).")
    num_classes = int(num_classes)

    model = SimpleCNN(num_classes=num_classes).to(device)

    # Load model weights (support common formats)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # assume ckpt itself is a state_dict
        model.load_state_dict(ckpt)

    # Evaluate + collect predictions
    results, y_true, y_pred = evaluate(model, loader, device)

    # Save eval summary
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

    # Save confusion matrix + per-class report
    save_reports(run_dir, ds, y_true, y_pred)


if __name__ == "__main__":
    main()
