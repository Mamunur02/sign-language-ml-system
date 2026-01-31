from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.data.dataset import ImageFolderDataset
from src.models.cnn import SimpleCNN


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: str) -> dict:
    """
    Supports checkpoints saved as:
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - raw state_dict
    """
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        # assume raw state dict
        model.load_state_dict(ckpt)

    return ckpt if isinstance(ckpt, dict) else {"raw_state_dict": True}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, topk: int = 3):
    model.eval()

    y_true = []
    y_pred = []
    correct1 = 0
    correctk = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        pred = probs.argmax(dim=1)
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

        # top-1 accuracy
        correct1 += (pred == y).sum().item()

        # top-k accuracy
        if topk is not None and topk > 1:
            topk_idx = probs.topk(k=topk, dim=1).indices
            # for each row, check if true label appears in topk indices
            hit = (topk_idx == y.unsqueeze(1)).any(dim=1)
            correctk += hit.sum().item()

        total += y.size(0)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    acc1 = correct1 / total if total > 0 else 0.0
    acck = (correctk / total if total > 0 else 0.0) if (topk is not None and topk > 1) else None
    return y_true, y_pred, acc1, acck


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a run checkpoint and write an eval report.")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to a run directory (contains checkpoints/)")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint filename inside run_dir/checkpoints/")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset root (class folders)")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--topk", type=int, default=3, help="Compute top-k accuracy (set 1 to disable top-k).")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Expected checkpoints folder: {ckpt_dir}")

    # Choose checkpoint
    if args.ckpt is None:
        # try common names first, else pick most recent .pt/.pth
        candidates = ["best.pt", "best.pth", "last.pt", "last.pth"]
        chosen = None
        for c in candidates:
            if (ckpt_dir / c).exists():
                chosen = ckpt_dir / c
                break
        if chosen is None:
            pts = sorted(list(ckpt_dir.glob("*.pt")) + list(ckpt_dir.glob("*.pth")), key=lambda p: p.stat().st_mtime)
            if not pts:
                raise FileNotFoundError(f"No .pt/.pth checkpoints found in {ckpt_dir}")
            chosen = pts[-1]
        ckpt_path = chosen
    else:
        ckpt_path = ckpt_dir / args.ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms (match your baseline)
    tfm = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
    ])

    # Dataset + loader
    dataset = ImageFolderDataset(args.data_dir, transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # Class names from dataset mapping
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Model
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    ckpt_meta = load_checkpoint(model, ckpt_path, device)

    # Evaluate
    topk = None if args.topk <= 1 else args.topk
    y_true, y_pred, acc1, acck = evaluate(model, loader, device, topk=topk)

    # Metrics
    report_dict = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    # Output dir
    report_dir = Path("reports") / run_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "classification_report.json").write_text(json.dumps(report_dict, indent=2))
    (report_dir / "confusion_matrix.json").write_text(json.dumps(cm.tolist(), indent=2))
    plot_confusion_matrix(cm, class_names, report_dir / "confusion_matrix.png")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "device": device,
        "num_samples": int(len(dataset)),
        "num_classes": int(len(class_names)),
        "top1_accuracy": float(acc1),
        "topk": int(args.topk),
        "topk_accuracy": (float(acck) if acck is not None else None),
        "ckpt_keys": (list(ckpt_meta.keys()) if isinstance(ckpt_meta, dict) else None),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("✅ Evaluation complete")
    print(f"Report directory: {report_dir}")
    print(f"Top-1 accuracy: {acc1:.4f}")
    if acck is not None:
        print(f"Top-{args.topk} accuracy: {acck:.4f}")


if __name__ == "__main__":
    main()
