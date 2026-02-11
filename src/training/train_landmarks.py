from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_val_split(X: np.ndarray, y: np.ndarray, val_fraction: float, seed: int):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(xb.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmarks-npz", type=str, required=True)
    parser.add_argument("--class-to-idx", type=str, required=True)
    parser.add_argument("--out-root", type=str, default="artifacts/runs_landmarks")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.landmarks_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)   # (N,63)
    y = data["y"].astype(np.int64)     # (N,)

    class_to_idx = json.loads(Path(args.class_to_idx).read_text())
    num_classes = len(class_to_idx)

    train_idx, val_idx = train_val_split(X, y, args.val_fraction, args.seed)

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(in_dim=X.shape[1], num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.out_root) / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps({
        "landmarks_npz": args.landmarks_npz,
        "num_classes": num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }, indent=2))

    best_val_acc = -1.0
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(xb.size(0))

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        # save last
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_acc": best_val_acc}, ckpt_dir / "last.pt")

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_val_acc": best_val_acc}, ckpt_dir / "best.pt")

        print(f"Epoch {epoch}/{args.epochs} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

    print(f"Run saved to: {run_dir}")
    print(f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
