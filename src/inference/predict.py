# src/inference/predict.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image

from src.models.cnn import SimpleCNN
from src.data.transforms import build_val_transforms

# ----------------------------
# Defaults (override via env vars or CLI later if you want)
# ----------------------------
DEFAULT_CKPT = Path(os.getenv("CKPT_PATH", "artifacts/runs/latest/checkpoints/best.pt"))
DEFAULT_CLASS_MAP = Path(os.getenv("CLASS_MAP_PATH", "data/processed/metadata/class_to_idx.json"))
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 128))


def load_class_mapping(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping not found at {path}. "
            f"Expected data/processed/metadata/class_to_idx.json. "
            f"If you trained in Colab, copy that file locally or set CLASS_MAP_PATH."
        )
    class_to_idx = json.loads(path.read_text())
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> SimpleCNN:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = SimpleCNN(num_classes=num_classes).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # Your training saves a dict with "model_state_dict"
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint missing 'model_state_dict'. Keys found: {list(ckpt.keys())}"
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    ckpt_path: Path = DEFAULT_CKPT,
    class_map_path: Path = DEFAULT_CLASS_MAP,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device_str: str | None = None,
    topk: int = 5,
):
    image_path = Path(image_path)

    device = torch.device(device_str) if device_str else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    class_to_idx, idx_to_class = load_class_mapping(class_map_path)
    num_classes = len(class_to_idx)

    model = load_model(ckpt_path, num_classes=num_classes, device=device)

    # Use the SAME val transform builder as training
    tfm = build_val_transforms(image_size)

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    # Top-1
    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx].item())

    # Top-k for ambiguous classes (U/V etc.)
    topk = min(topk, num_classes)
    top_vals, top_idxs = torch.topk(probs, k=topk)
    top = [
        {"label": idx_to_class[int(i.item())], "prob": float(v.item())}
        for v, i in zip(top_vals, top_idxs)
    ]

    return {
        "image": str(image_path),
        "prediction": pred_label,
        "confidence": pred_conf,
        "topk": top,
        "device": str(device),
        "ckpt": str(ckpt_path),
        "image_size": image_size,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to an input image")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--class-map", type=str, default=str(DEFAULT_CLASS_MAP))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu or cuda")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    out = predict_image(
        args.image,
        ckpt_path=Path(args.ckpt),
        class_map_path=Path(args.class_map),
        image_size=args.image_size,
        device_str=args.device,
        topk=args.topk,
    )
    print(json.dumps(out, indent=2))
