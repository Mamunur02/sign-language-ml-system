# src/inference/predict_hand_crop.py
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image

from src.models.cnn import SimpleCNN
from src.data.transforms import build_val_transforms
from src.preprocess.hand_crop import HandCropper


DEFAULT_CKPT = Path(os.getenv("CKPT_PATH", "artifacts/runs/latest/checkpoints/best.pt"))
DEFAULT_CLASS_MAP = Path(os.getenv("CLASS_MAP_PATH", "data/processed/metadata/class_to_idx.json"))
DEFAULT_IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", 128))


def load_class_mapping(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Class mapping not found: {path}. "
            f"Set CLASS_MAP_PATH to your class_to_idx.json."
        )
    class_to_idx = json.loads(path.read_text())
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def load_model(ckpt_path: Path, num_classes: int, device: torch.device) -> SimpleCNN:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = SimpleCNN(num_classes=num_classes).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint missing model_state_dict. Keys: {list(ckpt.keys())}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_image_with_hand_crop(
    image_path: str | Path,
    ckpt_path: Path = DEFAULT_CKPT,
    class_map_path: Path = DEFAULT_CLASS_MAP,
    image_size: int = DEFAULT_IMAGE_SIZE,
    device_str: str | None = None,
    topk: int = 5,
    mp_det_conf: float = 0.5,
    mp_pad: float = 0.20,
):
    image_path = Path(image_path)
    device = torch.device(device_str) if device_str else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    class_to_idx, idx_to_class = load_class_mapping(class_map_path)
    num_classes = len(class_to_idx)

    model = load_model(ckpt_path, num_classes=num_classes, device=device)
    tfm = build_val_transforms(image_size)

    img = Image.open(image_path).convert("RGB")

    cropper = HandCropper(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=mp_det_conf,
        min_tracking_confidence=0.5,
        pad=mp_pad,
    )

    try:
        cropped = cropper.crop_hand(img)
    finally:
        cropper.close()

    if cropped is None:
        return {
            "image": str(image_path),
            "error": "no_hand_detected",
            "device": str(device),
        }

    x = tfm(cropped).unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred_label = idx_to_class[pred_idx]
    pred_conf = float(probs[pred_idx].item())

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
        "class_map": str(class_map_path),
        "image_size": image_size,
        "num_classes": num_classes,
        "hand_crop": True,
    }


def batch_run(
    images_dir: Path,
    ckpt_path: Path,
    class_map_path: Path,
    image_size: int,
    device_str: str | None,
    topk: int,
):
    import pandas as pd

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    device = device_str or ("cuda" if torch.cuda.is_available() else "cpu")

    # warm-up
    _ = predict_image_with_hand_crop(
        image_paths[0],
        ckpt_path=ckpt_path,
        class_map_path=class_map_path,
        image_size=image_size,
        device_str=device,
        topk=topk,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    rows = []
    t0 = time.perf_counter()
    for p in image_paths:
        expected = p.stem  # expects A.jpg, del.jpg, space.jpg, nothing.jpg

        start = time.perf_counter()
        out = predict_image_with_hand_crop(
            p,
            ckpt_path=ckpt_path,
            class_map_path=class_map_path,
            image_size=image_size,
            device_str=device,
            topk=topk,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        if "error" in out:
            rows.append({
                "file": p.name,
                "expected": expected,
                "pred": None,
                "confidence": None,
                "correct": False,
                "error": out["error"],
                "latency_ms": (end - start) * 1000.0,
            })
            continue

        pred = out["prediction"]
        conf = out["confidence"]
        correct = (pred == expected)

        tk = out["topk"]
        top1 = f'{tk[0]["label"]}:{tk[0]["prob"]:.3f}' if len(tk) > 0 else ""
        top2 = f'{tk[1]["label"]}:{tk[1]["prob"]:.3f}' if len(tk) > 1 else ""

        rows.append({
            "file": p.name,
            "expected": expected,
            "pred": pred,
            "confidence": conf,
            "correct": correct,
            "error": "",
            "latency_ms": (end - start) * 1000.0,
            "top1": top1,
            "top2": top2,
        })

    t1 = time.perf_counter()
    df = pd.DataFrame(rows)

    total_s = t1 - t0
    acc = float(df["correct"].mean()) if len(df) else 0.0
    mean_ms = float(df["latency_ms"].mean()) if len(df) else float("nan")
    p95_ms = float(df["latency_ms"].quantile(0.95)) if len(df) else float("nan")
    throughput = len(df) / total_s if total_s > 0 else float("nan")

    summary = {
        "images": int(len(df)),
        "accuracy": acc,
        "total_s": total_s,
        "throughput_img_per_s": throughput,
        "latency_mean_ms": mean_ms,
        "latency_p95_ms": p95_ms,
        "device": device,
    }
    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--images-dir", type=str, default=None, help="Batch: directory of images")
    parser.add_argument("--ckpt", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--class-map", type=str, default=str(DEFAULT_CLASS_MAP))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--topk", type=int, default=5)

    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    class_map = Path(args.class_map)

    if args.images_dir:
        df, summary = batch_run(
            images_dir=Path(args.images_dir),
            ckpt_path=ckpt,
            class_map_path=class_map,
            image_size=args.image_size,
            device_str=args.device,
            topk=args.topk,
        )
        print(json.dumps(summary, indent=2))
        # show worst cases first if running locally
        print(df.sort_values(["correct", "confidence"], ascending=[True, True]).head(50).to_string(index=False))
        return

    if not args.image:
        raise ValueError("Provide --image or --images-dir")

    out = predict_image_with_hand_crop(
        args.image,
        ckpt_path=ckpt,
        class_map_path=class_map,
        image_size=args.image_size,
        device_str=args.device,
        topk=args.topk,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
