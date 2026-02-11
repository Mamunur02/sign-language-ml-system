from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.features.hand_landmarks import HandLandmarkExtractor
from src.training.train_landmarks import MLP  # reuse model definition


@torch.no_grad()
def predict_one(
    image_path: Path,
    ckpt_path: Path,
    class_to_idx_path: Path,
    task_model_path: Path = Path("artifacts/models/hand_landmarker.task"),
    device_str: str | None = None,
    topk: int = 5,
):
    class_to_idx = json.loads(class_to_idx_path.read_text())
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = MLP(in_dim=63, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    extractor = HandLandmarkExtractor(task_model_path=task_model_path)
    try:
        pil = Image.open(image_path).convert("RGB")
        feats = extractor.extract(pil)
    finally:
        extractor.close()

    if feats is None:
        return {"image": str(image_path), "error": "no_hand_detected"}

    x = torch.from_numpy(feats.x.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred = idx_to_class[pred_idx]
    conf = float(probs[pred_idx].item())

    k = min(topk, num_classes)
    vals, idxs = torch.topk(probs, k=k)
    top = [{"label": idx_to_class[int(i.item())], "prob": float(v.item())} for v, i in zip(vals, idxs)]

    return {"image": str(image_path), "prediction": pred, "confidence": conf, "topk": top}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--class-to-idx", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    class_map = Path(args.class_to_idx)

    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted([p for p in Path(args.images_dir).iterdir() if p.is_file() and p.suffix.lower() in exts])
        rows = []
        t0 = time.perf_counter()
        for p in paths:
            start = time.perf_counter()
            out = predict_one(p, ckpt, class_map, device_str=args.device, topk=args.topk)
            end = time.perf_counter()
            out["latency_ms"] = (end - start) * 1000.0
            out["expected"] = p.stem
            out["correct"] = (out.get("prediction") == out["expected"]) if "error" not in out else False
            rows.append(out)
        t1 = time.perf_counter()
        acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
        print(f"Images: {len(rows)} | Accuracy: {acc*100:.2f}% | Total: {t1-t0:.3f}s")
        for r in rows:
            if not r["correct"]:
                print(r)
        return

    if not args.image:
        raise ValueError("Provide --image or --images-dir")

    out = predict_one(Path(args.image), ckpt, class_map, device_str=args.device, topk=args.topk)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
