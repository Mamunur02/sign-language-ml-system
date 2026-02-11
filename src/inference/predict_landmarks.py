from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.features.hand_landmarks import HandLandmarkExtractor
from src.training.train_landmarks import MLP


def load_classifier(ckpt_path: Path, class_to_idx_path: Path, device: torch.device):
    class_to_idx = json.loads(class_to_idx_path.read_text())
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    ckpt = torch.load(ckpt_path, map_location=device)
    model = MLP(in_dim=63, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx_to_class


@torch.no_grad()
def predict_with_loaded(
    model: torch.nn.Module,
    extractor: HandLandmarkExtractor,
    idx_to_class: dict[int, str],
    image_path: Path,
    device: torch.device,
    topk: int = 5,
):
    pil = Image.open(image_path).convert("RGB")
    feats = extractor.extract(pil)
    if feats is None:
        return {"image": str(image_path), "error": "no_hand_detected"}

    x = torch.from_numpy(feats.x.astype(np.float32)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    pred = idx_to_class[pred_idx]
    conf = float(probs[pred_idx].item())

    k = min(topk, probs.numel())
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
    parser.add_argument("--task-model", type=str, default="artifacts/models/hand_landmarker.task")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    class_map = Path(args.class_to_idx)

    model, idx_to_class = load_classifier(ckpt, class_map, device)
    extractor = HandLandmarkExtractor(task_model_path=args.task_model)

    try:
        if args.images_dir:
            exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            paths = sorted([p for p in Path(args.images_dir).iterdir() if p.is_file() and p.suffix.lower() in exts])

            # warm-up
            if paths:
                _ = predict_with_loaded(model, extractor, idx_to_class, paths[0], device, topk=args.topk)

            rows = []
            t0 = time.perf_counter()
            for p in paths:
                start = time.perf_counter()
                out = predict_with_loaded(model, extractor, idx_to_class, p, device, topk=args.topk)
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

        out = predict_with_loaded(model, extractor, idx_to_class, Path(args.image), device, topk=args.topk)
        print(json.dumps(out, indent=2))
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
