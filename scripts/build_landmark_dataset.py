from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.features.hand_landmarks import HandLandmarkExtractor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_class_index(root_dir: Path) -> dict[str, int]:
    class_names = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class folders found in {root_dir}")
    return {name: i for i, name in enumerate(class_names)}


def iter_images(root_dir: Path, class_to_idx: dict[str, int]):
    for class_name, y in class_to_idx.items():
        class_dir = root_dir / class_name
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True, help="ImageFolder root: class subfolders")
    parser.add_argument("--out-dir", type=str, default="artifacts/landmarks/asl_alphabet_v1")
    parser.add_argument("--task-model", type=str, default="artifacts/models/hand_landmarker.task")
    parser.add_argument("--max-per-class", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_index(dataset_dir)
    (out_dir / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2))

    extractor = HandLandmarkExtractor(task_model_path=args.task_model)

    X = []
    y = []
    paths = []
    skipped = 0

    per_class_count = {k: 0 for k in class_to_idx.keys()}

    try:
        for img_path, label in iter_images(dataset_dir, class_to_idx):
            class_name = img_path.parent.name

            if args.max_per_class and per_class_count[class_name] >= args.max_per_class:
                continue

            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                skipped += 1
                continue

            feats = extractor.extract(pil)
            if feats is None:
                skipped += 1
                continue

            X.append(feats.x)
            y.append(label)
            paths.append(str(img_path))
            per_class_count[class_name] += 1
    finally:
        extractor.close()

    X = np.stack(X, axis=0).astype(np.float32)  # (N,63)
    y = np.array(y, dtype=np.int64)             # (N,)
    paths = np.array(paths, dtype=object)

    np.savez_compressed(out_dir / "landmarks.npz", X=X, y=y, paths=paths)

    summary = {
        "dataset_dir": str(dataset_dir),
        "out_dir": str(out_dir),
        "num_samples": int(len(y)),
        "num_classes": int(len(class_to_idx)),
        "skipped": int(skipped),
        "max_per_class": int(args.max_per_class),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_dir / 'landmarks.npz'}")


if __name__ == "__main__":
    main()
