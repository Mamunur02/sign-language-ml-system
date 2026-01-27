from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

import json

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


class ImageFolderDataset(Dataset):
    """
    Simple image classification dataset where:
      root/
        class_a/xxx.png
        class_b/yyy.jpg
    """

    def __init__(self, root_dir: str | Path, transform=None) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.transform = transform
        self.class_to_idx = self._build_class_index(self.root_dir)
        metadata_dir = Path("data/processed/metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = metadata_dir / "class_to_idx.json"
        if not mapping_path.exists():
            with open(mapping_path, "w") as f:
                json.dump(self.class_to_idx, f, indent=2)

        self.samples = self._gather_samples(self.root_dir, self.class_to_idx)

        if len(self.samples) == 0:
            raise ValueError(
                f"No image files found under {self.root_dir}. "
                f"Expected class subfolders with images."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[object, int]:
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, sample.label

    @staticmethod
    def _build_class_index(root_dir: Path) -> Dict[str, int]:
        class_names = sorted([p.name for p in root_dir.iterdir() if p.is_dir()])
        if not class_names:
            raise ValueError(
                f"No class subfolders found in {root_dir}. "
                f"Create folders like A/, B/, C/..."
            )
        return {name: i for i, name in enumerate(class_names)}

    @staticmethod
    def _gather_samples(root_dir: Path, class_to_idx: Dict[str, int]) -> List[Sample]:
        samples: List[Sample] = []
        for class_name, label in class_to_idx.items():
            class_dir = root_dir / class_name
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    samples.append(Sample(path=p, label=label))
        return samples
