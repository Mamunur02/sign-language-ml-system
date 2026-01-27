from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "data"
    raw_data_dir: Path = data_root / "raw"
    processed_data_dir: Path = data_root / "processed"
    dataset_dir: Path = raw_data_dir / "asl_alphabet" / "train"


    # Data / training
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 2

    # Training hyperparams (used starting Day 2)
    learning_rate: float = 1e-3
    num_epochs: int = 5
    seed: int = 42

    # Model
    num_classes: int = 29  # set once dataset is chosen

    def ensure_dirs(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
