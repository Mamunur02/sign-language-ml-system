import os
from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class TrainConfig:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "data"
    raw_data_dir: Path = data_root / "raw"
    processed_data_dir: Path = data_root / "processed"

    # Default local dataset path
    dataset_dir: Path = field(default_factory=lambda: Path(os.getenv("DATASET_DIR", "")))

    image_size: int = int(os.getenv("IMAGE_SIZE", 128))
    batch_size: int = int(os.getenv("BATCH_SIZE", 32))
    num_workers: int = int(os.getenv("NUM_WORKERS", 2))

    learning_rate: float = float(os.getenv("LR", 1e-3))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", 5))
    seed: int = int(os.getenv("SEED", 42))

    # Let training set this dynamically
    num_classes: int = 0

    def ensure_dirs(self) -> None:
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def with_overrides(self) -> "TrainConfig":
        ds = os.getenv("DATASET_DIR")
        if ds:
            return dataclass_replace(self, dataset_dir=Path(ds))
        return self

    def __post_init__(self) -> None:
        if not self.dataset_dir or str(self.dataset_dir).strip() == "":
            default_path = self.raw_data_dir / "asl_alphabet" / "train"
            object.__setattr__(self, "dataset_dir", default_path)


def dataclass_replace(cfg: TrainConfig, **kwargs):
    from dataclasses import replace
    return replace(cfg, **kwargs)
