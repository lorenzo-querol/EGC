from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""

    name: str
    image_size: int
    in_channels: int
    batch_size: int
    num_classes: int
    data_dir: Path = Path("./data")
    num_workers: int = 1

    def __post_init__(self):
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")

        self.dataset_path = self.data_dir / self.name
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset {self.name} not found in {self.data_dir}")


@dataclass
class TrainConfig:
    """Configuration for training parameters."""

    seed: int
    num_epochs: int
    lr: float

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError(f"Number of epochs must be positive, got {self.num_epochs}")

        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")


@dataclass
class ActiveLearnerConfig:
    """Configuration for active learner parameters."""

    budget_ratio: float
    sampling_method: str

    def __post_init__(self):
        if self.budget_ratio <= 0 or self.budget_ratio >= 1:
            raise ValueError(f"Budget ratio must be in (0, 1), got {self.budget_ratio}")

        if self.sampling_method not in ["uncertainty", "random"]:
            raise ValueError(f"Sampling method must be 'uncertainty' or 'random', got {self.sampling_method}")
