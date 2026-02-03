from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    run_dir: Path
    train_csv: Path
    val_csv: Path
    metadata_json: Path
    n_total: int
    n_train: int
    n_val: int
