from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class DataIngestionArtifact:
    run_dir: Path
    train_csv: Path
    val_csv: Path
    metadata_json: Path
    n_total: int
    n_train: int
    n_val: int

@dataclass(frozen=True)
class DataValidationArtifact:
    run_dir: Path
    report_json: Path
    issues_csv: Path
    is_valid: bool
    n_train: int
    n_val: int
