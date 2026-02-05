from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

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

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DataTransformationArtifact:
    run_dir: Path
    report_json: Path
    pointers_json: Path
    feature_runs: Dict[str, str]  # feature_type -> feature_store_run_dir


@dataclass(frozen=True)
class ModelTrainerArtifact:
    run_dir: Path
    model_path: Path
    metrics_path: Path
    feature_type: str
    transformation_run_id: str
