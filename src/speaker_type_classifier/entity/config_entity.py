from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    manifest_path: Path
    run_root: Path
    output_dirname: str
    train_filename: str
    val_filename: str
    metadata_filename: str
    val_size: float
    seed: int
    stratify: bool
    drop_missing_audio: bool
