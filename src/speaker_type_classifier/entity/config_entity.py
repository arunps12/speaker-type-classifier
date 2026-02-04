from dataclasses import dataclass
from pathlib import Path
from typing import List

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

@dataclass(frozen=True)
class DataValidationConfig:
    schema_path: Path
    run_root: Path
    output_dirname: str
    report_filename: str
    issues_filename: str
    fail_fast: bool = False


@dataclass(frozen=True)
class DataTransformationConfig:
    run_root: Path
    output_dirname: str

    feature_store_root: Path

    report_filename: str
    pointers_filename: str

    feature_types: List[str]

    target_sr: int
    max_seconds: float
    seed: int

    egemaps_dim: int

    wav2vec2_model_name: str
    hubert_model_name: str
    pooling: str

    device: str
    hf_batch_size: int
    hf_use_fp16: bool
