from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.exception.exception import SpeakerTypeClassifierException
from src.speaker_type_classifier.entity.config_entity import DataValidationConfig
from src.speaker_type_classifier.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.speaker_type_classifier.utils.common import make_run_dir
from src.speaker_type_classifier.utils.io_utils import write_json
from src.speaker_type_classifier.utils.validation_utils import load_schema

logger = get_logger(__name__, run_name="stage_02_data_validation")


def _add_issue(issues: List[Dict[str, Any]], split: str, row_idx: int, code: str, msg: str) -> None:
    issues.append({"split": split, "row_index": row_idx, "code": code, "message": msg})


def _validate_required_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    missing = [c for c in required if c not in df.columns]
    return missing


def _validate_labels(df: pd.DataFrame, allowed_labels: List[str]) -> List[Tuple[int, str]]:
    bad = []
    if "label" not in df.columns:
        return bad
    allowed = set(allowed_labels)
    for idx, v in df["label"].items():
        if pd.isna(v) or str(v) not in allowed:
            bad.append((idx, str(v)))
    return bad


def _validate_no_nulls(df: pd.DataFrame, cols: List[str]) -> List[Tuple[int, str]]:
    bad = []
    for c in cols:
        if c not in df.columns:
            continue
        null_mask = df[c].isna() | (df[c].astype(str).str.strip() == "")
        for idx in df[null_mask].index.tolist():
            bad.append((idx, c))
    return bad


def _validate_audio_paths_exist(df: pd.DataFrame) -> List[int]:
    bad_idx = []
    if "audio_path" not in df.columns:
        return bad_idx
    for idx, p in df["audio_path"].items():
        try:
            if pd.isna(p) or not Path(str(p)).exists():
                bad_idx.append(idx)
        except Exception:
            bad_idx.append(idx)
    return bad_idx


def _count_per_label(df: pd.DataFrame) -> Dict[str, int]:
    if "label" not in df.columns:
        return {}
    return df["label"].value_counts(dropna=False).to_dict()


class DataValidation:
    """
    Validates train/val CSVs produced by DataIngestion.
    """

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def run(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logger.info("=== Data Validation Started ===")

            schema = load_schema(Path(self.config.schema_path))
            required_columns = list(schema["required_columns"])
            allowed_labels = list(schema["dataset"]["allowed_labels"])

            checks = schema.get("checks", {})
            thresholds = schema.get("thresholds", {})

            train_path = Path(ingestion_artifact.train_csv)
            val_path = Path(ingestion_artifact.val_csv)

            if not train_path.exists():
                raise FileNotFoundError(f"train.csv not found: {train_path}")
            if not val_path.exists():
                raise FileNotFoundError(f"val.csv not found: {val_path}")

            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)

            issues: List[Dict[str, Any]] = []
            fatal_errors: List[str] = []

            # 1) required columns
            miss_train = _validate_required_columns(train_df, required_columns)
            miss_val = _validate_required_columns(val_df, required_columns)
            if miss_train:
                fatal_errors.append(f"train_missing_columns: {miss_train}")
            if miss_val:
                fatal_errors.append(f"val_missing_columns: {miss_val}")

            if fatal_errors and self.config.fail_fast:
                raise ValueError(" | ".join(fatal_errors))

            # 2) allowed labels
            for idx, v in _validate_labels(train_df, allowed_labels):
                _add_issue(issues, "train", idx, "bad_label", f"label={v} not in allowed_labels")
            for idx, v in _validate_labels(val_df, allowed_labels):
                _add_issue(issues, "val", idx, "bad_label", f"label={v} not in allowed_labels")

            # 3) no nulls in specific columns
            no_nulls_cols = list(checks.get("no_nulls_in", []))
            for idx, c in _validate_no_nulls(train_df, no_nulls_cols):
                _add_issue(issues, "train", idx, "null_or_empty", f"column={c} is null/empty")
            for idx, c in _validate_no_nulls(val_df, no_nulls_cols):
                _add_issue(issues, "val", idx, "null_or_empty", f"column={c} is null/empty")

            # 4) audio_path exists
            if bool(checks.get("audio_path_must_exist", False)):
                for idx in _validate_audio_paths_exist(train_df):
                    _add_issue(issues, "train", idx, "missing_audio", "audio_path does not exist")
                for idx in _validate_audio_paths_exist(val_df):
                    _add_issue(issues, "val", idx, "missing_audio", "audio_path does not exist")

            # 5) duplicates across splits
            if bool(checks.get("no_duplicates_audio_path", True)) and "audio_path" in train_df.columns and "audio_path" in val_df.columns:
                train_set = set(train_df["audio_path"].astype(str))
                val_set = set(val_df["audio_path"].astype(str))
                overlap = train_set.intersection(val_set)
                if overlap:
                    # row-level marking (first few, to avoid huge logs)
                    sample = list(overlap)[:50]
                    for p in sample:
                        _add_issue(issues, "both", -1, "train_val_overlap", f"audio_path duplicated across splits: {p}")
                    fatal_errors.append(f"train_val_overlap_count={len(overlap)}")

            # 6) thresholds on counts
            train_counts = _count_per_label(train_df)
            val_counts = _count_per_label(val_df)

            min_train = int(thresholds.get("min_rows_per_class_train", 0))
            min_val = int(thresholds.get("min_rows_per_class_val", 0))

            for lab in allowed_labels:
                if train_counts.get(lab, 0) < min_train:
                    fatal_errors.append(f"train_class_too_small:{lab}={train_counts.get(lab,0)} < {min_train}")
                if val_counts.get(lab, 0) < min_val:
                    fatal_errors.append(f"val_class_too_small:{lab}={val_counts.get(lab,0)} < {min_val}")

            # 7) require all labels present in each split
            if bool(checks.get("require_all_labels_in_each_split", False)):
                for lab in allowed_labels:
                    if lab not in train_counts:
                        fatal_errors.append(f"train_missing_label:{lab}")
                    if lab not in val_counts:
                        fatal_errors.append(f"val_missing_label:{lab}")

            # Save outputs
            run_dir = make_run_dir(
                run_root=Path(self.config.run_root),
                stage_name=self.config.output_dirname,
                prefix="run",
            )
            report_json = run_dir / self.config.report_filename
            issues_csv = run_dir / self.config.issues_filename

            issues_df = pd.DataFrame(issues)
            issues_df.to_csv(issues_csv, index=False)

            is_valid = (len(fatal_errors) == 0) and (len(issues) == 0 or True)
            
            report = {
                "stage": "data_validation",
                "schema_path": str(self.config.schema_path),
                "ingestion_run_dir": str(ingestion_artifact.run_dir),
                "train_csv": str(train_path),
                "val_csv": str(val_path),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "train_label_counts": train_counts,
                "val_label_counts": val_counts,
                "n_issues": int(len(issues_df)),
                "fatal_errors": fatal_errors,
                "is_valid": bool(is_valid),
            }
            write_json(report_json, report)

            logger.info(f"Saved validation report: {report_json}")
            logger.info(f"Saved issues CSV:        {issues_csv}")
            logger.info(f"Validation status: is_valid={is_valid}")

            logger.info("=== Data Validation Completed ===")

            return DataValidationArtifact(
                run_dir=run_dir,
                report_json=report_json,
                issues_csv=issues_csv,
                is_valid=is_valid,
                n_train=len(train_df),
                n_val=len(val_df),
            )

        except Exception as e:
            logger.exception("Data validation failed.")
            raise SpeakerTypeClassifierException(e, sys)
