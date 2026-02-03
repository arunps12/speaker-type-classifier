from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.exception.exception import SpeakerTypeClassifierException
from src.speaker_type_classifier.entity.config_entity import DataIngestionConfig
from src.speaker_type_classifier.entity.artifact_entity import DataIngestionArtifact
from src.speaker_type_classifier.utils.common import make_run_dir
from src.speaker_type_classifier.utils.io_utils import read_jsonl, write_json
from src.speaker_type_classifier.constant.constants import MASTER_FIELDS_REQUIRED

logger = get_logger(__name__, run_name="stage_01_data_ingestion")


def _validate_row(obj: Dict[str, Any]) -> Tuple[bool, str]:
    for k in MASTER_FIELDS_REQUIRED:
        if k not in obj or obj[k] in (None, ""):
            return False, f"missing_required_field:{k}"
    return True, ""


def _jsonl_to_df(
    manifest_path: Path,
    drop_missing_audio: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    bad: Dict[str, int] = {}

    for obj in read_jsonl(manifest_path):
        ok, reason = _validate_row(obj)
        if not ok:
            bad[reason] = bad.get(reason, 0) + 1
            continue

        audio_path = Path(obj["audio_path"])
        if drop_missing_audio and not audio_path.exists():
            bad["missing_audio_path"] = bad.get("missing_audio_path", 0) + 1
            continue

        # Flatten: keep meta as JSON string for simplicity
        meta = obj.get("meta", None)
        if isinstance(meta, (dict, list)):
            obj["meta"] = json.dumps(meta, ensure_ascii=False)

        rows.append(obj)

    df = pd.DataFrame(rows)

    logger.info(f"Loaded manifest: {manifest_path}")
    logger.info(f"Rows kept: {len(df)}")
    if bad:
        logger.warning(f"Rows dropped summary: {bad}")

    # normalize speaker_id None -> empty
    if "speaker_id" in df.columns:
        df["speaker_id"] = df["speaker_id"].replace({None: ""})

    return df


def _stratified_split(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
    stratify: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < val_size < 1.0):
        raise ValueError(f"val_size must be in (0,1), got {val_size}")

    if df.empty:
        raise ValueError("Manifest dataframe is empty after filtering.")

    if stratify:
        # manual stratified split without sklearn
        parts_train = []
        parts_val = []

        rng = pd.util.hash_pandas_object(df["label"], index=False).astype("int64")
        # shuffle deterministically by seed + hash trick
        df = df.assign(_rand=(rng + seed) % 10_000_000).sort_values("_rand").drop(columns=["_rand"])

        for label, g in df.groupby("label", sort=False):
            n = len(g)
            n_val = max(1, int(round(n * val_size))) if n > 1 else 0
            g_val = g.iloc[:n_val]
            g_train = g.iloc[n_val:]
            parts_val.append(g_val)
            parts_train.append(g_train)

        train_df = pd.concat(parts_train, ignore_index=True)
        val_df = pd.concat(parts_val, ignore_index=True)

        # If tiny classes created empty train, move one back
        if train_df.empty and len(df) > 1:
            # fallback: simple split
            split_idx = int(round(len(df) * (1 - val_size)))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()

    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(round(len(df) * (1 - val_size)))
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

    return train_df, val_df


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def run(self) -> DataIngestionArtifact:
        try:
            logger.info("=== Data Ingestion Started ===")

            manifest_path = Path(self.config.manifest_path)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest not found: {manifest_path}")

            df = _jsonl_to_df(
                manifest_path=manifest_path,
                drop_missing_audio=self.config.drop_missing_audio,
            )

            train_df, val_df = _stratified_split(
                df=df,
                val_size=self.config.val_size,
                seed=self.config.seed,
                stratify=self.config.stratify,
            )

            run_dir = make_run_dir(
                run_root=Path(self.config.run_root),
                stage_name=self.config.output_dirname,
                prefix="run",
            )

            train_csv = run_dir / self.config.train_filename
            val_csv = run_dir / self.config.val_filename
            metadata_json = run_dir / self.config.metadata_filename

            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)

            meta = {
                "stage": "data_ingestion",
                "manifest_path": str(manifest_path),
                "run_dir": str(run_dir),
                "val_size": self.config.val_size,
                "seed": self.config.seed,
                "stratify": self.config.stratify,
                "drop_missing_audio": self.config.drop_missing_audio,
                "n_total": int(len(df)),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "label_counts_total": df["label"].value_counts().to_dict() if "label" in df.columns else {},
                "label_counts_train": train_df["label"].value_counts().to_dict() if "label" in train_df.columns else {},
                "label_counts_val": val_df["label"].value_counts().to_dict() if "label" in val_df.columns else {},
                "columns": list(df.columns),
            }
            write_json(metadata_json, meta)

            logger.info(f"Saved train CSV: {train_csv}")
            logger.info(f"Saved val CSV:   {val_csv}")
            logger.info(f"Saved metadata:  {metadata_json}")
            logger.info("=== Data Ingestion Completed ===")

            return DataIngestionArtifact(
                run_dir=run_dir,
                train_csv=train_csv,
                val_csv=val_csv,
                metadata_json=metadata_json,
                n_total=len(df),
                n_train=len(train_df),
                n_val=len(val_df),
            )

        except Exception as e:
            logger.exception("Data ingestion failed.")
            raise SpeakerTypeClassifierException(e, sys)
