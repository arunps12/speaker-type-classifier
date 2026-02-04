from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.exception.exception import SpeakerTypeClassifierException
from src.speaker_type_classifier.entity.config_entity import DataTransformationConfig
from src.speaker_type_classifier.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
from src.speaker_type_classifier.utils.common import make_run_dir
from src.speaker_type_classifier.utils.io_utils import write_json
from src.speaker_type_classifier.utils.audio_utils import load_audio_mono, resample_audio, crop_or_pad
from src.speaker_type_classifier.utils.hf_utils import HFEmbedder
from src.speaker_type_classifier.constant.constants import LABEL2ID

logger = get_logger(__name__, run_name="stage_03_data_transformation")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _extract_egemaps_vector(audio_path: Path, egemaps_dim: int) -> np.ndarray:
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        feats = smile.process_file(str(audio_path))
        vec = feats.values.flatten().astype(np.float32)

        if vec.shape[0] != egemaps_dim:
            out = np.zeros(egemaps_dim, dtype=np.float32)
            m = min(egemaps_dim, vec.shape[0])
            out[:m] = vec[:m]
            return out

        return vec
    except Exception:
        return np.zeros(egemaps_dim, dtype=np.float32)


def _load_fixed_audio_16k(path: Path, target_sr: int, max_seconds: float) -> np.ndarray:
    x, sr = load_audio_mono(path)
    x = resample_audio(x, sr, target_sr)
    target_len = int(target_sr * max_seconds)
    x = crop_or_pad(x, target_len)
    return x


def _build_xy_egemaps(df: pd.DataFrame, egemaps_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for _, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        label = str(row["label"])
        y = LABEL2ID.get(label, -1)
        if y == -1:
            continue

        vec = _extract_egemaps_vector(audio_path, egemaps_dim)
        X_list.append(vec)
        y_list.append(int(y))

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def _build_xy_hf_batched(
    df: pd.DataFrame,
    cfg: DataTransformationConfig,
    hf_embedder: HFEmbedder,
) -> Tuple[np.ndarray, np.ndarray]:
    X_chunks: List[np.ndarray] = []
    y_all: List[int] = []

    batch_audio: List[np.ndarray] = []
    batch_y: List[int] = []

    bs = int(cfg.hf_batch_size)

    for _, row in df.iterrows():
        audio_path = Path(row["audio_path"])
        label = str(row["label"])
        y = LABEL2ID.get(label, -1)
        if y == -1:
            continue

        audio_16k = _load_fixed_audio_16k(audio_path, cfg.target_sr, cfg.max_seconds)
        batch_audio.append(audio_16k)
        batch_y.append(int(y))

        if len(batch_audio) >= bs:
            Xb = hf_embedder.embed_batch(batch_audio)  # (B,D)
            X_chunks.append(Xb)
            y_all.extend(batch_y)
            batch_audio, batch_y = [], []

    if batch_audio:
        Xb = hf_embedder.embed_batch(batch_audio)
        X_chunks.append(Xb)
        y_all.extend(batch_y)

    X = np.concatenate(X_chunks, axis=0).astype(np.float32) if X_chunks else np.zeros((0, 0), dtype=np.float32)
    y = np.array(y_all, dtype=np.int64)
    return X, y


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def run(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        try:
            logger.info("=== Data Transformation Started ===")

            train_df = pd.read_csv(ingestion_artifact.train_csv)
            val_df = pd.read_csv(ingestion_artifact.val_csv)

            # small artifacts dir in repo
            run_dir = make_run_dir(
                run_root=Path(self.config.run_root),
                stage_name=self.config.output_dirname,
                prefix="run",
            )
            ts_name = run_dir.name  # run_YYYYmmdd_HHMMSS

            report: Dict[str, Any] = {
                "stage": "data_transformation",
                "ingestion_run_dir": str(ingestion_artifact.run_dir),
                "run_dir": str(run_dir),
                "feature_store_root": str(self.config.feature_store_root),
                "feature_types": list(self.config.feature_types),
                "target_sr": self.config.target_sr,
                "max_seconds": self.config.max_seconds,
                "device": self.config.device,
                "hf_batch_size": self.config.hf_batch_size,
                "hf_use_fp16": self.config.hf_use_fp16,
                "pooling": self.config.pooling,
            }

            pointers: Dict[str, Any] = {}
            feature_runs: Dict[str, str] = {}

            for feature_type in self.config.feature_types:
                logger.info(f"--- Extracting feature_type={feature_type} ---")

                store_run_dir = Path(self.config.feature_store_root) / feature_type / ts_name
                _ensure_dir(store_run_dir)

                # Build X/y
                if feature_type == "egemaps":
                    X_train, y_train = _build_xy_egemaps(train_df, self.config.egemaps_dim)
                    X_val, y_val = _build_xy_egemaps(val_df, self.config.egemaps_dim)

                elif feature_type == "wav2vec2":
                    hf = HFEmbedder(
                        model_name=self.config.wav2vec2_model_name,
                        device=self.config.device,
                        pooling=self.config.pooling,
                        use_fp16=self.config.hf_use_fp16,
                    )
                    logger.info(f"wav2vec2 device_resolved={hf.device_resolved}")
                    X_train, y_train = _build_xy_hf_batched(train_df, self.config, hf)
                    X_val, y_val = _build_xy_hf_batched(val_df, self.config, hf)

                elif feature_type == "hubert":
                    hf = HFEmbedder(
                        model_name=self.config.hubert_model_name,
                        device=self.config.device,
                        pooling=self.config.pooling,
                        use_fp16=self.config.hf_use_fp16,
                    )
                    logger.info(f"hubert device_resolved={hf.device_resolved}")
                    X_train, y_train = _build_xy_hf_batched(train_df, self.config, hf)
                    X_val, y_val = _build_xy_hf_batched(val_df, self.config, hf)

                else:
                    raise ValueError(f"Unknown feature_type: {feature_type}")

                # Save arrays (big files on scratch)
                train_X_path = store_run_dir / "train_X.npy"
                train_y_path = store_run_dir / "train_y.npy"
                val_X_path = store_run_dir / "val_X.npy"
                val_y_path = store_run_dir / "val_y.npy"

                np.save(train_X_path, X_train)
                np.save(train_y_path, y_train)
                np.save(val_X_path, X_val)
                np.save(val_y_path, y_val)

                # Save index mapping (keeps exact row order)
                train_index_path = store_run_dir / "train_index.csv"
                val_index_path = store_run_dir / "val_index.csv"
                train_df[["audio_path", "label"]].to_csv(train_index_path, index=False)
                val_df[["audio_path", "label"]].to_csv(val_index_path, index=False)

                feature_runs[feature_type] = str(store_run_dir)
                pointers[feature_type] = {
                    "store_run_dir": str(store_run_dir),
                    "train_X": str(train_X_path),
                    "train_y": str(train_y_path),
                    "val_X": str(val_X_path),
                    "val_y": str(val_y_path),
                    "train_index": str(train_index_path),
                    "val_index": str(val_index_path),
                    "X_train_shape": list(X_train.shape),
                    "X_val_shape": list(X_val.shape),
                }

                report[f"{feature_type}_dims"] = int(X_train.shape[1])
                report[f"{feature_type}_train_rows"] = int(X_train.shape[0])
                report[f"{feature_type}_val_rows"] = int(X_val.shape[0])

                logger.info(f"Saved {feature_type} features -> {store_run_dir}")

            # small files in repo artifacts
            report_json = run_dir / self.config.report_filename
            pointers_json = run_dir / self.config.pointers_filename
            write_json(report_json, report)
            write_json(pointers_json, pointers)

            logger.info(f"Saved report:   {report_json}")
            logger.info(f"Saved pointers: {pointers_json}")
            logger.info("=== Data Transformation Completed ===")

            return DataTransformationArtifact(
                run_dir=run_dir,
                report_json=report_json,
                pointers_json=pointers_json,
                feature_runs=feature_runs,
            )

        except Exception as e:
            logger.exception("Data transformation failed.")
            raise SpeakerTypeClassifierException(e, sys)
