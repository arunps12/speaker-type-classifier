from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.speaker_type_classifier.utils.common import read_yaml
from src.speaker_type_classifier.entity.config_entity import DataIngestionConfig, DataValidationConfig , DataTransformationConfig, ModelTrainerConfig
from src.speaker_type_classifier.constant.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH


class ConfigurationManager:
    """
    Loads configs/config.yaml and returns typed config objects for each pipeline stage.
    """

    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        self.config_filepath = Path(config_filepath)
        self.config: Dict[str, Any] = read_yaml(self.config_filepath)
        self.params: Dict[str, Any] = read_yaml(PARAMS_FILE_PATH)
    def _require(self, key: str) -> Any:
        if key not in self.config:
            raise KeyError(f"Missing required top-level key in config.yaml: '{key}'")
        return self.config[key]

    def get_data_ingestion_config(
        self,
        override_run_root: Optional[str] = None,
    ) -> DataIngestionConfig:
        """
        Return DataIngestionConfig built from config.yaml.
        override_run_root can be used for quick experiments.
        """
        cfg = self._require("data_ingestion")

        # mandatory keys
        manifest_path = Path(cfg["manifest_path"])
        run_root = Path(override_run_root) if override_run_root else Path(cfg["run_root"])

        return DataIngestionConfig(
            manifest_path=manifest_path,
            run_root=run_root,
            output_dirname=str(cfg.get("output_dirname", "data_ingestion")),
            train_filename=str(cfg.get("train_filename", "train.csv")),
            val_filename=str(cfg.get("val_filename", "val.csv")),
            metadata_filename=str(cfg.get("metadata_filename", "metadata.json")),
            val_size=float(cfg.get("val_size", 0.10)),
            seed=int(cfg.get("seed", 42)),
            stratify=bool(cfg.get("stratify", True)),
            drop_missing_audio=bool(cfg.get("drop_missing_audio", True)),
        )

    def as_dict(self) -> Dict[str, Any]:
        """
        Useful for debugging/logging or dumping config snapshots.
        """
        return self.config



    def get_data_validation_config(self) -> DataValidationConfig:
        cfg = self._require("data_validation")
        return DataValidationConfig(
            schema_path=Path(cfg["schema_path"]),
            run_root=Path(cfg.get("run_root", "artifacts/runs")),
            output_dirname=str(cfg.get("output_dirname", "data_validation")),
            report_filename=str(cfg.get("report_filename", "validation_report.json")),
            issues_filename=str(cfg.get("issues_filename", "validation_issues.csv")),
            fail_fast=bool(cfg.get("fail_fast", False)),
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        cfg = self._require("data_transformation")
        return DataTransformationConfig(
            run_root=Path(cfg.get("run_root", "artifacts/runs")),
            output_dirname=str(cfg.get("output_dirname", "data_transformation")),

            feature_store_root=Path(cfg["feature_store_root"]),

            report_filename=str(cfg.get("report_filename", "transformation_report.json")),
            pointers_filename=str(cfg.get("pointers_filename", "pointers.json")),

            feature_types=list(cfg.get("feature_types", ["egemaps"])),

            target_sr=int(cfg.get("target_sr", 16000)),
            max_seconds=float(cfg.get("max_seconds", 10.0)),
            seed=int(cfg.get("seed", 42)),

            egemaps_dim=int(cfg.get("egemaps_dim", 88)),

            wav2vec2_model_name=str(cfg.get("wav2vec2_model_name", "facebook/wav2vec2-base")),
            hubert_model_name=str(cfg.get("hubert_model_name", "facebook/hubert-base-ls960")),
            pooling=str(cfg.get("pooling", "mean")),

            device=str(cfg.get("device", "cuda")),
            hf_batch_size=int(cfg.get("hf_batch_size", 16)),
            hf_use_fp16=bool(cfg.get("hf_use_fp16", True)),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        cfg = self._require("model_trainer")

        return ModelTrainerConfig(
            run_root=Path(cfg.get("run_root", "artifacts/runs")),
            output_dirname=str(cfg.get("output_dirname", "model_trainer")),

            feature_type=str(cfg.get("feature_type", "egemaps")),

            transformation_stage_dir=Path(cfg.get("transformation_stage_dir", "artifacts/runs/data_transformation")),
            transformation_report_filename=str(cfg.get("transformation_report_filename", "transformation_report.json")),
            use_latest_transformation_run=bool(cfg.get("use_latest_transformation_run", True)),
            pinned_transformation_run_id=cfg.get("pinned_transformation_run_id", None),

            model_subdir=str(cfg.get("model_subdir", "model")),
            metrics_subdir=str(cfg.get("metrics_subdir", "metrics")),
            model_filename=str(cfg.get("model_filename", "model.pkl")),

            save_to_models_dir=bool(cfg.get("save_to_models_dir", True)),
            models_root=Path(cfg.get("models_root", "models")),
            models_filename=str(cfg.get("models_filename", "model.pkl")),

            use_gpu=bool(cfg.get("use_gpu", True)),
            cpu_n_jobs=int(cfg.get("cpu_n_jobs", 16)),
            seed=int(cfg.get("seed", 42)),
        )
    
    def get_model_trainer_params(self) -> Dict[str, Any]:
        """
        Return Model Trainer parameters from params.yaml.
        """
        if "model_trainer" not in self.params:
            raise KeyError("Missing 'model_trainer' section in params.yaml")

        return self.params["model_trainer"]