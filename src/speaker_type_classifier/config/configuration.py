from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from src.speaker_type_classifier.utils.common import read_yaml
from src.speaker_type_classifier.entity.config_entity import DataIngestionConfig
from src.speaker_type_classifier.constant.constants import CONFIG_FILE_PATH


class ConfigurationManager:
    """
    Loads configs/config.yaml and returns typed config objects for each pipeline stage.
    """

    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        self.config_filepath = Path(config_filepath)
        self.config: Dict[str, Any] = read_yaml(self.config_filepath)

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
