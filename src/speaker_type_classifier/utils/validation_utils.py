from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from src.speaker_type_classifier.utils.common import read_yaml


def load_schema(schema_path: Path) -> Dict[str, Any]:
    schema = read_yaml(schema_path)

    if "required_columns" not in schema:
        raise KeyError("schema.yaml missing 'required_columns'")
    if "dataset" not in schema or "allowed_labels" not in schema["dataset"]:
        raise KeyError("schema.yaml missing 'dataset.allowed_labels'")
    if "checks" not in schema:
        schema["checks"] = {}
    if "thresholds" not in schema:
        schema["thresholds"] = {}

    return schema
