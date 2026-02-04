from pathlib import Path

CONFIG_FILE_PATH = Path("configs/config.yaml")

ARTIFACTS_DIR = "artifacts"
RUNS_DIR = "artifacts/runs"

MASTER_FIELDS_REQUIRED = ("audio_path", "label")

LABEL2ID = {
    "adult_male": 0,
    "adult_female": 1,
    "child": 2,
    "background": 3,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}
