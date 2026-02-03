import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')


repo_name = "speaker-type-classifier"          # GitHub repo folder name 
package_name = "speaker_type_classifier"       # Python import/package name 


list_of_files = [
    # GitHub / CI
    ".github/workflows/.gitkeep",
    ".github/ISSUE_TEMPLATE/.gitkeep",

    # src-layout package
    f"src/{package_name}/__init__.py",

    # Config + constants
    f"src/{package_name}/constant/__init__.py",
    f"src/{package_name}/constant/constants.py",

    # Entities (configs + artifacts)
    f"src/{package_name}/entity/__init__.py",
    f"src/{package_name}/entity/config_entity.py",
    f"src/{package_name}/entity/artifact_entity.py",

    # Configuration
    f"src/{package_name}/config/__init__.py",
    f"src/{package_name}/config/configuration.py",

    # Logging + exceptions
    f"src/{package_name}/logging/__init__.py",
    f"src/{package_name}/logging/logger.py",
    f"src/{package_name}/exception/__init__.py",
    f"src/{package_name}/exception/exception.py",

    # Utils
    f"src/{package_name}/utils/__init__.py",
    f"src/{package_name}/utils/common.py",
    f"src/{package_name}/utils/io_utils.py",
    f"src/{package_name}/utils/audio_utils.py",
    f"src/{package_name}/utils/hf_utils.py",

    # Components (pipeline stages)
    f"src/{package_name}/components/__init__.py",
    f"src/{package_name}/components/data_ingestion.py",
    f"src/{package_name}/components/data_validation.py",
    f"src/{package_name}/components/data_transformation.py",
    f"src/{package_name}/components/model_trainer.py",
    f"src/{package_name}/components/model_evaluation.py",
    f"src/{package_name}/components/model_pusher.py",

    # Pipelines (stage runners)
    f"src/{package_name}/pipeline/__init__.py",
    f"src/{package_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{package_name}/pipeline/stage_02_data_validation.py",
    f"src/{package_name}/pipeline/stage_03_data_transformation.py",
    f"src/{package_name}/pipeline/stage_04_model_trainer.py",
    f"src/{package_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{package_name}/pipeline/stage_06_model_pusher.py",

    # Configs
    "configs/config.yaml",
    "configs/params.yaml",

    # Artifacts (runtime outputs)
    "artifacts/runs/.gitkeep",

    # Data placeholders 
    "data/.gitkeep",
    "data/raw/.gitkeep",
    "data/external/.gitkeep",

    # Scripts
    "scripts/run_stage_01.sh",
    "scripts/run_all.sh",

    # Tests
    "tests/test_smoke.py",
    "tests/test_imports.py",

    # Top-level
    "README.md",        
    ".env",
    "requirements.txt",
]

def safe_touch(filepath: Path) -> None:
    """
    Create an empty file if missing.
    Do not overwrite existing non-empty files.
    """
    if not filepath.exists():
        filepath.touch()
        logging.info(f"Created file: {filepath}")
        return

    # If file exists but empty, keep it as is
    if filepath.is_file() and filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Ensured empty file exists: {filepath}")
        return

    logging.info(f"Exists, not modified: {filepath}")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Ensured directory: {path}")

def main():
    logging.info(f"Repo: {repo_name}")
    logging.info(f"Package: {package_name}")
    logging.info("Creating project architecture...")

    for fp in list_of_files:
        filepath = Path(fp)
        filedir = filepath.parent

        if str(filedir) != ".":
            ensure_dir(filedir)

        safe_touch(filepath)

    # Helpful extra dirs to have
    ensure_dir(Path("notebooks"))
    ensure_dir(Path("docs"))
    ensure_dir(Path("models"))
    ensure_dir(Path("logs"))

    logging.info("Project architecture created successfully.")

if __name__ == "__main__":
    main()
