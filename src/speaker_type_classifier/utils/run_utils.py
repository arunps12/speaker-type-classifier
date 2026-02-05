from pathlib import Path
from src.speaker_type_classifier.entity.artifact_entity import DataIngestionArtifact


def load_latest_ingestion_artifact() -> DataIngestionArtifact:
    base = Path("artifacts/runs/data_ingestion")
    runs = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No data_ingestion runs found")

    run_dir = runs[0]
    return DataIngestionArtifact(
        run_dir=run_dir,
        train_csv=run_dir / "train.csv",
        val_csv=run_dir / "val.csv",
        metadata_json=run_dir / "metadata.json",
        n_total=-1,
        n_train=-1,
        n_val=-1,
    )
