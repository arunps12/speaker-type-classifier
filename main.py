from src.speaker_type_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.speaker_type_classifier.pipeline.stage_02_data_validation import DataValidationPipeline

if __name__ == "__main__":
    DataIngestionPipeline().run()

if __name__ == "__main__":
    ingestion_artifact = DataIngestionPipeline().run()
    validation_artifact = DataValidationPipeline().run(ingestion_artifact)

    if not validation_artifact.is_valid:
        raise SystemExit("Validation failed. Check validation_report.json and validation_issues.csv")
