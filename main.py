from src.speaker_type_classifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.speaker_type_classifier.pipeline.stage_02_data_validation import DataValidationPipeline
from src.speaker_type_classifier.pipeline.stage_03_data_transformation import DataTransformationPipeline


def main():
    # Stage 01: Ingestion
    ingestion_artifact = DataIngestionPipeline().run()

    # Stage 02: Validation
    validation_artifact = DataValidationPipeline().run(ingestion_artifact)

    if not validation_artifact.is_valid:
        raise SystemExit(
            "Validation failed. Check validation_report.json and validation_issues.csv"
        )

    # Stage 03: Transformation (feature extraction to scratch feature store)
    transformation_artifact = DataTransformationPipeline().run(ingestion_artifact)

    print("\n Pipeline finished successfully")
    print("Ingestion run dir:       ", ingestion_artifact.run_dir)
    print("Validation run dir:      ", validation_artifact.run_dir)
    print("Transformation run dir:  ", transformation_artifact.run_dir)
    print("Feature store runs:      ", transformation_artifact.feature_runs)


if __name__ == "__main__":
    main()
