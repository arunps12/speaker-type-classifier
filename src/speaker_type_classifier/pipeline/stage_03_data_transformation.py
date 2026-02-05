from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.components.data_transformation import DataTransformation
from src.speaker_type_classifier.config.configuration import ConfigurationManager
from src.speaker_type_classifier.entity.artifact_entity import DataIngestionArtifact
from src.speaker_type_classifier.utils.run_utils import load_latest_ingestion_artifact

logger = get_logger(__name__, run_name="stage_03_data_transformation")


class DataTransformationPipeline:
    def run(self, ingestion_artifact: DataIngestionArtifact):
        logger.info("Running Stage 03: Data Transformation")
        config = ConfigurationManager().get_data_transformation_config()
        artifact = DataTransformation(config).run(ingestion_artifact)
        logger.info(f"Stage 03 completed. Run dir: {artifact.run_dir}")
        return artifact


def run(ingestion_artifact: DataIngestionArtifact):
    return DataTransformationPipeline().run(ingestion_artifact)


def main():
    ingestion_artifact = load_latest_ingestion_artifact()
    run(ingestion_artifact)


if __name__ == "__main__":
    main()
