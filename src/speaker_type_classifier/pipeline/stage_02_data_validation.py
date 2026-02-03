from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.components.data_validation import DataValidation
from src.speaker_type_classifier.config.configuration import ConfigurationManager
from src.speaker_type_classifier.entity.artifact_entity import DataIngestionArtifact

logger = get_logger(__name__, run_name="stage_02_data_validation")


class DataValidationPipeline:
    def run(self, ingestion_artifact: DataIngestionArtifact):
        logger.info("Running Stage 02: Data Validation")
        config = ConfigurationManager().get_data_validation_config()
        artifact = DataValidation(config).run(ingestion_artifact)
        logger.info(f"Stage 02 completed. Run dir: {artifact.run_dir} | is_valid={artifact.is_valid}")
        return artifact
