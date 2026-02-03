from src.speaker_type_classifier.logging.logger import get_logger
from src.speaker_type_classifier.components.data_ingestion import DataIngestion
from src.speaker_type_classifier.config.configuration import ConfigurationManager

logger = get_logger(__name__, run_name="stage_01_data_ingestion")


class DataIngestionPipeline:
    def run(self):
        logger.info("Running Stage 01: Data Ingestion")
        config = ConfigurationManager().get_data_ingestion_config()
        artifact = DataIngestion(config).run()
        logger.info(f"Stage 01 completed. Run dir: {artifact.run_dir}")
        return artifact
