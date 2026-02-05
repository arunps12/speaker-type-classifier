from __future__ import annotations

import sys

from src.speaker_type_classifier.config.configuration import ConfigurationManager
from src.speaker_type_classifier.components.model_trainer import ModelTrainer
from src.speaker_type_classifier.exception.exception import SpeakerTypeClassifierException
from src.speaker_type_classifier.logging.logger import get_logger

logger = get_logger(__name__, run_name="stage_04_model_trainer")


class ModelTrainerPipeline:
    def run(self):
        """
        Stage 04 does not require ingestion artifact directly.
        It loads the latest transformation_report.json internally
        (via ModelTrainer logic).
        """
        try:
            logger.info("Running Stage 04: Model Trainer")

            cm = ConfigurationManager()
            trainer_cfg = cm.get_model_trainer_config()
            trainer_params = cm.get_model_trainer_params()

            trainer = ModelTrainer(config=trainer_cfg, params=trainer_params)
            artifact = trainer.run()

            logger.info(f"Stage 04 completed. Run dir: {artifact.run_dir}")
            logger.info(f"Model:   {artifact.model_path}")
            logger.info(f"Metrics: {artifact.metrics_path}")

            return artifact

        except Exception as e:
            raise SpeakerTypeClassifierException(e, sys) from e


def run():
    return ModelTrainerPipeline().run()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
