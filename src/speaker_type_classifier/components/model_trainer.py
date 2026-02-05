from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
from xgboost import XGBClassifier

from src.speaker_type_classifier.exception.exception import SpeakerTypeClassifierException
from src.speaker_type_classifier.logging.logger import get_logger

from src.speaker_type_classifier.constant.constants import ID2LABEL
from src.speaker_type_classifier.entity.config_entity import ModelTrainerConfig
from src.speaker_type_classifier.entity.artifact_entity import ModelTrainerArtifact

from src.speaker_type_classifier.utils.common import make_run_dir
from src.speaker_type_classifier.utils.io_utils import read_json, write_json
from src.speaker_type_classifier.utils.ml_utils import (
    ensure_dir,
    find_latest_run_dir,
    resolve_feature_dir,
    resolve_run_id_from_run_dir_str,
    load_feature_pack,
    compute_multiclass_metrics,
)

logger = get_logger(__name__, run_name="model_trainer")


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, params: Dict[str, Any]):
        self.config = config
        self.params = params

    def _resolve_transformation_report(self) -> Dict[str, Any]:
        run_dir = find_latest_run_dir(
            stage_dir=self.config.transformation_stage_dir,
            pinned_run_id=self.config.pinned_transformation_run_id,
        )
        report_path = run_dir / self.config.transformation_report_filename
        if not report_path.exists():
            raise FileNotFoundError(f"transformation_report.json not found: {report_path}")
        return read_json(report_path)

    def _build_xgb(self) -> XGBClassifier:
        xgb_params = self.params.get("xgboost", {})
        num_classes = int(self.params.get("num_classes", 4))

        base = dict(
            objective=xgb_params.get("objective", "multi:softprob"),
            num_class=num_classes,
            eval_metric=xgb_params.get("eval_metric", "mlogloss"),
            tree_method=xgb_params.get("tree_method", "hist"),
            random_state=int(self.config.seed),
        )

        # Merge tuned params over base
        cfg = dict(base)
        cfg.update(xgb_params)

        # Device handling
        if bool(self.config.use_gpu):
            cfg.update(device="cuda")
        else:
            cfg.update(device="cpu", n_jobs=int(self.config.cpu_n_jobs))

        return XGBClassifier(**cfg)

    def run(self) -> ModelTrainerArtifact:
        try:
            report = self._resolve_transformation_report()
            transformation_run_id = resolve_run_id_from_run_dir_str(report["run_dir"])
            feature_store_root = report["feature_store_root"]

            feature_dir = resolve_feature_dir(
                feature_store_root=feature_store_root,
                feature_type=self.config.feature_type,
                run_id=transformation_run_id,
            )

            logger.info(f"Using transformation run: {transformation_run_id}")
            logger.info(f"Loading features from: {feature_dir}")

            Xtr, ytr, Xva, yva = load_feature_pack(feature_dir)

            logger.info(f"Train: X={Xtr.shape}, y={ytr.shape}")
            logger.info(f"Val:   X={Xva.shape}, y={yva.shape}")

            clf = self._build_xgb()

            t0 = time.time()
            clf.fit(Xtr, ytr, verbose=False)
            train_time_sec = time.time() - t0

            y_pred = clf.predict(Xva)
            metrics = compute_multiclass_metrics(y_true=yva, y_pred=y_pred, id2label=ID2LABEL)

            # Create run dir for this stage
            run_dir = make_run_dir(
                run_root=self.config.run_root,
                stage_name=self.config.output_dirname,
            )

            model_dir = ensure_dir(run_dir / self.config.model_subdir)
            metrics_dir = ensure_dir(run_dir / self.config.metrics_subdir)

            model_path = model_dir / self.config.model_filename
            metrics_path = metrics_dir / "metrics.json"

            # Save model + metrics
            joblib.dump(clf, model_path)

            out_metrics = {
                "stage": "model_trainer",
                "feature_type": self.config.feature_type,
                "transformation_run_id": transformation_run_id,
                "train_time_sec": float(train_time_sec),
                "metrics": metrics,
                "xgboost_params": self.params.get("model_trainer", {}).get("xgboost", {}),
            }
            write_json(metrics_path, out_metrics)

            logger.info(f"Saved model:   {model_path}")
            logger.info(f"Saved metrics: {metrics_path}")

            # Save a copy into ./models/
            if bool(self.config.save_to_models_dir):
                models_root = ensure_dir(self.config.models_root)
                dst = models_root / self.config.models_filename
                joblib.dump(clf, dst)
                logger.info(f"Saved model copy to: {dst}")

            return ModelTrainerArtifact(
                run_dir=run_dir,
                model_path=model_path,
                metrics_path=metrics_path,
                feature_type=self.config.feature_type,
                transformation_run_id=transformation_run_id,
            )

        except Exception as e:
            raise SpeakerTypeClassifierException(e, sys) from e
