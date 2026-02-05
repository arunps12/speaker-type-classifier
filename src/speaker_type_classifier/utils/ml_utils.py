from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_feature_pack(feature_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expects:
      train_X.npy, train_y.npy, val_X.npy, val_y.npy
    """
    feature_dir = Path(feature_dir)

    Xtr = np.load(feature_dir / "train_X.npy")
    ytr = np.load(feature_dir / "train_y.npy")
    Xva = np.load(feature_dir / "val_X.npy")
    yva = np.load(feature_dir / "val_y.npy")
    return Xtr, ytr, Xva, yva


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Dict[int, str],
) -> Dict[str, Any]:
    """
    Returns:
      accuracy, uar_macro_recall, macro_f1, per_class(list)
    """
    label_ids = sorted(id2label.keys())
    label_names = [id2label[i] for i in label_ids]

    acc = float(accuracy_score(y_true, y_pred))

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )

    uar = float(np.mean(r))
    macro_f1 = float(np.mean(f1))

    per_class = []
    for lab_id, lab_name, s, pp, rr, ff in zip(label_ids, label_names, sup, p, r, f1):
        per_class.append(
            {
                "label_id": int(lab_id),
                "label": str(lab_name),
                "support": int(s),
                "precision": float(pp),
                "recall": float(rr),
                "f1": float(ff),
            }
        )

    return {
        "accuracy": acc,
        "uar_macro_recall": uar,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def find_latest_run_dir(stage_dir: Path, pinned_run_id: Optional[str] = None) -> Path:
    """
    Find latest run directory under: artifacts/runs/<stage>/run_YYYYMMDD_HHMMSS
    If pinned_run_id is provided, use it.
    """
    stage_dir = Path(stage_dir)

    if pinned_run_id:
        run_dir = stage_dir / pinned_run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Pinned run directory not found: {run_dir}")
        return run_dir

    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    candidates = [p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under: {stage_dir}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_run_id_from_run_dir_str(run_dir_str: str) -> str:
    """
    transformation_report.json has: "run_dir": "artifacts/runs/data_transformation/run_...."
    Extract the basename: run_YYYYMMDD_HHMMSS
    """
    return Path(run_dir_str).name


def resolve_feature_dir(feature_store_root: str, feature_type: str, run_id: str) -> Path:
    """
    Build feature directory:
      <feature_store_root>/<feature_type>/<run_id>
    """
    return Path(feature_store_root) / feature_type / run_id
