#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " Running full pipeline: Stage 01 → Stage 04"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"
echo "Python: $(which python)"
echo "Timestamp: $(date)"

# Sanity checks
if [ ! -f "configs/config.yaml" ]; then
  echo "ERROR: configs/config.yaml not found"
  exit 1
fi

if [ ! -f "configs/schema.yaml" ]; then
  echo "ERROR: configs/schema.yaml not found"
  exit 1
fi

if [ ! -f "configs/params.yaml" ]; then
  echo "ERROR: configs/params.yaml not found"
  exit 1
fi

# ------------------------
# Stage 01
# ------------------------
echo ""
echo ">>> Stage 01: Data Ingestion"
python -m src.speaker_type_classifier.pipeline.stage_01_data_ingestion

# ------------------------
# Stage 02
# ------------------------
echo ""
echo ">>> Stage 02: Data Validation"
python -m src.speaker_type_classifier.pipeline.stage_02_data_validation

# ------------------------
# Stage 03
# ------------------------
echo ""
echo ">>> Stage 03: Data Transformation"
python -m src.speaker_type_classifier.pipeline.stage_03_data_transformation

# ------------------------
# Stage 04
# ------------------------
echo ""
echo ">>> Stage 04: Model Trainer"
python -m src.speaker_type_classifier.pipeline.stage_04_model_trainer

echo ""
echo "=========================================="
echo " Pipeline completed successfully"
echo "=========================================="
