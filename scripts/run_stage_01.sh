#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " Stage 01: Data Ingestion"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"
echo "Python: $(which python)"
echo "Timestamp: $(date)"

if [ ! -f "configs/config.yaml" ]; then
  echo "ERROR: configs/config.yaml not found"
  exit 1
fi

python -m src.speaker_type_classifier.pipeline.stage_01_data_ingestion

echo "=========================================="
echo " Stage 01 completed successfully"
echo "=========================================="
