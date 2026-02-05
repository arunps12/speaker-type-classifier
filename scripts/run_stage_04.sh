#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " Stage 04: Model Trainer"
echo "=========================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"
echo "Python: $(which python)"
echo "Timestamp: $(date)"

python -m src.speaker_type_classifier.pipeline.stage_04_model_trainer

echo "=========================================="
echo " Stage 04 completed successfully"
echo "=========================================="
