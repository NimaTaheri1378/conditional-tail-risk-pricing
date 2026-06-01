#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/scratch/nt612/Github/Conditional Tail-Risk Pricing in U.S. Equities"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m ctrsdf.pipeline results --config configs/project.yaml
