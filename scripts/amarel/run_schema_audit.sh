#!/usr/bin/env bash
set -euo pipefail
cd "/scratch/nt612/Github/Conditional Tail-Risk Pricing in U.S. Equities"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -x "$HOME/.conda/envs/ml_core/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/ml_core/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-/projects/community/miniconda3/bin/python}"
fi
"$PYTHON_BIN" -m pip install -e ".[dev,ml]" --quiet
"$PYTHON_BIN" -m ctrsdf.pipeline schema-audit --config configs/project.yaml
