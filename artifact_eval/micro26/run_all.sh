#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

export DVFS_GA_EXACT_BATCH_SIZE="${DVFS_GA_EXACT_BATCH_SIZE-32}"
export DVFS_MS_CANDIDATE_BATCH_SIZE="${DVFS_MS_CANDIDATE_BATCH_SIZE-8}"
export DVFS_PARETO_BATCH_SIZE="${DVFS_PARETO_BATCH_SIZE-512}"
export DVFS_PARETO_MAX_INFLIGHT_BATCHES="${DVFS_PARETO_MAX_INFLIGHT_BATCHES-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "error: Python interpreter '$PYTHON_BIN' was not found; set PYTHON=/path/to/python" >&2
    exit 127
fi

exec "$PYTHON_BIN" "$SCRIPT_DIR/pipeline.py" "$@"
