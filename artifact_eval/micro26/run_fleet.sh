#!/usr/bin/env bash

# Shared fresh static-fleet reproduction for MICRO'26 Figures 5 and 22.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${NEUSIM_PYTHON:-${PYTHON:-python}}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/experiments/run_fleet.py" \
    "$@"
