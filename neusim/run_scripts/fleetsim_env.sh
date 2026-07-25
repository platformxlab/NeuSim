#!/usr/bin/env bash

# Shared, location-independent defaults for the FleetSim experiment launchers.
# Every value can be overridden by exporting it before invoking a launcher.

FLEETSIM_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEUSIM_REPO_ROOT="${NEUSIM_REPO_ROOT:-$(cd "${FLEETSIM_SCRIPT_DIR}/../.." && pwd)}"
NEUSIM_RESULTS_DIR="${NEUSIM_RESULTS_DIR:-${PWD}/results/fleetsim}"
NEUSIM_TRACES_DIR="${NEUSIM_TRACES_DIR:-${PWD}/traces/inference}"
NEUSIM_CONFIGS_DIR="${NEUSIM_CONFIGS_DIR:-${NEUSIM_REPO_ROOT}/configs}"
NEUSIM_REQUEST_CACHE_DIR="${NEUSIM_REQUEST_CACHE_DIR:-${NEUSIM_RESULTS_DIR}/request_lookup_cache}"
NEUSIM_BACKEND_CACHE_DIR="${NEUSIM_BACKEND_CACHE_DIR:-${NEUSIM_RESULTS_DIR}/.cache/npusim_backend}"
NEUSIM_PYTHON="${NEUSIM_PYTHON:-python}"

export NEUSIM_REPO_ROOT NEUSIM_RESULTS_DIR NEUSIM_TRACES_DIR
export NEUSIM_CONFIGS_DIR NEUSIM_REQUEST_CACHE_DIR NEUSIM_BACKEND_CACHE_DIR
export NEUSIM_PYTHON
export PYTHONPATH="${NEUSIM_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_fleetsim() {
    "${NEUSIM_PYTHON}" -m neusim.run_scripts.fleetsim_main \
        --configs_path="${NEUSIM_CONFIGS_DIR}" \
        --npusim_backend_cache_dir="${NEUSIM_BACKEND_CACHE_DIR}" \
        --request_results_cache_dir="${NEUSIM_REQUEST_CACHE_DIR}" \
        --traces_dir="${NEUSIM_TRACES_DIR}" \
        "$@"
}
