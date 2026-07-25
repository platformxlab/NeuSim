#!/usr/bin/env bash

# One-request end-to-end FleetSim -> EventSim -> NeuSim smoke test.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fleetsim_env.sh"

SMOKE_OUTPUT_DIR="${NEUSIM_RESULTS_DIR}/smoke"
mkdir -p "${SMOKE_OUTPUT_DIR}"

run_fleetsim \
    --model=llama3-8b \
    --system=Static \
    --request_pattern=synthetic \
    --synthetic_num_requests=1 \
    --synthetic_request_rate=1 \
    --synthetic_input_len=32 \
    --synthetic_output_len=4 \
    --chip_versions=4 \
    --static_vpod_allocation="${NEUSIM_CONFIGS_DIR}/fleetsim/static_tpuv4_llama3_8b_tp4.json" \
    --output_dir="${SMOKE_OUTPUT_DIR}" \
    >"${SMOKE_OUTPUT_DIR}/output.log" 2>&1

echo "FleetSim smoke test passed; results: ${SMOKE_OUTPUT_DIR}"
