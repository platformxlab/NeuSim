#!/bin/bash

### Run some simple experiments to test the installation of NeuSim.

# stop on error
set -xe

export NPUSIM_HOME=$(cd "../../"; pwd)
export RESULTS_DIR="$NPUSIM_HOME/results"
export CONFIGS_PATH="$NPUSIM_HOME/configs"

export RAY_ADDRESS="http://127.0.0.1:8265"

# ray job submit prefix
function RAY_PREFIX() {
    # if $1 is set, use it as job_id, else use default
    if [ -n "$1" ]; then
        ray_prefix="ray job submit --working-dir ./ --runtime-env ./runtime_env.json --submission-id $1-$(date +'%F-%T') --"
    else
        ray_prefix="ray job submit --working-dir ./ --runtime-env ./runtime_env.json --"
    fi
    echo "$ray_prefix"
}


# launch performance simulation runs
# LLM training
$(RAY_PREFIX trace_llm_training) python run_sim.py \
    --output_dir="$RESULTS_DIR/raw" \
    --configs_path="$CONFIGS_PATH" \
    --models="llama3_1-405b" \
    --num_chips="1,64,1024,2048,4096" \
    --versions="4,5p" \
    --training_batch_sizes="32" \
    --workload=training --verbosity=-1 &
# LLM inference
$(RAY_PREFIX trace_llm_inference) python run_sim.py \
    --output_dir="$RESULTS_DIR/raw" \
    --configs_path="$CONFIGS_PATH" \
    --models="llama3_1-405b,deepseekv2-236b" \
    --num_chips="1,64,1024" \
    --versions="4,5p" \
    --inference_batch_sizes="1,8,16" \
    --workload=inference --verbosity=-1 &
# DLRM inference
$(RAY_PREFIX trace_dlrm_inference) python run_sim.py \
    --output_dir="$RESULTS_DIR/raw" \
    --configs_path="$CONFIGS_PATH" \
    --models="dlrm-l" \
    --num_chips="1,64,128" \
    --versions="4,5p" \
    --inference_batch_sizes="1,8,16" \
    --workload=inference --verbosity=-1 &
# DiT inference
$(RAY_PREFIX trace_dit_inference) python run_sim.py \
    --output_dir="$RESULTS_DIR/raw" \
    --configs_path="$CONFIGS_PATH" \
    --models="dit-xl" \
    --num_chips="1,4" \
    --versions="4,5p" \
    --inference_batch_sizes="1,8,16" \
    --workload=inference --verbosity=-1 &
# GLIGEN inference
$(RAY_PREFIX trace_gligen_inference) python run_sim.py \
    --output_dir="$RESULTS_DIR/raw" \
    --configs_path="$CONFIGS_PATH" \
    --models="gligen" \
    --num_chips="1,4" \
    --versions="4,5p" \
    --inference_batch_sizes="1,8,16" \
    --workload=inference --verbosity=-1 &

wait


# launch per op energy analysis runs
# LLMs
$(RAY_PREFIX energy_operator_llm_inference) python energy_operator_analysis_main.py --models="llama3_1-405b,deepseekv2-236b" --npu_versions="4,5p" --workload=inference --results_path="$RESULTS_DIR/raw" &
$(RAY_PREFIX energy_operator_llm_training) python energy_operator_analysis_main.py --models="llama3_1-405b" --npu_versions="4,5p" --workload=training --results_path="$RESULTS_DIR/raw" &
# DLRM
$(RAY_PREFIX energy_operator_dlrm_inference) python energy_operator_analysis_main.py --models="dlrm-l" --npu_versions="4,5p" --workload=inference --results_path="$RESULTS_DIR/raw" &
# DiT
$(RAY_PREFIX energy_operator_dit_inference) python energy_operator_analysis_main.py --models="dit-xl" --npu_versions="4,5p" --workload=inference --results_path="$RESULTS_DIR/raw" &
# GLIGEN
$(RAY_PREFIX energy_operator_gligen_inference) python energy_operator_analysis_main.py --models="gligen" --npu_versions="4,5p" --workload=inference --results_path="$RESULTS_DIR/raw" &

wait


# launch carbon analysis runs
pg_strategies=("NoPG")
util_factors=("0.6")
for PG_STRATEGY in "${pg_strategies[@]}"; do
    for UTIL_FACTOR in "${util_factors[@]}"; do
        CI="0.0624"
        # LLMs
        $(RAY_PREFIX carbon_analysis_llm_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="llama3_1-405b,deepseekv2-236b" --npu_versions="4,5p" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_None" &
        $(RAY_PREFIX carbon_analysis_llm_training_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="llama3_1-405b" --npu_versions="4,5p" --workload=training --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_None" &
        # DLRM
        $(RAY_PREFIX carbon_analysis_dlrm_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="dlrm-l" --npu_versions="4,5p" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_None" &
        # DiT
        $(RAY_PREFIX carbon_analysis_dit_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="dit-xl" --npu_versions="4,5p" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_None" &
        # GLIGEN
        $(RAY_PREFIX carbon_analysis_gligen_inference_${PG_STRATEGY}_${UTIL_FACTOR}_${CI}) python carbon_analysis_main.py --models="gligen" --npu_versions="4,5p" --workload=inference --carbon_intensity=$CI --utilization_factor=$UTIL_FACTOR --power_gating_strategy=$PG_STRATEGY --results_path="$RESULTS_DIR/raw_None" &
    done
done

wait


# launch SLO analysis runs
python slo_analysis_main.py \
    --results_path="$RESULTS_DIR/carbon_NoPG/dvfs_None/CI0.0624/UTIL0.6" \
    --output_path="$RESULTS_DIR/slo" \
    --workload=inference \
    --npu_versions="4,5p" \
    --models="llama3_1-405b,deepseekv2-236b,dlrm-l,dit-xl,gligen"
python slo_analysis_main.py \
    --results_path="$RESULTS_DIR/carbon_NoPG/dvfs_None/CI0.0624/UTIL0.6" \
    --output_path="$RESULTS_DIR/slo" \
    --workload=training \
    --npu_versions="4,5p" \
    --models="llama3_1-405b"

wait
