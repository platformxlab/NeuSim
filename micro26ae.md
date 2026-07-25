# NeuScale MICRO 2026 artifact evaluation

This branch contains the artifact for the MICRO'26 paper "Auto-Scaling Heterogeneous Neural Processing Units for Energy and Cost-Efficient LLM Serving."
It contains an NPU cluster simulator for LLM serving workloads built on top of the NeuSim framework. The cluster simulator supports heterogeneous NPU instances and auto-scaling.

The general NPU cluster simulator interface is documented in
[`neusim/fleetsim/README.md`](neusim/fleetsim/README.md).
A sample input trace and automated launcher scripts are provided for the experiments in the MICRO'26 paper.

Run all commands below from the repository root.

## Hardware and Software Requirements

The artifact runs on any Linux x86 machine with
Python 3.12.2 or later; Ubuntu 24.04 is recommended. `conda` and `pip` are used for managing the Python environment and dependencies.
We recommend 24 or more physical CPU cores, 32 GiB of RAM (64 GiB preferred), and
20 GiB of free disk space.

The complete set of experiments in the paper rely on the full traces from the [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset), which can take more than one week to finish. This artifact uses a three-hour sample trace (truncated and downsampled from the original Azure Code trace) for validating the simulator functionalities.

## Installation

Create an isolated `conda` environment and install NeuSim with its test dependencies:

```bash
conda create --name neuscale-ae python=3.12.2
conda activate neuscale-ae
pip install -e ".[dev]"
```

Clone the published input archives into a sibling directory:

```bash
git clone --depth 1 https://github.com/XZman/micro26ae_supplementary_files \
  ../micro26ae_supplementary_files
printf '%s\n' \
  '4bd1a1b0715d894582370a4238195afc2a518a3fbaa0d5ff60ba23bb135fb427  AzureLLMInferenceTrace_code_3h_sampled.zip' \
  '30f0989dddb13c58c7129f1a92521403495344b3f4cb9eb90867df57b506133e  request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip' | \
  (cd ../micro26ae_supplementary_files && sha256sum -c -)
```

JSON hardware, model, system, and FleetSim configurations are under the
repository-level [`configs/`](configs/) directory.

## Experiment Workflow

### 1. Extract and validate the supplied inputs

The supplementary repository contains the three-hour Azure trace and a compact optimal-configuration cache for DeepSeekV3-671B, v5p/v6e,
prefill/decode, and energy/monetary objectives. Extract each archive to the
path shown below; the trace CSV is at the root of its ZIP, while the cache ZIP
already contains its required top-level directory.

```bash
unzip ../micro26ae_supplementary_files/AzureLLMInferenceTrace_code_3h_sampled.zip \
  -d artifact_eval/micro26/data
unzip ../micro26ae_supplementary_files/request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip \
  -d artifact_eval/micro26
(cd artifact_eval/micro26/data && sha256sum -c SHA256SUMS)
python -m neusim.run_scripts.prepare_micro26ae_sample_cache --validate-only
```

The extracted cache is
`artifact_eval/micro26/request_lookup_cache_deepseekv3_azure_3h/`.

### 2. Run functional checks

First, run a smoke test on the full simulator path with one synthetic static
request:

```bash
bash neusim/run_scripts/run_fleetsim_smoke.sh
```

Then, run 100 requests per setting as a bounded check:

```bash
python -m neusim.run_scripts.run_micro26ae_sample --list
python -m neusim.run_scripts.run_micro26ae_sample --max-requests 100
```

### 3. Run with the three-hour sample trace

```bash
python -m neusim.run_scripts.run_micro26ae_sample
```

The launcher runs energy and monetary jobs concurrently (`--jobs=2`).
Use `--force` to rerun it or `--n-cpu N` to select the Ideal baseline's worker count.

Results are written under `artifact_eval/micro26/results/`. Each setting has
`request_trace.csv`, `stats.json`, `run.log`, and `run_contract.json`; the last
file fingerprints the exact command, trace, cache manifest, and configuration
inputs used for safe resume. Event-driven systems also emit vPod
lifecycle/reconfiguration traces and a final checkpoint. The top-level
`ae_run_manifest.json` records every command, parameter, output directory, and
completion state.

`stats.json` summarizes request and sequence counts, TTFT/TPOT, arrival and
completion throughput, and prefill/decode/total tokens per joule and per dollar.
`request_trace.csv` has one completed request per row with the following schema.
All timestamps and latency fields ending in `_ns` are simulation nanoseconds.

| Column | Meaning |
| --- | --- |
| `request_id` | Unique request identifier. |
| `input_seqlen` | Padded input/context length used by the simulation. |
| `output_seqlen` | Padded output length used by the simulation; token 1 is emitted by prefill. |
| `enqueue_timestamp` | Request arrival time on the normalized simulation clock. |
| `prefill_start_timestamp` | Time prefill execution starts. |
| `prefill_end_timestamp` | Time prefill completes and the first token is emitted. |
| `decode_start_timestamp` | Time decode execution starts. |
| `decode_end_timestamp` | Time the final output token completes. |
| `prefill_queuing_delay_ns` | `prefill_start_timestamp - enqueue_timestamp`. |
| `prefill_latency_ns` | Prefill execution time, from prefill start to end. |
| `TTFT_ns` | Time to first token, from enqueue to prefill end. |
| `decode_queuing_delay_per_iteration_ns` | `TPOT_ns - ideal_TPOT_ns`; decode queuing overhead relative to the last decode batch/configuration. |
| `TPOT_ns` | Average inter-token time for output tokens 2…N, including prefill-to-decode waiting. |
| `ideal_TTFT_ns` | Contention-free prefill latency for the selected configuration. |
| `ideal_TPOT_ns` | Contention-free per-token decode latency for the last decode batch/configuration. |
| `prefill_energy_J` | Total prefill energy for the request, in joules. |
| `decode_energy_per_token_J` | Mean decode energy per token over the `output_seqlen - 1` decode iterations. |
| `prefill_cost_dollars` | Total prefill monetary cost for the request. |
| `decode_cost_per_token_dollars` | Mean monetary cost per decode token over tokens 2…N. |
| `config_prefill_npu_type` | NPU version used for prefill. |
| `config_prefill_input_seqlen` | Padded input length of the prefill configuration, when recorded. |
| `config_prefill_batch_size` | Actual prefill batch size. |
| `config_prefill_num_chips` | Number of NPU chips in the prefill vPod. |
| `config_prefill_pcfg` | Prefill batch/parallelism string, for example `bs2-dp1-tp8-pp4-ep1`. |
| `config_decode_npu_types` | Slash-separated NPU-version history, one entry per decode iteration. |
| `config_decode_input_seqlens` | Slash-separated padded input-length history for decode. |
| `config_decode_output_seqlens` | Slash-separated padded output-length history for decode. |
| `config_decode_batch_sizes` | Slash-separated actual decode batch-size history. |
| `config_decode_num_chips` | Slash-separated decode vPod chip-count history. |
| `config_decode_pcfgs` | Slash-separated decode batch/parallelism strings. |

An empty configuration-history field or `-1` means that value was unavailable or
not recorded.

## Experiment Customization

`run_micro26ae_sample` is also the launcher for sensitivity studies. A typical
custom invocation is:

```bash
python -m neusim.run_scripts.run_micro26ae_sample \
  --trace-file /path/to/Azure_LVEval_trace.csv \
  --trace-name Azure-LVEval \
  --model llama3-70b \
  --request-cache-dir /path/to/generated_request_lookup_cache \
  --systems Base-Max,NeuScale,Ideal \
  --goals energy \
  --results-dir /path/to/results
```

The supplied request lookup cache is only for the default DeepSeekV3/Azure/
v5p-v6e sample.
The following command generates a full matching cache from scratch before every
sensitivity study, and pass that cache to the launcher. Use a fresh output directory; `--skip-existing` resumes only
when its recorded generation contract exactly matches the requested inputs.

```bash
python -m neusim.run_scripts.generate_fleetsim_optimal_cache \
  --trace /path/to/Azure_trace.csv \
  --models llama3-70b,llama3_1-405b,deepseekv2-236b,deepseekv3-671b \
  --versions 5p,6e \
  --top-k all \
  --output-dir /path/to/generated_request_lookup_cache \
  --skip-existing
```

Repeat `--trace` for additional workloads. Keep `--top-k all` for when allocation failover is desired; this option specifies how many "best-fit" NPU allocation configurations will be stored in the lookup cache. Cache generation explores chip counts, batch
sizes, and model-parallel configurations and can require many CPU
hours and substantial disk space. Use `--dry-run` to inspect both commands and
`--help` to tune the search space.

Useful common options include `--hours`, `--max-requests`, `--request-cache-dir`,
`--chip-versions`, `--prefill-chip-versions`, `--decode-chip-versions`,
`--max-chips-per-version`, `--num-pools`, `--prediction-accuracies`, `--jobs`,
and `--results-dir`. Run the launcher with `--help`, and see the
[FleetSim reference](neusim/fleetsim/README.md) for the more details on the simulator
parameters.

## Citation

```bibtex
(To be updated after the camera ready DOI is released)
```

The supplied Azure trace is a three-hour prefix of a sampled version of
[Microsoft Azure’s Azure LLM inference trace 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md)
Code workload and is redistributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The upstream
[license](artifact_eval/micro26/data/AzurePublicDataset_LICENSE.txt),
[source metadata and required DynamoLLM citation](artifact_eval/micro26/data/AzureLLMInferenceDataset2024.md)
are included with this artifact. NeuSim citations are listed in the
[repository README](README.md).