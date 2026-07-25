# FleetSim

`neusim.fleetsim` is NeuSim's event-driven simulator for an NPU fleet serving
LLM inference. It models request arrivals, separate prefill and decode queues,
batching, virtual NPU pods (vPods), heterogeneous NPU allocation, autoscaling,
queueing delay, latency SLOs, energy, and monetary cost.

FleetSim uses:

- `neusim.eventsim` for discrete-event scheduling;
- `neusim.npusim` directly for per-batch operator latency and energy estimates;
- JSON model, chip, system, and static-allocation inputs under the repository's
  top-level `configs/` directory; and
- a precomputed optimal-configuration lookup cache for dynamic policies.

## Architecture

The main event-driven path is:

```text
trace or synthetic workload
        |
        v
LoadGenerator -> LLMInferenceServiceClient -> EventSim event queue
                                                |
                                                v
             NPUClusterManager <-> LLMInferenceEndpoint/vPods
                                      |                 |
                                      v                 v
                              autoscaling events   neusim.npusim
                                      ^
                                      |
                                 MetricsServer
```

The service client injects requests into EventSim. The endpoint queues, groups,
batches, and dispatches them to prefill and decode vPods. Each vPod calls the
NeuSim backend for latency and energy estimates, while the cost model derives
monetary cost. The metrics server records completed work and maintains the
windows used by autoscalers. Scaling recommendations become vPod creation,
destruction, or reconfiguration events, with the cluster manager enforcing NPU
availability and capacity.

`--system=Ideal` is intentionally different: it evaluates the ideal energy/cost/latency of requests independently
through `ideal_baseline.py`, without EventSim queueing, contention, or auto-scaling
events.

## Install and invoke

From the repository root:

```bash
pip install -e ".[dev]"
python -m neusim.run_scripts.fleetsim_main --help
```

Run the dependency-light end-to-end smoke test:

```bash
bash neusim/run_scripts/run_fleetsim_smoke.sh
```

The smoke test sends one synthetic request through FleetSim, EventSim, and
NeuSim using a fixed TPU v4 allocation. It writes to
`results/fleetsim/smoke/` unless `NEUSIM_RESULTS_DIR` is set.

For NeuScale's auto-scaling algorithm, a precomputed optimal-configuration lookup cache is required.
For a simulation with an input trace and precomputed optimal-configuration lookup cache, invoke:

```bash
python -m neusim.run_scripts.fleetsim_main \
  --configs_path="$PWD/configs" \
  --model=llama3-70b \
  --system=NeuScale \
  --request_pattern=trace \
  --trace=Azure-Code \
  --trace_file=/path/to/request_trace.csv \
  --chip_versions=5p,6e \
  --request_results_cache_dir=/path/to/request_lookup_cache \
  --npusim_backend_cache_dir=/path/to/npusim_backend_cache \
  --opt_goal=energy \
  --max_timestamp_hours=6 \
  --output_dir="$PWD/results/fleetsim/neuscale-energy"
```

Set both `--trace_file` and `--trace` for named experiments. The former selects
the CSV, while the latter remains the experiment label.

To validate flag combinations and Pydantic configuration without running the
event simulation:

```bash
python -m neusim.run_scripts.fleetsim_main \
  --validate_only \
  --model=llama3-70b \
  --system=Base \
  --trace=Azure-test \
  --request_results_cache_dir=neusim/fleetsim/tests/data/request_lookup_cache \
  --output_dir=/tmp/fleetsim-validate
```

### Optional shell environment

`neusim/run_scripts/fleetsim_env.sh` defines a `run_fleetsim` shell function
that passes commonly reused paths to the Python entry point. It recognizes:

- `NEUSIM_REPO_ROOT`;
- `NEUSIM_CONFIGS_DIR`;
- `NEUSIM_RESULTS_DIR`;
- `NEUSIM_TRACES_DIR`;
- `NEUSIM_REQUEST_CACHE_DIR`;
- `NEUSIM_BACKEND_CACHE_DIR`; and
- `NEUSIM_PYTHON`.

When invoking the Python module directly, `NEUSIM_CONFIGS_DIR`,
`NEUSIM_RESULTS_DIR`, and `NEUSIM_TRACES_DIR` affect its path defaults.
`NEUSIM_REQUEST_CACHE_DIR` supplies the workload model's cache default.
`NEUSIM_BACKEND_CACHE_DIR` is passed by `fleetsim_env.sh`; set
`--npusim_backend_cache_dir` explicitly for a direct invocation.

## Supported Auto-scaling Systems

| `--system` | Behavior |
| --- | --- |
| `Base` | A single maximum-sequence configuration with horizontal replica scaling. It is currently equivalent to `Base-Max`. |
| `Base-Max` | Horizontal scaling initialized from the maximum sequence lengths in the admitted workload. |
| `Base-Avg` | Horizontal scaling initialized from the workload mean plus 0.25 standard deviations. |
| `NeuScale` | Heterogeneous, sequence-specific vPod groups with multidimensional horizontal/vertical scaling and best-fit routing. |
| `MultiPool` | A fixed number of percentile-derived sequence pools whose replica counts scale dynamically. |
| `Static` | An immutable prefill/decode allocation loaded from `--static_vpod_allocation`. |
| `Ideal` | A parallel per-request oracle with no event-driven queueing or auto-scaling overhead. |

## `fleetsim_main` parameters

The tables below document FleetSim's application flags.

### Run and configuration

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `llama3-8b` | Model JSON stem under `configs/models/`. |
| `--system` | `Base` | Fleet policy listed in the preceding table. |
| `--opt_goal` | `energy` | Lookup objective: `energy` or `monetary`. |
| `--configs_path` | repository `configs/` | Root containing `models/`, `chips/`, and `systems/`. |
| `--output_dir` | `$PWD/results/fleetsim` | Exact output directory. FleetSim does not append an experiment subdirectory. |
| `--validate_only` | `false` | Build and validate the configuration, then exit without simulating. |
| `--tqdm` | `false` | Show request-completion progress in a progress bar. |
| `--enable_profile` | `false` | Enable EventSim event profiling and slow NeuSim-call logging. |
| `--n_cpu` | all detected CPUs | Worker count for `Ideal` only; event-driven systems ignore it. |

Bundled LLM model JSONs include `llama3-8b`, `llama3-70b`,
`llama3_1-405b`, `llama-qwen3-32b`, `llama2-13b`,
`deepseekv2-236b`, and `deepseekv3-671b`.

### Trace workload

| Flag | Default | Meaning |
| --- | --- | --- |
| `--request_pattern` | `trace` | Workload source: `trace` or `synthetic`. |
| `--trace` | `Azure-Conv` | Trace alias or an existing CSV path.|
| `--trace_file` | unset | Explicit trace CSV; overrides the alias only for file selection. |
| `--traces_dir` | `$PWD/traces/inference` | Root used to resolve external trace aliases. |
| `--request_rate` | `1.0` | Trace-rate multiplier. Values above one are rounded to an integer; each request is replicated and copies are spread across the next interarrival gap. Values at or below one do not downsample. |
| `--max_timestamp_hours` | `-1` | Admit requests through this normalized trace horizon. A value at or below zero is unlimited. |
| `--max_num_requests` | `-1` | Request admission cap. A value at or below zero is unlimited. |

Trace CSVs must use one of these schemas:

- Azure format: `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`;
- BurstGPT format: `Timestamp`, `Request tokens`, `Response tokens`.

FleetSim normalizes timestamps to the first request, clamps output lengths to
at least two tokens, and applies the workload's sequence-padding schedule.

### Synthetic workload

These flags are used only with `--request_pattern=synthetic`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--synthetic_num_requests` | `2000` | Number of requests; must be positive. |
| `--synthetic_request_rate` | `10.0` | Poisson arrival rate in requests/s. Zero places every request at time zero. |
| `--synthetic_input_len` | `512` | Mean input length; must be positive. |
| `--synthetic_input_len_std` | `0` | Input-length standard deviation; zero makes it fixed. |
| `--synthetic_output_len` | `128` | Mean output length; must be at least two. |
| `--synthetic_output_len_std` | `0` | Output-length standard deviation; zero makes it fixed. |
| `--synthetic_seed` | `42` | Seed for reproducible arrival times and sequence lengths. |

### NPUs, allocation, and caches

| Flag | Default | Meaning |
| --- | --- | --- |
| `--chip_versions` | `5p,6e` | Comma-separated base NPU-version set. Each version needs `configs/chips/tpuv<VERSION>.json`. |
| `--prefill_chip_versions` | unset | Phase-specific prefill versions; falls back to `--chip_versions`. |
| `--decode_chip_versions` | unset | Phase-specific decode versions; falls back to `--chip_versions`. |
| `--max_chips_per_version` | unset | Shared hard capacity, for example `5p=1024,6e=2048`. Counts are nonnegative integers. |
| `--allocation_success_rate` | `1.0` | Pod-allocation success probability in `[0,1]`; sampled deterministically and cached for ten simulated minutes per NPU type. Hard capacity is checked independently. |
| `--expert_load_imbalance_factor` | `-1.0` | MoE expert load: `1` is balanced, `E/K` is worst case, and `-1` selects automatic worst case. Ignored for dense models. |
| `--static_vpod_allocation` | unset | Static allocation JSON, required for `--system=Static`. |
| `--request_results_cache_dir` | workload default | Precomputed optimal configurations and SLOs, organized by objective, model, padded sequence pair, NPU version, and phase. |
| `--npusim_backend_cache_dir` | `$PWD/results/fleetsim/.cache/npusim_backend` | Joblib cache for direct NeuSim operator analysis. This is separate from the optimal-configuration cache and does not follow a separately overridden `--output_dir`. |
| `--npusim_backend_cache_use_mmap` | `false` | Read the NeuSim disk cache with memory mapping. Do not share an mmap cache directory across machines. |

Dynamic auto-scaling systems consume the optimal-configuration cache; they do not generate
it. The shipped AE cache, preparation command, and optional maintainer
regeneration workflow are documented in
[`micro26ae.md`](../../micro26ae.md). Tiny fixtures under `tests/data/` exist
only for tests and local validation.

A static allocation has one entry per phase:

```json
{
  "prefill": {
    "count": 2,
    "npu_type": "4",
    "num_chips": 4,
    "batch_size": 1,
    "dp": 1,
    "tp": 4,
    "pp": 1,
    "ep": 1
  },
  "decode": {
    "count": 6,
    "npu_type": "4",
    "num_chips": 4,
    "batch_size": 1,
    "dp": 1,
    "tp": 4,
    "pp": 1,
    "ep": 1
  }
}
```

`count`, `num_chips`, `batch_size`, and all parallelism degrees must be
positive. Each `npu_type` must be enabled for that phase.

### Autoscaling

| Flag | Default | Meaning |
| --- | --- | --- |
| `--hs_interval_minutes` | `30.0` | Horizontal recommendation period. |
| `--vs_interval_minutes` | `10.0` | Vertical recommendation period. |
| `--hs_window_minutes` | `30.0` | Horizontal observation window. |
| `--vs_window_minutes` | `30.0` | Vertical observation window. |
| `--ewma_alpha` | `0.6` | EWMA smoothing factor in `(0,1]` for peak-rate estimation. |
| `--ewma_interval_seconds` | `10.0` | Positive request-count bin width for the EWMA rate. |
| `--scaling_headroom_factor` | `1.1` | Positive demand multiplier applied before replica rounding. |
| `--queue_drain_target_seconds` | `60.0` | Target backlog-drain time. Dynamic policies use the larger of rate and queue sizing; zero disables this signal. |
| `--coalesce_nl_threshold` | `0.5` | NeuScale normalized-load threshold for merging an underutilized group into a larger compatible group; zero disables coalescing. |
| `--num_pools` | `3` | Positive pool count for `MultiPool`. |
| `--decode_pool_single_config` | `false` | NeuScale-only decode mode that collapses decode groups into one largest-sequence umbrella configuration. |

NeuScale and MultiPool schedule multidimensional recommendations at the smaller
of the horizontal and vertical intervals and observe the larger of their two
windows. Horizontal policies use the horizontal interval/window. Static and the
direct Ideal path do not scale.

### Decode batching and prediction

| Flag | Default | Meaning |
| --- | --- | --- |
| `--decode_batch_seqlen_ratio_threshold` | `2.0` | At or above the length floor, requests batch only when the larger-to-smaller total-sequence ratio is no greater than this value. |
| `--decode_batch_seqlen_min_threshold` | `256` | Below this total-sequence floor, the ratio restriction is skipped. |
| `--output_prediction_accuracy` | `1.0` | Probability in `[0,1]` of routing to the best-fit deployed decode group. A miss deterministically selects another deployed group that can hold the full request. |
| `--output_prediction_seed` | `42` | Seed for stable correct/miss decisions and wrong-group selection. |

The prediction-accuracy model simulates wrong group selection only. It does not
model periodic re-prediction, vPod migration, or migration overhead.

## Outputs

Every completed run writes to the exact `--output_dir`:

- `request_trace.csv`: request lengths, arrival and phase timestamps, queueing
  and latency measurements, energy and cost, and the actual
  NPU/configuration/batch/parallelism used by each phase;
- `stats.json`: aggregate request, sequence-length, latency, arrival,
  throughput, token-per-joule, and token-per-dollar statistics.

Event-driven systems also write:

- `vpod_lifecycle_traces.json`;
- `vpod_reconfiguration_traces.json`;
- `checkpoint_<final_timestamp>/` with the serialized configuration, cluster
  manager, endpoint/vPods/autoscalers, metrics server, and client.

The direct `Ideal` path writes only `request_trace.csv` and `stats.json`.
`--validate_only` writes no result files. Reusing an output directory overwrites
the named reports; checkpoint timestamp directories can accumulate.

## Files and modules

| Path | Responsibility |
| --- | --- |
| `NPUFleetSimulator.py` | Top-level `EventSimulator`: configures the backend cache, owns the cluster manager/client/metrics/endpoint, runs events, dumps results, and checkpoints state. |
| `SimObject.py` | Common initialization, statistics, and checkpoint lifecycle interface for simulator-owned objects. |
| `LoadGenerator.py` | `LLMRequest` state plus Azure/BurstGPT trace parsing, trace-rate replication, sequence padding, and seeded synthetic workload generation. |
| `LLMInferenceServiceClient.py` | Applies time/request admission limits and injects request events in bounded chunks. |
| `LLMInferenceEvents.py` | Typed request, prefill/decode, vPod lifecycle/reconfiguration, and scaling recommendation/action events and listener filters. |
| `LLMInferenceEndpoint.py` | Serving data and control plane: queues, vPods, batching, routing, prediction errors, backend calls, phase transitions, scaling execution, accounting, traces, and checkpoints. |
| `MetricsServer.py` | Online queue/rate windows, completed-request collection, aggregate metrics, `request_trace.csv`, and `stats.json`. |
| `NPUClusterManager.py` | Chip JSON loading, phase/version validation, probabilistic allocation availability, hard capacity accounting, and pod allocation/deallocation. |
| `vPodAutoScaler.py` | Horizontal, Ideal, NeuScale, Vertical, MultiPool, and Static policy implementations and recommendation scheduling. |
| `vPodAutoScaler_lib.py` | Optimal-cache loading and fallback, SLO/config selection, sequence grouping/ranges, best-fit routing, and efficiency helpers. |
| `npusim_backend_interface.py` | Adapter from a FleetSim batch/configuration to `neusim.npusim`, with in-memory and optional joblib disk caching. |
| `ideal_baseline.py` | Parallel per-request oracle without event-driven contention; emits the common CSV/JSON result format. |
| `cost_model.py` | Pipeline-aware monetary-cost accounting. |
| `util.py` | Sequence padding, NPU shape, parallelism-string, sampling, and `ListMap` helpers. |
| `__init__.py` | Package marker; currently has no runtime behavior or re-exports. |
| `tests/` | Unit/integration coverage and intentionally small trace/cache fixtures, not full experiment data. |

The CLI entry point itself is
`neusim/run_scripts/fleetsim_main.py`. General FleetSim tests can be run with:

```bash
python -m pytest -q neusim/fleetsim/tests
```
