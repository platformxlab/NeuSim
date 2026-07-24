# FleetSim

`neusim.fleetsim` is NeuSim's event-driven simulator for a fixed fleet of
NPU instances serving LLM inference. It uses `neusim.eventsim` for event
scheduling and invokes `neusim.npusim` for operator latency and energy.

Every run must provide a json configuration specifying the NPU instance allocation for both prefill and decode (with prefill/decode disaggregation).

```json
{
  "prefill": {
    "count": 20,
    "npu_type": "5p",
    "num_chips": 4,
    "batch_size": 1,
    "dp": 1,
    "tp": 4,
    "pp": 1
  },
  "decode": {
    "count": 8,
    "npu_type": "5p",
    "num_chips": 4,
    "batch_size": 1,
    "dp": 1,
    "tp": 4,
    "pp": 1
  }
}
```

Run a synthetic experiment from any checkout location with:

```bash
python -m neusim.run_scripts.fleetsim_main \
  --configs_path=./configs \
  --model=llama3-8b \
  --request_pattern=synthetic \
  --synthetic_num_requests=10 \
  --synthetic_request_rate=1 \
  --static_vpod_allocation=./configs/fleetsim/smoke_llama3_8b_tpuv4.json \
  --output_dir=./results/fleetsim/example
```

Chip versions are inferred from the allocation file.

Run the deterministic one-request integration smoke test with:

```bash
bash neusim/run_scripts/run_fleetsim_smoke.sh
```

The shared shell helper honors these environment variables:

- `NEUSIM_CONFIGS_DIR`
- `NEUSIM_RESULTS_DIR`
- `NEUSIM_TRACES_DIR`
- `NEUSIM_BACKEND_CACHE_DIR`
- `NEUSIM_PYTHON`
