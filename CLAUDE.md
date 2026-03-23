# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuSim is a simulator framework for modeling performance, power, and carbon behaviors of Neural Processing Units (NPUs) running ML workloads. It supports LLMs (LLaMA, DeepSeek), diffusion models (DiT, GLIGEN), and recommendation models (DLRM).

## Build & Development Commands

```bash
# Install (use conda with Python 3.12.2)
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run a single test file
pytest neusim/npusim/frontend/tests/test_operator.py

# Run regression test (compares current code against previous commit)
pytest neusim/tests/regression/test_regression.py --runslow -v
# Or via env var
NEUSIM_RUN_SLOW_TESTS=1 pytest neusim/tests/regression/test_regression.py -v
# Compare against a specific commit/tag
NEUSIM_REGRESSION_BASELINE=<commit> pytest neusim/tests/regression/test_regression.py --runslow -v

# Lint (critical ruff errors are blocking; style, format, mypy are warnings)
./lint.sh

# Individual lint tools
ruff check --select E9,F821,F822,F823 .   # critical errors only
ruff format --check .                       # format check
mypy .                                      # type check
```

## Architecture

The simulator has three main layers:

**Frontend** (`neusim/npusim/frontend/`) — Model graph generation, orchestration, and analysis:
- Ops generators create operator graphs for each model type (`llm_ops_generator.py`, `dlrm_ops_generator.py`, `dit_ops_generator.py`, `gligen_ops_generator.py`)
- Orchestration wrappers for power/energy analysis (`power_analysis_lib.py`, `energy_carbon_analysis_lib.py`)
- Re-export facades for backward compatibility (`dvfs_policy_lib.py`, `dvfs_power_getter.py`)
- Memory footprint analysis (`memory_footprint_analysis_lib.py`)
- Simulation orchestration (`run_sim_lib.py`)

**Backend** (`neusim/npusim/backend/`) — Core performance and power simulation:
- `npusim_lib.py` — Operator execution time calculation for einsum, conv2d, elementwise, attention ops; tensor shape parsing, component utilization tracking
- `power_model.py` — Energy modeling: dynamic/static energy per component, DVFS time scaling, regulator efficiency, peak FLOPS computation
- `dvfs_policy_lib.py` — DVFS policy selection: V/f band tables, frequency slowdown, voltage selection
- `dvfs_power_getter.py` — V/f lookup tables and power computation for SA, VU, SRAM, HBM, ICI components

**XLA HLO Parser** (`neusim/xla_hlo_parser/`) — Parses XLA-style programs for operator semantics analysis. May be refactored into backend in the future.

**Config System** (`neusim/configs/` + `configs/`):
- Pydantic models in `neusim/configs/` validate JSON configs from `configs/`
- Chip configs (`configs/chips/`): TPU hardware specs (v2 through v6p)
- Model configs (`configs/models/`): DNN architecture parameters
- System configs (`configs/systems/`): Datacenter-level settings (PUE, carbon intensity)
- Power gating configs (`neusim/configs/power_gating/`): `PowerGatingConfig` class and named presets (NoPG, Ideal, Base, HW, Full, etc.)

## Key Abstractions

- `Operator` class — Represents tensor operations with execution stats, DVFS config, and power gating. `OpcodeType` enum: Conv2D, Einsum, FlashAttention, Elementwise, Collective, Embedding, Other.
- `ChipConfig` — Hardware specs (systolic arrays, vector units, HBM/SRAM, bandwidth, power).
- `ModelConfig` — Combines model + parallelism + hardware + system config.
- `PowerGatingConfig` — Power gating settings per component (temporal/spatial granularity, delay cycles, power level factors).

## Run Scripts

Entry points in `neusim/run_scripts/`:
- `run_sim.py` — Performance simulation (Ray-distributed)
- `energy_operator_analysis_main.py` — Power/energy per-operator analysis
- `carbon_analysis_main.py` — Carbon footprint computation
- `slo_analysis_main.py` — SLO-aware NPU config search

## Parallelism

Supports data parallelism (DP), tensor parallelism (TP), pipeline parallelism (PP), and expert parallelism (EP) for MoE models. Multi-chip simulation includes ICI and DCN communication modeling.

## Testing

- Unit tests live alongside source in `tests/` subdirectories
- Regression tests live in `neusim/tests/regression/`
  - `regression_runner.py` — Standalone simulation runner (no absl flags)
  - `test_regression.py` — Compares outputs between current code and a baseline commit
  - Marked `@pytest.mark.slow` and `@pytest.mark.regression`, skipped by default
  - Always passes — generates a diff report to `results/regression/regression_report.txt` for human review
  - Uses git worktree + lightweight venv to run baseline code in isolation
  - Covers 7 experiments: LLM inference/training, DLRM, DiT, DeepSeek with varied parallelism (TP, PP, DP, EP, DP_DCN, PP_DCN, EP_DCN)
- `neusim/conftest.py` adds `--runslow` flag and `NEUSIM_RUN_SLOW_TESTS` env var to gate slow tests

## Code Standards

- Python 3.12, ruff for linting (ignores E501), mypy for type checking
- Type hints encouraged
- New files should target 90% test coverage
- Tests live alongside source in `tests/` subdirectories
