# Enabling Spatially Fine-Grained DVFS in Neural Processing Units for Energy-Efficient LLM Serving (Artifact)

This branch contains the artifact for the MICRO'26 paper "Enabling Spatially Fine-Grained DVFS in Neural Processing Units for
Energy-Efficient LLM Serving."
It includes the chip-level and fleet-level simulators for reproducing the key figures in the paper
(Figures 2, 3, 4, 5, 11, 12, 13, 16, 17, 18, 20, 21, and 22).
The simulators are built on the NeuSim framework.

Run every command below from the NeuSim repository root. The simulator framework contains two main components:

- `neusim.npusim` is the chip-level simulator thatreproduces Figures 2, 3, 4, 11–13, 16–18, 20, and 21. It simulates the execution time and power for a single LLM inference request.
- `neusim.fleetsim` is the NPU fleet-level LLM serving simulator that reproduces Figures 5 and 22. It takes an input request trace and simulates LLM serving on a cluster of NPU instances, with support for prefill/decode disaggregation.


## Hardware and Software Requirements

*Hardware*. The artifact can be run on any x86 machine.
We recommend:

- 24 or more physical CPU cores;
- at least 32 GiB of RAM, with 64 GiB preferred;
- at least 20 GiB of free disk space.

*Software*. The artifact requires Python 3.12.2 or later on a Linux system (preferrably Ubuntu 24.04).
`pip` is used to install the simulator and the required dependencies. `conda` is preferred for managing the Python environment.

The large fleet-level inputs are hosted in the
[supplementary-files repository](https://github.com/XZman/micro26ae_supplementary_files).
It supplies the one-day Azure Code trace and separate DVFS-C and eNPU-All
lookup-cache archives. This keeps large generated/data files out of the NeuSim
Git history. The lookup caches can also be regenerated from scratch as
described under Experiment Customization, although doing so takes several
hours.


## Installation

Use Python 3.12.2 and install NeuSim via `pip`.
It is highly recommended to use conda for managing the Python environment. If you do not have conda installed, you can install it from https://www.anaconda.com/docs/getting-started/miniconda/install/linux-install.

To install the artifact:

```bash
conda create --name neusim-dvfs-ae python=3.12.2
conda activate neusim-dvfs-ae
python -m pip install -e .
export PYTHON=python
```

Clone and verify the supplementary input archives:

```bash
SUPPLEMENTARY_DIR=/tmp/micro26ae_supplementary_files
git clone --depth 1 https://github.com/XZman/micro26ae_supplementary_files.git "$SUPPLEMENTARY_DIR"
(cd "$SUPPLEMENTARY_DIR" && sha256sum -c SHA256SUMS)
```

Extract the trace and both lookup caches into the paths expected by FleetSim:

```bash
unzip -q -o "$SUPPLEMENTARY_DIR/AzureLLMInferenceTrace_code_1day.zip" -d artifact_eval/micro26/data/azure
unzip -q -o "$SUPPLEMENTARY_DIR/dvfs_lookup_dvfsc.zip" -d artifact_eval/micro26/data
unzip -q -o "$SUPPLEMENTARY_DIR/dvfs_lookup_enpu_all.zip" -d artifact_eval/micro26/data
```

This creates the extracted trace CSV and
`artifact_eval/micro26/data/dvfs_lookup/{DVFSC,CustomAll}`. Both extraction
destinations are ignored by Git.

Verify the installation and extracted inputs:

```bash
python -c 'import neusim; print(neusim.__version__)'
./artifact_eval/micro26/run_all.sh --list
sha256sum -c artifact_eval/micro26/data/SHA256SUMS
```

The first command should print `0.1.0`; the second should list 11 ready
chip-level figures; and all three extracted-input checksum entries should
report `OK`.

Run a quick end-to-end workflow check (approximately 1-3 minutes):

```bash
./artifact_eval/micro26/run_all.sh --quick --resume \
  --results-dir /tmp/neusim-micro26-quick \
  --output-dir /tmp/neusim-micro26-quick
```

This generates a set of quick plots marked `INCOMPLETE QUICK-SMOKE MATRIX`.

Optionally, developers can install the optional tools and run the relevant unit tests
(typically less than one minute):

```bash
python -m pip install -e ".[dev]"
python -m pytest -q \
  artifact_eval/micro26/tests \
  neusim/fleetsim/tests \
  neusim/run_scripts/tests/test_fleetsim_cli.py \
  neusim/configs/tests/test_npu_fleet_config.py
```

Plain `pytest` uses the project's default `neusim` test path and therefore does
not discover `artifact_eval/micro26/tests`; use the explicit command above.

## Experiment Workflow

The paper's evaluation is divided into two scopes: request-level (i.e., chip-level simulations with `neusim.npusim`) and LLM service-level (i.e., end-to-end fleet-level simulations replaying production traces with `neusim.fleetsim`).

### Chip-Level Simulations

This workflow produces Figures 2, 3, 4, 11, 12, 13, 16, 17, 18, 20, and 21.
On a 24-core machine, allow approximately 6–8 hours:

```bash
CHIP_OUT=artifact_eval/micro26/reproduced/chip

./artifact_eval/micro26/run_all.sh \
  --results-dir "$CHIP_OUT" \
  --output-dir "$CHIP_OUT" \
  --jobs 24 --trace-workers 4 \
  --group-trace-workers standard_sweep=8 \
  --group-trace-workers domain_count=8 \
  --group-trace-workers temporal_granularity=8 \
  --group-trace-workers fixed_sequence_sweep=24 \
  --group-trace-workers expert_imbalance=1 \
  --group-trace-workers power_gating=2 \
  --allow-current-ideal --resume
```

`--resume` is safe on the first invocation. A stage is reused only when its
command, source, inputs, and outputs match the recorded fingerprints.
`--allow-current-ideal` selects NeuSim's current bounded/reduced Ideal search,
not the complete theoretical 48.1-million-state lattice (which may take more than a day and do not produce significantly better results).

The following outputs are generated:

- `artifact_eval/micro26/reproduced/chip/FIGURE_REVIEW.md`
- `artifact_eval/micro26/reproduced/chip/figures/` (11 PDF figures)
- `artifact_eval/micro26/reproduced/chip/previews/` (optional PNG previews)
- `artifact_eval/micro26/reproduced/chip/raw/` (simulator outputs)
- `artifact_eval/micro26/reproduced/chip/validation_report.json`
- `artifact_eval/micro26/reproduced/chip/.micro26/resolved_run.json`

Each figure entry in `FIGURE_REVIEW.md` links its dedicated plotting script,
PDF, preview, and raw input.

After a completed experiment, you can replot and rebuild the report without rerunning the simulations:

```bash
./artifact_eval/micro26/run_all.sh plot validate \
  --results-dir "$CHIP_OUT" \
  --output-dir "$CHIP_OUT"
```

Use `--figures 2,3,4` to run a subset. See the
[artifact pipeline README](artifact_eval/micro26/README.md) for more artifact-specific
commands and the [NeuSim README](README.md) for general simulator usage.

### Fleet-Level Simulations

This workflow produces Figure 5 and Figure 22.
It replays the one-day Azure Code trace with the NPU allocation specified in Table 3.
The SLO targets are specified in Table 1.

The experiment relies on a DVFS lookup table that specifies the optimal V/f plan for a specific request batch size and sequence length. To speed up the artifact evaluation process, this lookup table is precomputed and published in the supplementary-files repository (see Installation for more details).

On an 24-core machine, allow approximately 4 hours:

```bash
FLEET_OUT=artifact_eval/micro26/reproduced/fleet

./artifact_eval/micro26/run_fleet.sh \
  --output-dir "$FLEET_OUT" \
  --resume
```

The command defaults to these supplied inputs:

- Trace: `artifact_eval/micro26/data/azure/AzureLLMInferenceTrace_code_1day.csv` (created during Installation)
- DVFS-C cache: `artifact_eval/micro26/data/dvfs_lookup/DVFSC/`
- eNPU-All cache: `artifact_eval/micro26/data/dvfs_lookup/CustomAll/`

The portable validator checks the trace identity, model/topology/budgets,
manifest identity, every cache file hash, and the aggregate cache-tree hash.
Both policy replays use strict lookup mode and must finish with zero misses.


(Optional, long-running) To generate the DVFS lookup table from scratch while running the fleet-level simulations, run the following command. Allow approximately 8 hours total on a 24-core machine:

```bash
FLEET_FRESH_OUT=artifact_eval/micro26/reproduced/fleet-fresh-cache

./artifact_eval/micro26/run_fleet.sh \
  --output-dir "$FLEET_FRESH_OUT" \
  --regenerate-lookup-cache \
  --dvfsc-cache-workers 28 \
  --customall-cache-workers 11 \
  --resume
```

The generated outputs of the fleet-level simulations are:

- `artifact_eval/micro26/reproduced/fleet/FIGURE_REVIEW.md`
- `artifact_eval/micro26/reproduced/fleet/figure05/FIGURE_05_REVIEW.md`
- `artifact_eval/micro26/reproduced/fleet/figure05/figures/figure_05_slo_slack.pdf`
- `artifact_eval/micro26/reproduced/fleet/figure22/FIGURE_22_REVIEW.md`
- `artifact_eval/micro26/reproduced/fleet/figure22/figures/figure_22_fleetsim_dvfs_timeseries.pdf`
- `artifact_eval/micro26/reproduced/fleet/figure22/figures/figure_22_fleetsim_dvfs_timeseries.csv`
- `artifact_eval/micro26/reproduced/fleet/workflow_provenance.json`

Check [`FIGURE_REVIEW.md`](artifact_eval/micro26/reproduced/fleet/FIGURE_REVIEW.md) for a summary of the generated figures.

The three fresh per-request traces are under `figure05/runs/NoDVFS/` and
`figure22/runs/{DVFSC,CustomAll}/`. See the
[FleetSim README](neusim/fleetsim/README.md) and
[EventSim README](neusim/eventsim/README.md) for more details on the fleet-level simulator structure.

## Experiment Customization

Always use a new output directory after changing an input, model, topology,
policy, or hardware table. This makes provenance comparisons unambiguous.

### Chip-Level Simulations

The paper-specific controls are concentrated in two files:

- [`paper_experiments.json`](artifact_eval/micro26/config/paper_experiments.json)
  defines models, chip/parallelism topologies, sequence lengths, degradation
  thresholds, MoE expert-capacity factors, etc.
- [`pipeline.json`](artifact_eval/micro26/config/pipeline.json) maps each figure
  to its experiment groups, raw evidence, plotting script, and PDF output.

The native runner evaluates NoDVFS, DVFS-C, eNPU-C, eNPU-All, and the Ideal baseline.
Temporal-granularity experiments additionally evaluate
DVFS-C-ms and eNPU-ms. Model and chip definitions are in `configs/models/` and
`configs/chips/`. Run `./artifact_eval/micro26/run_all.sh --list` after editing
the manifests, and use `--figures` to select the affected figures.

The hardware component V/f and power lookup tables are embedded as Python constants.
They are defined in
[`dvfs_power_getter.py`](neusim/npusim/backend/dvfs_power_getter.py):
`SA_POINTS`, `VU_POINTS`, `SRAM_POINTS`, split HBM/ICI tables, and voltage-
regulator efficiency tables. Policy selection and snapping logic lives in
[`dvfs_policy_lib.py`](neusim/npusim/backend/dvfs_policy_lib.py).

To evaluate a custom hardware V/f curve, edit those constants in a separate
branch, run the unit tests, and reproduce into a fresh output directory.

### Fleet-Level Simulations

#### Custom traces

FleetSim accepts a CSV with these required columns:

```csv
TIMESTAMP,ContextTokens,GeneratedTokens
```

Timestamps must be chronological and parseable as ISO 8601. `ContextTokens`
must be a positive integer; `GeneratedTokens` must be an integer. Extra columns are ignored.

Use the FleetSim CLI for a custom trace. Run:

```bash
python -m neusim.run_scripts.fleetsim_main --helpshort
```

for more information. Please also see the [FleetSim README](neusim/fleetsim/README.md).

#### Regenerating a lookup table

The DVFS lookup cache is keyed by padded input/output shape,
phase, and batch size. A custom trace therefore needs a cache covering all of
its padded shapes. Generate the cacheas follows for DVFS-C (`DVFSC`) and eNPU-All (`CustomAll`):

```bash
CUSTOM_TRACE=/absolute/path/to/custom_trace.csv
CUSTOM_CACHE=/absolute/path/to/custom_lookup

python artifact_eval/micro26/experiments/generate_figure_22_dvfs_cache.py \
  --trace "$CUSTOM_TRACE" \
  --output-dir "$CUSTOM_CACHE" \
  --policy DVFSC \
  --workers 28

python artifact_eval/micro26/experiments/generate_figure_22_dvfs_cache.py \
  --trace "$CUSTOM_TRACE" \
  --output-dir "$CUSTOM_CACHE" \
  --policy CustomAll \
  --workers 11
```

This generator currently targets the paper's Llama3-70B, TPU v5p, four-chip
TP4 configuration and its nine performance degradation thresholds. Changing the
model or per-vPod topology requires modifying the generator script.

#### Fleet size and topology

The final allocation is versioned in
[`figure_05_llama3_70b_tpuv5p_p20d8.json`](configs/fleetsim/figure_05_llama3_70b_tpuv5p_p20d8.json).
Each `prefill`/`decode` object specifies `count`, `npu_type`, `num_chips`,
`batch_size`, `dp`, `tp`, `pp`, and optional `ep`. For paper-style topologies,
keep `num_chips = dp × tp × pp × ep`. Pass a custom file through
`--static_vpod_allocation` to the lower-level FleetSim CLI. A changed
per-vPod topology also requires a matching service lookup cache.

## Citation

The paper is cited as follows:
```bibtex
(To be updated after the camera ready DOI is released)
```

The supplied Azure trace is licensed under CC BY 4.0. Its license, source
metadata, and required DynamoLLM citation are in
[`artifact_eval/micro26/data/azure`](artifact_eval/micro26/data/azure). NeuSim
citations are listed in the [repository README](README.md).
