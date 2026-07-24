# MICRO 2026 artifact directory

Start with the repository-root [reviewer guide](../../micro26ae.md) for
installation, full commands, runtime estimates, and experiment customization.
This README maps the files under `artifact_eval/micro26` and explains which
paths are inputs, source code, and reviewer-generated outputs.

The artifact reproduces Figures 2, 3, 4, 5, 11, 12, 13, 16, 17, 18, 20, 21,
and 22.

## Workflow index

| Workflow | Figures | Entry point | Primary inputs | Reviewer outputs |
|---|---|---|---|---|
| Chip-level simulation | 2, 3, 4, 11–13, 16–18, 20, 21 | `run_all.sh` | `config/paper_experiments.json`, `config/pipeline.json` | `reproduced/chip/` |
| Fleet-level simulation | 5, 22 | `run_fleet.sh` | Azure trace, DVFS lookup cache, SLO and static-allocation configs | `reproduced/fleet/` |

## Directory map

```text
artifact_eval/micro26/
├── README.md
├── run_all.sh                    # chip-level workflow shell entry point
├── run_fleet.sh                  # fleet-level workflow shell entry point
├── pipeline.py                   # chip experiment/plot/validation orchestrator
├── build_figure_22.py            # Figure 22 result validation and assembly
├── generate_review_report.py     # chip figure review report
├── config/
│   ├── paper_experiments.json    # chip models, phases, sweeps, and topologies
│   ├── pipeline.json             # figure-to-experiment/plot/output registry
│   └── figure_05_slo_llama3_70b_azure_code.json
├── data/
│   ├── SHA256SUMS
│   └── azure/
│       ├── ATTRIBUTION.md
│       ├── AzureLLMInferenceDataset2024.md
│       └── LICENSE
├── experiments/
│   ├── run_native.py             # native chip experiment groups
│   ├── neusim_adapter.py         # NeuSim trace and policy adapter
│   ├── run_figure_05.py          # NoDVFS serving replay and Figure 5
│   ├── run_fleet.py              # combined Figures 5 and 22 workflow
│   └── generate_figure_22_dvfs_cache.py
├── plots/
│   ├── common.py
│   └── figure_NN.py              # one dedicated script per reproduced figure
├── tests/                        # artifact workflow, plot, cache, and report tests
└── reproduced/
    ├── chip/                     # results produced by a reviewer chip run
    └── fleet/                    # results produced by a reviewer serving run
```

## Inputs

### Experiment configuration

- `config/paper_experiments.json` defines the chip-level model/phase cases,
  request lengths, NPU parallelism, performance degradation thresholds, and sensitivity
  sweeps.
- `config/pipeline.json` selects the supported figures and maps each figure to
  its experiment groups, raw input, plotting script, and PDF filename.
- `config/figure_05_slo_llama3_70b_azure_code.json` contains the serving SLO
  buckets used by Figures 5 and 22.
- `configs/fleetsim/figure_05_llama3_70b_tpuv5p_p20d8.json` contains the serving NPU instance allocation.

### Trace and lookup cache

The large input archives are published in the
[supplementary-files repository](https://github.com/XZman/micro26ae_supplementary_files):

- `AzureLLMInferenceTrace_code_1day.zip` contains the exact one-day Azure Code
  trace used by the serving experiments.
- `dvfs_lookup_dvfsc.zip` and `dvfs_lookup_enpu_all.zip` contain the DVFS-C and
  eNPU-All service-level lookup tables, respectively.

Installation verifies those archives and extracts them to:

```text
data/azure/AzureLLMInferenceTrace_code_1day.csv
data/dvfs_lookup/
├── DVFSC/
└── CustomAll/
```

The extracted paths are ignored by Git. `data/SHA256SUMS` validates the
extracted trace and both cache manifests. The `SHA256SUMS` in the
supplementary repository validates the three downloaded archives. Trace licensing, source
information, and attribution remain under `data/azure/`.

## Scripts and plots

`run_all.sh` invokes `pipeline.py`, which discovers the selected chip figures,
runs the required native experiment groups, renders PDFs, validates them, and
generates `FIGURE_REVIEW.md`.

`run_fleet.sh` invokes `experiments/run_fleet.py`. It validates the extracted
trace and lookup caches, runs fresh NoDVFS, DVFS-C, and eNPU-All serving
replays, and then renders and validates Figures 5 and 22.

Each figure has one plotting script under `artifact_eval/micro26/plots`.
`plots/common.py` contains shared formatting and input helpers.

## Output structure

### Generated outputs

Full experiment runs will generate `reproduced/chip/` and
`reproduced/fleet/`. The most useful starting points (the summary files and individual figure PDFs) are:

- `reproduced/chip/FIGURE_REVIEW.md`
- `reproduced/chip/figures/`
- `reproduced/fleet/FIGURE_REVIEW.md`
- `reproduced/fleet/figure05/figures/figure_05_slo_slack.pdf`
- `reproduced/fleet/figure22/figures/figure_22_fleetsim_dvfs_timeseries.pdf`

## Tests

The artifact tests cover orchestration, trace/cache validation, plot input
handling, output reports, and resume behavior. Run them explicitly because the
repository's default Pytest discovery path does not include this directory:

```bash
python -m pytest -q \
  artifact_eval/micro26/tests \
  neusim/fleetsim/tests \
  neusim/run_scripts/tests/test_fleetsim_cli.py \
  neusim/configs/tests/test_npu_fleet_config.py
```
