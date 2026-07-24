#!/usr/bin/env python3
"""Generate numerical and visual-review notes for MICRO26 artifact outputs.

WARN findings are triage signals, not pipeline failures: a deviation may be a
legitimate result from the updated simulator. If ``pdftoppm`` is available the
first page of each PDF is embedded as a PNG; PDF links work without it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PIPELINE = HERE / "config" / "pipeline.json"
DEFAULT_PAPER = HERE / "config" / "paper_experiments.json"
ORDER = (
    "2",
    "3",
    "4",
    "11",
    "12",
    "13",
    "16",
    "17",
    "18",
    "20",
    "21",
)
RANK = {"PASS": 0, "MANUAL": 1, "WARN": 2}
ALIASES = {
    "llama3-70b": "llama3_70b",
    "llama3_1-405b": "llama3_1_405b",
    "deepseekv2-236b": "deepseekv2_236b",
    "deepseekv3-671b": "deepseekv3_671b",
}
PAPER_UTIL = {
    ("llama3_70b", "prefill"): [98, 19, 17, 8, 1],
    ("llama3_70b", "decode"): [79, 13, 20, 95, 5],
    ("llama3_1_405b", "prefill"): [91, 17, 16, 9, 8],
    ("llama3_1_405b", "decode"): [74, 13, 18, 90, 12],
    ("deepseekv2_236b", "prefill"): [71, 13, 14, 40, 23],
    ("deepseekv2_236b", "decode"): [72, 16, 19, 99, 0],
    ("deepseekv3_671b", "prefill"): [44, 11, 10, 47, 45],
    ("deepseekv3_671b", "decode"): [77, 15, 19, 99, 0],
}

ENERGY_COMPONENTS = ("sa", "sram", "vu", "hbm", "ici", "other")
FIGURE_2_COMPONENTS = ("sa", "vu", "sram", "ici", "hbm")
FIGURE_3_UTILIZATION_FIELDS = (
    "sa_temp_util",
    "vu_temp_util",
    "sram_temp_util",
    "hbm_temp_util",
    "ici_temp_util",
)
FIGURE_4_PANELS = frozenset(
    {
        (model, phase)
        for model in ("llama3-70b", "llama3_1-405b", "deepseekv3-671b")
        for phase in ("prefill", "decode")
    }
)
FIGURE_11_RECOVERED_ZERO_THRESHOLD_MEAN_PCT = 1.504
FIGURE_11_ZERO_THRESHOLD_MEAN_TOLERANCE_PCT = 0.5
FIGURE_13_RECOVERED_PREFILL_GROWTH_PP = 0.79
FIGURE_13_PREFILL_GROWTH_TOLERANCE_PP = 0.15
AUTHORITATIVE_MS_SOURCE = {
    "repository_commit": "8ad6961b2a266e91ebb1162c7e2c5df61d10b1a4",
    "dvfs_enpu_ms_sha256": (
        "5dac84fa466e1270ec9b2d9909cc5e1221bd9b517f0d2038fc3363251230c112"
    ),
    "dvfs_region_merge_sha256": (
        "4e7433fc726dcf9ae902c9441a353d919a19baaa43a42c07c89b3a916233f86c"
    ),
    "test_dvfs_enpu_ms_sha256": (
        "223a9eae47ac83a8579738c2b21921cb32a58f107854140655823a35e8432992"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def coerce(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_dir():
        output: list[dict[str, Any]] = []
        for child in sorted(path.rglob("*.json")) + sorted(path.rglob("*.csv")):
            output.extend(records(child))
        return output
    if path.suffix.lower() == ".json":
        value = load_json(path).get("records", [])
        return [dict(row) for row in value if isinstance(row, dict)]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [
                {key: coerce(value) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    return []


def figure_4_records(operator_records_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Load exactly the six NoDVFS inputs consumed by Figure 4."""
    source = operator_records_dir / "raw_energy_None"
    selected = [
        row
        for row in records(source)
        if row.get("policy", "NoDVFS") == "NoDVFS"
        and (str(row.get("model", "")), str(row.get("phase", ""))) in FIGURE_4_PANELS
    ]
    return source, selected


def _nested_values(value: Any, key: str) -> set[str]:
    """Return non-empty scalar values for *key* anywhere in JSON-like evidence."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        candidate = value.get(key)
        candidate_is_collection = isinstance(candidate, Mapping) or (
            isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes)
        )
        if candidate is not None and not candidate_is_collection:
            text = str(candidate).strip()
            if text:
                found.add(text)
        for child in value.values():
            found.update(_nested_values(child, key))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for child in value:
            found.update(_nested_values(child, key))
    return found


def run_provenance_summary(
    results_dir: Path,
    pipeline: Mapping[str, Any],
    selected: Sequence[str],
    *,
    ga_execution_modes: set[str],
    ga_batch_sizes: set[str],
) -> dict[str, Any]:
    """Summarize execution settings for the selected experiment groups."""
    groups = sorted(
        {
            str(group)
            for key in selected
            for group in pipeline["figures"][key].get("experiment_groups", [])
        }
    )
    revisions: set[str] = set()
    dirty_states: set[str] = set()
    jobs: set[str] = set()
    configured_workers: dict[str, str] = {}
    effective_workers: dict[str, str] = {}
    environment_batches: set[str] = set()
    missing: list[str] = []

    for group in groups:
        path = results_dir / "raw" / group / "provenance.json"
        if not path.is_file():
            missing.append(group)
            continue
        payload = load_json(path)
        revision = payload.get("simulator_revision")
        if revision is not None:
            revisions.add(str(revision))
        dirty = payload.get("workspace_dirty")
        if dirty is not None:
            dirty_states.add(str(bool(dirty)).lower())

        host = payload.get("host")
        if isinstance(host, Mapping):
            requested = host.get("requested_jobs")
            if requested is not None:
                jobs.add(str(requested))
            environment = host.get("optimizer_environment")
            if isinstance(environment, Mapping):
                batch = environment.get("DVFS_GA_EXACT_BATCH_SIZE")
                if batch is not None and str(batch).strip():
                    environment_batches.add(str(batch))

        parallelism = payload.get("trace_parallelism")
        if isinstance(parallelism, Mapping):
            configured = parallelism.get("configured_worker_cap")
            effective = parallelism.get("effective_workers")
            if configured is not None:
                configured_workers[group] = str(configured)
            if effective is not None:
                effective_workers[group] = str(effective)

    return {
        "experiment_groups": groups,
        "missing_group_provenance": missing,
        "simulator_revisions": sorted(revisions),
        "workspace_dirty": sorted(dirty_states),
        "requested_jobs": sorted(jobs),
        "trace_workers": {
            "configured_by_group": configured_workers,
            "effective_by_group": effective_workers,
        },
        "exact_ga": {
            "execution_modes": sorted(ga_execution_modes),
            "recorded_batch_sizes": sorted(ga_batch_sizes),
            "environment_batch_sizes": sorted(environment_batches),
        },
    }


def _display_values(values: Sequence[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "not recorded"


def _display_worker_map(
    configured: Mapping[str, str], effective: Mapping[str, str]
) -> str:
    groups = sorted(set(configured) | set(effective))
    if not groups:
        return "not recorded"
    return ", ".join(
        f"`{group}`: `{configured.get(group, '?')}/{effective.get(group, '?')}`"
        for group in groups
    )


def num(row: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def model_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("model_id")
        or ALIASES.get(str(row.get("model", "")), row.get("model", ""))
    )


def threshold(row: Mapping[str, Any]) -> float:
    for key in ("threshold_pct", "pd_pct"):
        value = num(row, key)
        if math.isfinite(value):
            return value
    value = num(row, "pd", 0.0)
    return value * 100 if abs(value) <= 1 else value


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return "n/a" if not math.isfinite(value) else f"{value:.3g}"
    return str(value)


def span(values: Sequence[float]) -> str:
    finite = [value for value in values if math.isfinite(value)]
    return "n/a" if not finite else f"{min(finite):.2f}–{max(finite):.2f}"


def check(name: str, expected: str, observed: str, status: str) -> dict[str, str]:
    return {"name": name, "expected": expected, "observed": observed, "status": status}


def missing(name: str, source: Path) -> list[dict[str, str]]:
    return [
        check(
            name, "complete finite normalized input", f"missing/empty: {source}", "WARN"
        )
    ]


def base_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        model_id(row),
        row.get("phase"),
        row.get("input_tokens"),
        row.get("output_tokens"),
        row.get("config"),
    )


def savings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.get("policy") == "NoDVFS"
            and row.get("pg_strategy", "NoPG") == "NoPG"
            and num(row, "total_energy_J") > 0
        ):
            baselines[base_key(row)].append(num(row, "total_energy_J"))
    output = []
    for row in rows:
        baseline = baselines.get(base_key(row), [])
        energy = num(row, "total_energy_J")
        if baseline and energy > 0:
            output.append({**row, "saving_pct": 100 * (1 - energy / mean(baseline))})
    return output


def policy_diff(rows: list[dict[str, Any]], left: str, right: str) -> list[float]:
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in savings(rows):
        grouped[(model_id(row), row.get("phase"), threshold(row))][
            str(row.get("policy"))
        ].append(num(row, "saving_pct"))
    return [
        mean(cell[left]) - mean(cell[right])
        for cell in grouped.values()
        if cell.get(left) and cell.get(right)
    ]


def range_status(values: list[float], low: float, high: float) -> str:
    return (
        "PASS"
        if values and abs(min(values) - low) <= 4 and abs(max(values) - high) <= 4
        else "WARN"
    )


def evaluate(key: str, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if key == "2":
        base = [row for row in rows if row.get("policy") == "NoDVFS"]
        if not base:
            return missing("NoDVFS energy breakdown", Path("energy_records.json"))
        static, dynamic = [], []
        other, closure_errors = [], []
        eligible = 0
        for row in base:
            total = num(row, "total_energy_J")
            if total > 0:
                eligible += 1
                static_energy = sum(
                    num(row, f"static_{component}_energy_J")
                    for component in FIGURE_2_COMPONENTS
                )
                dynamic_energy = sum(
                    num(row, f"dynamic_{component}_energy_J")
                    for component in FIGURE_2_COMPONENTS
                )
                other_energy = num(row, "other_energy_J")
                if not math.isfinite(other_energy):
                    static_other = num(row, "static_other_energy_J")
                    dynamic_other = num(row, "dynamic_other_energy_J")
                    if math.isfinite(static_other) and math.isfinite(dynamic_other):
                        other_energy = static_other + dynamic_other
                if all(
                    math.isfinite(value)
                    for value in (static_energy, dynamic_energy, other_energy)
                ):
                    static_pct = 100 * static_energy / total
                    dynamic_pct = 100 * dynamic_energy / total
                    other_pct = 100 * other_energy / total
                    static.append(static_pct)
                    dynamic.append(dynamic_pct)
                    other.append(other_pct)
                    closure_errors.append(
                        abs(static_pct + dynamic_pct + other_pct - 100)
                    )
        output = []
        for name, values, low, high in (
            ("Dynamic-energy span", dynamic, 18, 61),
            ("Static-energy span", static, 27, 55),
        ):
            good = (
                values
                and abs(min(values) - low) <= 10
                and abs(max(values) - high) <= 10
            )
            output.append(
                check(
                    name,
                    f"about {low}%–{high}%",
                    f"{span(values)}%",
                    "PASS" if good else "WARN",
                )
            )
        max_closure = max(closure_errors) if closure_errors else math.nan
        output.append(
            check(
                "Combined Other-energy span",
                "static Other + dynamic Other, reported separately; stack closure ≤0.1 pp",
                (
                    f"{span(other)}%; {len(other)}/{eligible} rows; "
                    f"max closure error={fmt(max_closure)} pp"
                ),
                "PASS"
                if eligible and len(other) == eligible and max_closure <= 0.1
                else "WARN",
            )
        )
        matches = 0
        for phase, wanted in (("prefill", "sa"), ("decode", "hbm")):
            for row in (item for item in base if item.get("phase") == phase):
                components = {
                    name: num(row, f"dynamic_{name}_energy_J")
                    for name in ("sa", "vu", "sram", "hbm", "ici")
                }
                matches += max(components, key=components.get) == wanted
        output.append(
            check(
                "Dominant-component trend",
                "SA prefill; HBM decode",
                f"{matches}/{len(base)} matching cases",
                "PASS" if base and matches >= math.ceil(len(base) / 2) else "WARN",
            )
        )
        return output

    if key == "3":
        errors = []
        for row in (item for item in rows if item.get("policy") == "NoDVFS"):
            wanted = PAPER_UTIL.get((model_id(row), str(row.get("phase"))))
            if wanted:
                actual = [num(row, field) for field in FIGURE_3_UTILIZATION_FIELDS]
                actual = [value * 100 if abs(value) <= 1 else value for value in actual]
                errors.extend(
                    abs(a - b)
                    for a, b in zip(actual, wanted, strict=False)
                    if math.isfinite(a)
                )
        if not errors:
            return missing(
                "Printed utilization comparison", Path("energy_records.json")
            )
        status = (
            "PASS"
            if len(errors) == 40 and mean(errors) <= 15 and max(errors) <= 35
            else "MANUAL"
            if len(errors) < 40
            else "WARN"
        )
        return [
            check(
                "40 printed values",
                "MAE ≤15 pp; max ≤35 pp; historical HBM temporal field",
                f"n={len(errors)}, MAE={mean(errors):.2f}, max={max(errors):.2f} pp",
                status,
            )
        ]

    if key == "4":
        return (
            missing("Timeline source", Path("operator_records"))
            if not rows
            else [
                check(
                    "Timeline reconstruction",
                    "3 models × 2 phases; visual comparison",
                    f"{len(rows):,} operator rows",
                    "MANUAL",
                )
            ]
        )

    if key == "11":
        grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
        baseline_time = {
            base_key(row): num(row, "total_exe_time_ns")
            for row in rows
            if row.get("policy") == "NoDVFS"
        }
        overhead = []
        for row in savings(rows):
            if row.get("policy") == "eNPU-All":
                grouped[(str(row.get("phase")), threshold(row))].append(
                    num(row, "saving_pct")
                )
                raw = baseline_time.get(base_key(row), math.nan)
                if abs(threshold(row)) < 1e-8 and raw > 0:
                    overhead.append(100 * (num(row, "total_exe_time_ns") / raw - 1))
        targets = {
            ("prefill", 0.0): 16.9,
            ("decode", 0.0): 18.6,
            ("prefill", 10.0): 22.9,
            ("decode", 10.0): 18.6,
        }
        deviations = [
            abs(mean(grouped[cell]) - value)
            for cell, value in targets.items()
            if grouped.get(cell)
        ]
        overhead_mean = mean(overhead) if overhead else math.nan
        return [
            check(
                "eNPU-All savings anchors",
                "16.9/18.6% at 0%; 22.9/18.6% at 10%",
                f"mean anchor error={mean(deviations):.2f} pp"
                if deviations
                else "missing",
                "PASS" if len(deviations) == 4 and mean(deviations) <= 5 else "WARN",
            ),
            check(
                "Zero-threshold latency",
                (
                    "recovered source mean≈1.504% "
                    "(range 0.066%–4.738%); paper prose 0.78%–4.6%"
                ),
                (
                    f"mean={overhead_mean:.3f}%; range={span(overhead)}%"
                    if overhead
                    else "missing"
                ),
                "PASS"
                if overhead
                and abs(overhead_mean - FIGURE_11_RECOVERED_ZERO_THRESHOLD_MEAN_PCT)
                <= FIGURE_11_ZERO_THRESHOLD_MEAN_TOLERANCE_PCT
                else "WARN",
            ),
        ]

    if key == "12":
        selected = [
            row
            for row in rows
            if model_id(row) == "llama3_70b"
            and row.get("pg_strategy", "NoPG") == "NoPG"
        ]
        expected = {
            (phase, policy, float(budget))
            for phase, budgets in (
                ("prefill", (0, 2, 5, 10, 20)),
                ("decode", (0, 2)),
            )
            for policy in ("DVFS-C", "eNPU-C", "eNPU-All", "Ideal")
            for budget in budgets
        }
        expected.update({("prefill", "NoDVFS", 0.0), ("decode", "NoDVFS", 0.0)})
        selected = [
            row
            for row in selected
            if (str(row.get("phase")), str(row.get("policy")), threshold(row))
            in expected
        ]
        cells = {
            (str(row.get("phase")), str(row.get("policy")), threshold(row))
            for row in selected
        }

        closure_errors = []
        valid_stacks = 0
        for row in selected:
            total = num(row, "total_energy_J")
            components = [
                num(row, f"{component}_energy_J") for component in ENERGY_COMPONENTS
            ]
            if total > 0 and all(
                math.isfinite(value) and value >= 0 for value in components
            ):
                valid_stacks += 1
                closure_errors.append(abs(sum(components) - total) / total)

        baselines: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in selected:
            if row.get("policy") == "NoDVFS" and num(row, "total_energy_J") > 0:
                baselines[base_key(row)].append(num(row, "total_energy_J"))
        paired, lower_energy = 0, 0
        for row in selected:
            reference = baselines.get(base_key(row), [])
            energy = num(row, "total_energy_J")
            if row.get("policy") != "NoDVFS" and reference and energy > 0:
                paired += 1
                lower_energy += energy <= mean(reference) * 1.01

        max_closure = max(closure_errors) if closure_errors else math.nan
        return [
            check(
                "Component matrix",
                "30 Llama3-70B baseline/policy cells",
                f"{len(cells & expected)}/30 expected cells",
                "PASS" if expected <= cells else "WARN",
            ),
            check(
                "Finite nonnegative stack closure",
                "all six components sum to total energy",
                f"{valid_stacks}/{len(selected)} rows; max relative error={fmt(max_closure)}",
                "PASS"
                if selected and valid_stacks == len(selected) and max_closure <= 1e-6
                else "WARN",
            ),
            check(
                "Policy total versus matched NoDVFS",
                "optimized totals do not exceed baseline (1% tolerance)",
                f"{lower_energy}/{paired} paired policy cells",
                "PASS" if paired == 28 and lower_energy == paired else "WARN",
            ),
        ]

    if key == "13":
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for row in rows:
            if row.get("policy") == "eNPU-All" and row.get("phase") == "prefill":
                grouped[model_id(row)].append(
                    (threshold(row), num(row, "ivr_overhead_pct"))
                )
        deltas = []
        for values in grouped.values():
            values = sorted((x, y) for x, y in values if math.isfinite(y))
            if len(values) >= 2:
                deltas.append(values[-1][1] - values[0][1])
        growth_mean = mean(deltas) if deltas else math.nan
        return [
            check(
                "Prefill IVR growth (recovered source)",
                "recovered plotting-source mean≈0.79 pp",
                f"mean={growth_mean:.2f} pp; range={span(deltas)}"
                if deltas
                else "missing",
                "PASS"
                if deltas
                and abs(growth_mean - FIGURE_13_RECOVERED_PREFILL_GROWTH_PP)
                <= FIGURE_13_PREFILL_GROWTH_TOLERANCE_PP
                else "WARN",
            ),
            check(
                "Paper-prose IVR growth anchor",
                "paper prose says roughly +1.5 pp",
                "recovered plotting-source mean≈0.79 pp (about 0.71 pp lower)",
                "WARN",
            ),
        ]

    if key == "16":
        spatial = policy_diff(rows, "eNPU-All", "eNPU-ms")
        temporal = policy_diff(rows, "eNPU-ms", "DVFS-C-ms")
        ms_rows = [row for row in rows if row.get("policy") == "eNPU-ms"]
        provenance = {str(row.get("implementation_provenance")) for row in ms_rows}
        source_matches = [
            isinstance(row.get("authoritative_ms_source"), Mapping)
            and dict(row["authoritative_ms_source"]) == AUTHORITATIVE_MS_SOURCE
            for row in ms_rows
        ]
        provenance_ok = (
            bool(ms_rows)
            and provenance == {"authoritative_trace_util_port"}
            and all(source_matches)
        )
        provenance_observed = f"{sum(source_matches)}/{len(ms_rows)} exact source hashes; labels={sorted(provenance)}"
        negative_spatial = sum(value < -1e-9 for value in spatial)
        return [
            check(
                "eNPU-All minus eNPU-ms",
                "0.6%–6.3%",
                f"{span(spatial)} pp; negative cells={negative_spatial}/{len(spatial)}",
                "PASS"
                if range_status(spatial, 0.6, 6.3) == "PASS" and negative_spatial == 0
                else "WARN",
            ),
            check(
                "eNPU-ms minus DVFS-C-ms",
                "4.4%–9.6%",
                f"{span(temporal)} pp",
                range_status(temporal, 4.4, 9.6),
            ),
            check(
                "Millisecond source",
                "every eNPU-ms row has authoritative label + exact commit/file hashes",
                provenance_observed,
                "PASS" if provenance_ok else "WARN",
            ),
        ]

    if key == "17":
        grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            grouped[(model_id(row), row.get("phase"), threshold(row))][
                str(row.get("mode"))
            ].append(num(row, "energy_saving_pct"))
        four_three, five_four = [], []
        for cell in grouped.values():
            if cell.get("dom4_savu") and cell.get("dom3"):
                four_three.append(mean(cell["dom4_savu"]) - mean(cell["dom3"]))
            if cell.get("dom5") and cell.get("dom4_savu"):
                five_four.append(mean(cell["dom5"]) - mean(cell["dom4_savu"]))
        return [
            check(
                "4-domain minus 3-domain",
                "0.8%–7.7%",
                f"{span(four_three)} pp",
                range_status(four_three, 0.8, 7.7),
            ),
            check(
                "5-domain minus 4-domain",
                "2.3%–3.5%",
                f"{span(five_four)} pp",
                "PASS"
                if range_status(five_four, 2.3, 3.5) == "PASS"
                and max(five_four, default=-math.inf) >= 2.3
                else "WARN",
            ),
        ]

    if key == "18":
        cells: dict[tuple[str, str, int, float], list[float]] = defaultdict(list)
        for row in savings(rows):
            if row.get("policy") == "eNPU-All":
                cells[
                    (
                        model_id(row),
                        str(row.get("phase")),
                        int(num(row, "input_tokens", -1)),
                        threshold(row),
                    )
                ].append(num(row, "saving_pct"))

        def cell(model: str, phase: str, length: int, budget: float) -> float:
            value = cells.get((model, phase, length, budget), [])
            return mean(value) if value else math.nan

        first, last = (
            cell("llama3_70b", "prefill", 256, 0),
            cell("llama3_70b", "prefill", 1_048_576, 0),
        )
        llama_decode = [
            mean(value)
            for (model, phase, _, _), value in cells.items()
            if model == "llama3_70b" and phase == "decode"
        ]
        deep_decode = [
            mean(value)
            for (model, phase, _, _), value in cells.items()
            if model == "deepseekv3_671b" and phase == "decode"
        ]
        deep_prefill = [
            (length, mean(value))
            for (model, phase, length, budget), value in cells.items()
            if model == "deepseekv3_671b" and phase == "prefill" and abs(budget) < 1e-8
        ]
        peak = max(deep_prefill, key=lambda item: item[1])[0] if deep_prefill else None
        return [
            check(
                "Llama prefill endpoints",
                "~13% at 256; ~23% at 1M",
                f"{fmt(first)}%, {fmt(last)}%",
                "PASS"
                if math.isfinite(first)
                and math.isfinite(last)
                and abs(first - 13) <= 5
                and abs(last - 23) <= 5
                else "WARN",
            ),
            check(
                "Llama decode",
                "18%–19%",
                f"{span(llama_decode)}%",
                "PASS"
                if llama_decode and min(llama_decode) >= 14 and max(llama_decode) <= 23
                else "WARN",
            ),
            check(
                "DeepSeek prefill peak",
                "near 4K",
                str(peak or "missing"),
                "PASS" if peak and 1024 <= peak <= 16384 else "WARN",
            ),
            check(
                "DeepSeek decode",
                "19%–20%",
                f"{span(deep_decode)}%",
                "PASS"
                if deep_decode and min(deep_decode) >= 15 and max(deep_decode) <= 24
                else "WARN",
            ),
        ]

    if key == "20":
        target = [
            row
            for row in rows
            if row.get("policy") == "eNPU-All"
            and abs(num(row, "real_f") - 1) < 1e-8
            and abs(threshold(row) - 20) < 1e-8
        ]
        if not target:
            return missing(
                "Capacity-factor-1 point", Path("expert_imbalance_records.json")
            )
        saving = mean(num(row, "saving_wc_pct") for row in target)
        gap = mean(num(row, "gap_pp") for row in target)
        return [
            check(
                "Worst-case eNPU saving",
                "about 22%",
                f"{saving:.2f}%",
                "PASS" if abs(saving - 22) <= 3 else "WARN",
            ),
            check(
                "Provisioning loss",
                "about 4.3 pp",
                f"{gap:.2f} pp",
                "PASS" if abs(gap - 4.3) <= 3 else "WARN",
            ),
        ]

    if key == "21":
        baseline = {
            str(row.get("phase")): num(row, "total_energy_J")
            for row in rows
            if row.get("policy") == "NoDVFS" and row.get("pg_strategy") == "NoPG"
        }
        grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
        schedule_hashes: dict[tuple[str, float], dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        schedule_pairs: set[tuple[str, float]] = set()
        for row in rows:
            phase, raw = (
                str(row.get("phase")),
                baseline.get(str(row.get("phase")), math.nan),
            )
            if raw > 0:
                design = (
                    "PG"
                    if row.get("policy") == "NoDVFS"
                    and row.get("pg_strategy") == "Full"
                    else "DVFS"
                    if row.get("policy") == "eNPU-All"
                    and row.get("pg_strategy") == "NoPG"
                    else "Both"
                    if row.get("policy") == "eNPU-All"
                    and row.get("pg_strategy") == "Full"
                    else "Base"
                )
                grouped[(phase, design, threshold(row))].append(
                    100 * (1 - num(row, "total_energy_J") / raw)
                )
            if row.get("policy") == "eNPU-All" and row.get("pg_strategy") in {
                "NoPG",
                "Full",
            }:
                pair = (phase, threshold(row))
                strategy = str(row["pg_strategy"])
                schedule_pairs.add(pair)
                if row.get("schedule_sha256"):
                    schedule_hashes[pair][strategy].add(str(row["schedule_sha256"]))
        zero = []
        maxima: dict[str, float] = {}
        for phase in ("prefill", "decode"):
            pg, dvfs = (
                grouped.get((phase, "PG", 0.0), []),
                grouped.get((phase, "DVFS", 0.0), []),
            )
            if pg and dvfs:
                zero.append(mean(pg) >= mean(dvfs) - 1)
            values = [
                mean(value)
                for (p, design, _), value in grouped.items()
                if p == phase and design == "Both"
            ]
            maxima[phase] = max(values) if values else math.nan
        identical_schedule_pairs = 0
        for pair in schedule_pairs:
            by_strategy = schedule_hashes[pair]
            no_pg = by_strategy.get("NoPG", set())
            full = by_strategy.get("Full", set())
            if len(no_pg) == len(full) == 1 and no_pg == full:
                identical_schedule_pairs += 1
        return [
            check(
                "PG-only at 0%",
                "slightly beats DVFS-only",
                f"{sum(zero)}/{len(zero)} phases",
                "PASS" if len(zero) == 2 and all(zero) else "WARN",
            ),
            check(
                "Combined maximum",
                "~30% / 31.5% prefill/decode",
                f"{fmt(maxima.get('prefill'))}% / {fmt(maxima.get('decode'))}%",
                "PASS"
                if abs(maxima.get("prefill", math.nan) - 30) <= 5
                and abs(maxima.get("decode", math.nan) - 31.5) <= 5
                else "WARN",
            ),
            check(
                "Schedule identity",
                "NoPG and Full hashes identical",
                f"{identical_schedule_pairs}/{len(schedule_pairs)} paired phase/threshold cells",
                "PASS"
                if schedule_pairs and identical_schedule_pairs == len(schedule_pairs)
                else "WARN",
            ),
        ]

    raise ValueError(f"no evaluator for {key}")


def selection(specification: str) -> list[str]:
    selected: set[str] = set()
    for token in specification.lower().replace("figure", "").split(","):
        token = token.strip()
        if token == "all":
            selected.update(ORDER)
        elif "-" in token:
            first, last = (int(value) for value in token.split("-", 1))
            selected.update(str(value) for value in range(first, last + 1))
        elif token:
            selected.add(str(int(token)))
    unknown = selected - set(ORDER)
    if unknown:
        raise ValueError("unknown review items: " + ", ".join(sorted(unknown)))
    return [key for key in ORDER if key in selected]


def relative(path: Path, report: Path) -> str:
    return Path(os.path.relpath(path, report.parent)).as_posix().replace(" ", "%20")


def preview(pdf: Path, key: str, directory: Path, mode: str) -> Path | None:
    target = directory / f"figure_{key}_preview.png"
    if mode == "skip" or not pdf.is_file():
        return None
    if (
        mode == "auto"
        and target.is_file()
        and target.stat().st_size
        and target.stat().st_mtime_ns >= pdf.stat().st_mtime_ns
    ):
        return target
    binary = shutil.which("pdftoppm")
    if binary is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            binary,
            "-f",
            "1",
            "-singlefile",
            "-png",
            "-r",
            "120",
            str(pdf),
            str(target.with_suffix("")),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode and mode == "render":
        raise RuntimeError(f"pdftoppm failed for {pdf}: {completed.stderr.strip()}")
    return target if target.is_file() and target.stat().st_size else None


def build(args: argparse.Namespace) -> dict[str, Any]:
    pipeline = load_json(args.pipeline_manifest)
    selected = selection(args.figures or pipeline["default_selection"])
    report = (args.output or args.output_dir / "FIGURE_REVIEW.md").resolve()
    preview_dir = (args.preview_dir or args.output_dir / "previews").resolve()
    sections = []
    ga_execution_modes: set[str] = set()
    ga_batch_sizes: set[str] = set()
    for key in selected:
        item = pipeline["figures"][key]
        plot_script = item.get("plot_script")
        if not isinstance(plot_script, str) or not plot_script:
            raise ValueError(f"Figure {key} must define one plotting script")
        render_script = (HERE / "plots" / plot_script).resolve()
        render_kind = "plotting script"
        if not render_script.is_file():
            raise FileNotFoundError(f"missing {render_kind}: {render_script}")
        render_label = render_script.relative_to(HERE.parents[1]).as_posix()
        artifact = args.output_dir / item["output"]
        source = args.results_dir / item["input"] if item.get("input") else artifact
        rows = records(source)
        if key == "4":
            source, rows = figure_4_records(source)
        ga_execution_modes.update(_nested_values(rows, "ga_execution_mode"))
        ga_batch_sizes.update(_nested_values(rows, "ga_exact_batch_size"))
        checks = [
            check(
                "Artifact file",
                "non-empty PDF",
                f"{artifact.stat().st_size:,} bytes"
                if artifact.is_file()
                else "missing",
                "PASS" if artifact.is_file() and artifact.stat().st_size else "WARN",
            )
        ]
        quick_evidence = args.quick or any(
            row.get("quick_smoke") is True
            or str(row.get("quick_smoke", "")).lower() == "true"
            or str(row.get("matrix_mode", "")).lower().startswith("quick")
            for row in rows
        )
        if quick_evidence:
            checks.append(
                check(
                    "Run completeness",
                    "full paper experiment matrix",
                    "quick smoke matrix",
                    "WARN",
                )
            )
        # Figures 2 and 3 consume NoDVFS rows only. Optimized rows share their
        # input file, but their candidate construction cannot affect either plot.
        candidate_semantics_relevant = key not in {"2", "3"}
        shared_heuristic_rows = [
            row
            for row in rows
            if candidate_semantics_relevant
            and row.get("candidate_set_shared_across_budgets") is True
            and row.get("independent_per_budget_candidate_semantics_preserved") is False
        ]
        if shared_heuristic_rows:
            checks.append(
                check(
                    "Shared candidate-envelope semantics",
                    "independent per-budget candidates for exact paper equivalence",
                    f"{len(shared_heuristic_rows)} rows use the labeled 100%-envelope heuristic",
                    "WARN",
                )
            )
        checks.extend(evaluate(key, rows))
        status = max((item["status"] for item in checks), key=RANK.get)
        image = preview(artifact, key, preview_dir, args.previews)
        sections.append(
            {
                "key": key,
                "title": item["title"],
                "artifact": artifact,
                "source": source,
                "preview": image,
                "render_script": render_script,
                "render_kind": render_kind,
                "render_label": render_label,
                "checks": checks,
                "status": status,
            }
        )

    run_provenance = run_provenance_summary(
        args.results_dir,
        pipeline,
        selected,
        ga_execution_modes=ga_execution_modes,
        ga_batch_sizes=ga_batch_sizes,
    )
    summary = {
        status: sum(section["status"] == status for section in sections)
        for status in RANK
    }
    lines = [
        "# MICRO26 Figure Numerical Review",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        *(
            [
                f"Paper: [NPU DVFS paper PDF]({relative(args.paper_pdf.resolve(), report)})",
                "",
            ]
            if args.paper_pdf is not None and args.paper_pdf.is_file()
            else []
        ),
        "Verdicts are check-level diagnostics, not a claim of exact paper equivalence. "
        "A **PASS** for `Artifact file` validates only that the generated output exists and is non-empty. "
        "For numerical checks, **PASS** is within an encoded paper trend/tolerance; "
        "**WARN** is divergent or missing; **MANUAL** requires visual or host-specific judgment. "
        "WARN does not mean the simulator is wrong, and an overall PASS does not establish exact "
        "numerical or visual equivalence to the published paper.",
        "",
        f"Summary: **{summary['PASS']} PASS**, **{summary['WARN']} WARN**, **{summary['MANUAL']} MANUAL**.",
        "",
        "## Run provenance",
        "",
        "Collected from the selected groups' raw `provenance.json` files and DVFS-C planning fields.",
        "",
        f"- Simulator revision: {_display_values(run_provenance['simulator_revisions'])}",
        f"- Workspace dirty: {_display_values(run_provenance['workspace_dirty'])}",
        f"- Requested jobs: {_display_values(run_provenance['requested_jobs'])}",
        "- Trace workers (configured/effective by group): "
        + _display_worker_map(
            run_provenance["trace_workers"]["configured_by_group"],
            run_provenance["trace_workers"]["effective_by_group"],
        ),
        "- Exact GA execution mode: "
        + _display_values(run_provenance["exact_ga"]["execution_modes"]),
        "- Exact GA batch size (record/environment): "
        + _display_values(run_provenance["exact_ga"]["recorded_batch_sizes"])
        + " / "
        + _display_values(run_provenance["exact_ga"]["environment_batch_sizes"]),
        *(
            [
                "- Missing selected-group provenance: "
                + _display_values(run_provenance["missing_group_provenance"])
            ]
            if run_provenance["missing_group_provenance"]
            else []
        ),
        "",
        "## Outputs and reproduction scripts",
        "",
        "Each requested output is generated by exactly one repository plotting script.",
        "",
        "| Item | Plotting script | Artifact | Verdict |",
        "|---|---|---|---|",
    ]
    for section in sections:
        label = f"Figure {section['key']}"
        lines.append(
            f"| {label} | [{section['render_label']}]({relative(section['render_script'], report)}) "
            f"| [{section['title']}]({relative(section['artifact'], report)}) | **{section['status']}** |"
        )
    for section in sections:
        label = f"Figure {section['key']}"
        lines.extend(
            [
                "",
                f"## {label} — {section['title']}",
                "",
                f"Artifact: [{section['artifact'].name}]({relative(section['artifact'], report)})  ",
                f"{section['render_kind'].capitalize()}: "
                f"[{section['render_label']}]({relative(section['render_script'], report)})  ",
                f"Raw evidence: [{section['source'].name}]({relative(section['source'], report)})  ",
                f"Verdict: **{section['status']}**",
                "",
            ]
        )
        if section["preview"]:
            lines.extend(
                [
                    f"![{label} first-page preview]({relative(section['preview'], report)})",
                    "",
                ]
            )
        lines.extend(
            ["| Check | Paper expectation | Observed | Verdict |", "|---|---|---|---|"]
        )
        for item in section["checks"]:
            lines.append(
                f"| {item['name']} | {item['expected']} | {item['observed']} | **{item['status']}** |"
            )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path = (args.json_output or report.with_suffix(".json")).resolve()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": summary,
        "run_provenance": run_provenance,
        "sections": [
            {
                **{
                    name: value
                    for name, value in section.items()
                    if name not in {"artifact", "source", "preview", "render_script"}
                },
                "artifact": str(section["artifact"]),
                "source": str(section["source"]),
                "render_script": str(section["render_script"]),
                "preview": str(section["preview"]) if section["preview"] else None,
            }
            for section in sections
        ],
    }
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"report": report, "json": json_path, "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pipeline-manifest", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--paper-manifest", type=Path, default=DEFAULT_PAPER)
    parser.add_argument("--figures")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-pdf", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument(
        "--previews", choices=("auto", "render", "skip"), default="auto"
    )
    args = parser.parse_args(argv)
    args.results_dir, args.output_dir = (
        args.results_dir.resolve(),
        args.output_dir.resolve(),
    )
    load_json(args.paper_manifest)  # fail early on experiment-manifest drift
    result = build(args)
    print(f"wrote {result['report']} ({result['summary']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
