"""Shared, side-effect-free loaders and style for the MICRO'26 plots.

The loaders accept either a normalized CSV/JSON table or a NeuSim/trace_util
result directory.  No machine-specific path is embedded here.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    "llama3-70b",
    "llama3_1-405b",
    "deepseekv2-236b",
    "deepseekv3-671b",
]
MODEL_LABELS = {
    "llama3-70b": "Llama3-70B",
    "llama3_1-405b": "Llama3.1-405B",
    "deepseekv2-236b": "DeepSeekV2-236B",
    "deepseekv3-671b": "DeepSeekV3-671B",
}
PHASES = ["prefill", "decode"]
COMPONENTS = ["SA", "SRAM", "VU", "HBM", "ICI", "Other"]
COMPONENT_COLORS = {
    "SA": "#4878cf",
    "SRAM": "#f1ce54",
    "VU": "#2a6f48",
    "HBM": "#df7070",
    "ICI": "#dddddd",
    "Other": "#aaaaaa",
}
COMPONENT_HATCHES = {
    "SA": "//",
    "SRAM": "//",
    "VU": "",
    "HBM": "",
    "ICI": "",
    "Other": "//",
}
POLICY_ORDER = ["DVFS-C", "eNPU-C", "eNPU-All", "Ideal"]
POLICY_STYLE = {
    "NoDVFS": dict(color="black", marker="", linestyle=":"),
    "DVFS-C": dict(color="#2a9d4e", marker="^", linestyle="-."),
    "eNPU-C": dict(color="#1a54a6", marker="o", linestyle="-"),
    "eNPU-All": dict(
        color="#e8750a",
        marker="D",
        linestyle="--",
        markerfacecolor="white",
        markeredgecolor="#e8750a",
        markeredgewidth=1.0,
    ),
    "Ideal": dict(color="#8b0000", marker="", linestyle=(0, (1, 1))),
    "DVFS-C-ms": dict(color="#7bc67b", marker="v", linestyle="-."),
    "eNPU-ms": dict(color="#f2ac65", marker="s", linestyle="--"),
}


def paper_style(font_size: float = 11) -> None:
    """Install a compact vector-PDF style without requiring Times fonts."""
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size + 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.5,
        }
    )


def canonical_model(value: str) -> str:
    s = value.strip().lower().replace("deepseek-v", "deepseekv")
    s = s.replace("llama3.1", "llama3_1").replace("_236b", "-236b")
    s = s.replace("_671b", "-671b").replace("_70b", "-70b")
    for model in MODELS:
        if model in s:
            return model
    return value


def canonical_policy(value: str) -> str:
    s = value.strip()
    aliases = {
        "None": "NoDVFS",
        "NoDVFS": "NoDVFS",
        "DVFSC": "DVFS-C",
        "DVFSCNoPareto": "DVFS-C",
        "Custom": "eNPU-C",
        "CustomAll": "eNPU-All",
        "Ideal": "Ideal",
        "DVFSC_ms": "DVFS-C-ms",
        "DVFSCms": "DVFS-C-ms",
        "DVFS_C_ms": "DVFS-C-ms",
        "CustomAll_ms": "eNPU-ms",
        "CustomAllms": "eNPU-ms",
        "CUSTOM_ALL_ms": "eNPU-ms",
    }
    return aliases.get(s, s)


def _number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return float(text)
    except ValueError:
        return value


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out = {str(k): _number(v) for k, v in record.items()}
    if "model" in out:
        out["model"] = canonical_model(str(out["model"]))
    if "phase" in out:
        out["phase"] = str(out["phase"]).lower()
    if "policy" in out:
        out["policy"] = canonical_policy(str(out["policy"]))
    return out


def load_table(path: Path) -> list[dict[str, Any]]:
    """Load a normalized CSV or JSON table.

    JSON may be a list, ``{"results": [...]}``, or ``{"records": [...]}``.
    """
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            return [_normalize_record(row) for row in csv.DictReader(handle)]
    with path.open() as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict) and isinstance(obj.get("results"), list):
        rows = obj["results"]
    elif isinstance(obj, dict) and isinstance(obj.get("records"), list):
        rows = obj["records"]
    else:
        raise ValueError(f"{path} is not a normalized record table")
    return [_normalize_record(row) for row in rows]


_PHASE_RE = re.compile(r"inference-v[^_]+_(prefill|decode)")


def infer_path_metadata(path: Path) -> dict[str, Any]:
    """Infer model, phase, policy, threshold, and configuration from a result path."""
    text = str(path)
    meta: dict[str, Any] = {}
    phase_match = _PHASE_RE.search(path.name)
    if phase_match:
        meta["phase"] = phase_match.group(1)
    for part in path.parts:
        model = canonical_model(part)
        if model in MODELS:
            meta["model"] = model
            seq = re.search(r"_(\d+)_(\d+)$", part)
            if seq:
                meta["input_tokens"] = int(seq.group(1))
                meta["output_tokens"] = int(seq.group(2))
        if part.startswith("dp") and "-tp" in part:
            meta["config"] = part
    match = re.search(r"(?:^|/)raw_[^/]*?energy_(pg_)?([^/]+)(?:/|$)", text)
    if match:
        token = match.group(2)
        threshold = None
        policy_token = token
        tail = token.rsplit("_", 1)
        if len(tail) == 2:
            try:
                threshold = float(tail[1]) * 100.0
                policy_token = tail[0]
            except ValueError:
                pass
        meta["policy"] = canonical_policy(policy_token)
        meta["threshold_pct"] = 0.0 if threshold is None else threshold
        meta["pg_run"] = bool(match.group(1))
    return meta


def load_energy_results(path: Path) -> list[dict[str, Any]]:
    """Load normalized records or recursively extract NeuSim energy JSONs."""
    if path.is_file():
        return load_table(path)
    normalized = path / "energy_records.json"
    if normalized.is_file():
        return load_table(normalized)
    records: list[dict[str, Any]] = []
    for json_path in sorted(path.rglob("inference-v*.json")):
        try:
            with json_path.open() as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        energy_stats = doc.get("energy_stats")
        if not isinstance(energy_stats, dict):
            continue
        base = infer_path_metadata(json_path)
        base["source"] = str(json_path)
        for pg_strategy, stats in energy_stats.items():
            if not isinstance(stats, dict) or "total_energy_J" not in stats:
                continue
            row = dict(base)
            row["pg_strategy"] = pg_strategy
            row.update(
                {k: v for k, v in stats.items() if not isinstance(v, dict | list)}
            )
            component_stats = stats.get("component_stats", {})
            if isinstance(component_stats, dict):
                row.update(
                    {
                        k: v
                        for k, v in component_stats.items()
                        if not isinstance(v, dict | list)
                    }
                )
            records.append(_normalize_record(row))
    if not records:
        raise FileNotFoundError(f"no energy result JSONs found under {path}")
    return records


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    """Load one CSV or all inference CSVs below a directory, adding path metadata."""
    files = [path] if path.is_file() else sorted(path.rglob("inference-v*.csv"))
    records: list[dict[str, Any]] = []
    for csv_path in files:
        meta = infer_path_metadata(csv_path)
        with csv_path.open(newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                item = dict(meta)
                item.update(_normalize_record(row))
                item["row_index"] = index
                item["source"] = str(csv_path)
                records.append(item)
    if not records:
        raise FileNotFoundError(f"no inference CSV rows found at {path}")
    return records


def threshold_pct(row: Mapping[str, Any]) -> float:
    if "threshold_pct" in row:
        return float(row["threshold_pct"])
    value = float(row.get("perf_degrad", row.get("threshold", 0.0)))
    return value * 100.0 if abs(value) <= 1.0 else value


def count_weight(row: Mapping[str, Any]) -> float:
    return float(row.get("Count", row.get("count", 1.0)))


def weighted_mean(
    rows: Sequence[Mapping[str, Any]], key: str, weight: str | None = None
) -> float:
    vals: list[tuple[float, float]] = []
    for row in rows:
        if key not in row or row[key] in (None, ""):
            continue
        w = float(row.get(weight, 1.0)) if weight else 1.0
        vals.append((float(row[key]), w))
    denom = sum(w for _, w in vals)
    return sum(v * w for v, w in vals) / denom if denom else math.nan


def grouped_mean(
    rows: Iterable[Mapping[str, Any]], keys: Sequence[str], value: str
) -> dict[tuple[Any, ...], float]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if value not in row:
            continue
        groups[tuple(row.get(k) for k in keys)].append(float(row[value]))
    return {k: mean(v) for k, v in groups.items()}


def component_energy(
    row: Mapping[str, Any], component: str, kind: str | None = None
) -> float:
    prefix = component.lower()
    if kind:
        key = f"{kind}_{prefix}_energy_J"
        if key in row:
            return float(row[key])
    key = f"{prefix}_energy_J"
    if key in row:
        return float(row[key])
    if component == "Other" and "total_energy_J" in row:
        known = sum(component_energy(row, c) for c in COMPONENTS if c != "Other")
        return max(0.0, float(row["total_energy_J"]) - known)
    return 0.0


def mark_incomplete(fig: mpl.figure.Figure, allow_incomplete: bool) -> None:
    """Visibly identify figures produced from an explicitly incomplete matrix."""
    if allow_incomplete:
        fig.text(
            0.995,
            0.005,
            "INCOMPLETE QUICK-SMOKE MATRIX",
            ha="right",
            va="bottom",
            color="#a00000",
            fontsize=7,
            fontweight="bold",
        )


def save_figure(fig: mpl.figure.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches = getattr(fig, "_micro26_bbox_inches", "tight")
    save_dpi = getattr(fig, "_micro26_save_dpi", None)
    fig.savefig(output, bbox_inches=bbox_inches, dpi=save_dpi)
    plt.close(fig)
    print(f"[saved] {output}")


def require_fields(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str], source: Path
) -> None:
    missing = [field for field in fields if not any(field in row for row in rows)]
    if missing:
        raise ValueError(f"{source} is missing required field(s): {', '.join(missing)}")


def require_combinations(
    observed: Iterable[tuple[Any, ...]],
    expected: Iterable[tuple[Any, ...]],
    label: str,
    allow_partial: bool = False,
) -> None:
    """Reject an incomplete paper matrix unless partial plotting is explicit."""
    if allow_partial:
        return
    missing = sorted(
        set(expected) - set(observed), key=lambda item: tuple(map(str, item))
    )
    if not missing:
        return
    preview = ", ".join("/".join(map(str, item)) for item in missing[:12])
    remainder = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
    raise ValueError(
        f"{label} is incomplete; missing {preview}{remainder}. "
        "Use --allow-incomplete only for reduced smoke runs."
    )


def plot_policy_line(
    ax: Any, xs: Sequence[float], ys: Sequence[float], policy: str, **kwargs: Any
) -> None:
    style = dict(POLICY_STYLE.get(policy, {}))
    style.update(kwargs)
    ax.plot(xs, ys, label=policy, **style)


def grid(ax: Any) -> None:
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.55)
