#!/usr/bin/env python3
"""Run the plot-free NeuScale AE sample and sensitivity experiments.

The default matrix replays the three-hour Azure Code sample for
DeepSeekV3-671B on v5p/v6e with Base-Max, NeuScale, and Ideal, optimizing for
energy and monetary cost.  The launcher writes FleetSim outputs and one JSON
manifest; it does not plot or post-process results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifact_eval" / "micro26"
DEFAULT_TRACE_FILE = (
    ARTIFACT_DIR / "data" / "AzureLLMInferenceTrace_code_3h_sampled.csv"
)
DEFAULT_CACHE_DIR = ARTIFACT_DIR / "request_lookup_cache_deepseekv3_azure_3h"
DEFAULT_RESULTS_DIR = ARTIFACT_DIR / "results"
DEFAULT_BACKEND_CACHE_DIR = ARTIFACT_DIR / ".cache" / "npusim_backend"
DEFAULT_SYSTEMS = ("Base-Max", "NeuScale", "Ideal")
DEFAULT_GOALS = ("energy", "monetary")
DEFAULT_CHIP_VERSIONS = ("5p", "6e")
DEFAULT_HOURS = 3.0
DEFAULT_PREDICTION_ACCURACY = 0.6
SUPPORTED_SYSTEMS = {
    "Base",
    "Base-Avg",
    "Base-Max",
    "Ideal",
    "MultiPool",
    "NeuScale",
}
SUPPORTED_GOALS = {"energy", "monetary"}
PREDICTION_SYSTEMS = {"NeuScale"}
RUN_CONTRACT_NAME = "run_contract.json"
CACHE_MANIFEST_NAMES = (
    "micro26ae_sample_cache_manifest.json",
    "fleetsim_cache_generation_manifest.json",
)


class LauncherError(RuntimeError):
    """Raised when launcher inputs or FleetSim outputs are invalid."""


@dataclass(frozen=True)
class Experiment:
    system: str
    goal: str
    max_chips_per_version: str | None
    num_pools: int | None
    prediction_accuracy: float | None


@dataclass(frozen=True)
class Config:
    model: str
    trace_name: str
    trace_file: Path
    systems: tuple[str, ...]
    goals: tuple[str, ...]
    chip_versions: tuple[str, ...]
    prefill_chip_versions: tuple[str, ...] | None
    decode_chip_versions: tuple[str, ...] | None
    chip_caps: tuple[str | None, ...]
    num_pools: tuple[int, ...]
    prediction_accuracies: tuple[float, ...]
    hours: float
    max_requests: int
    n_cpu: int
    jobs: int
    force: bool
    results_dir: Path
    cache_dir: Path
    backend_cache_dir: Path
    configs_dir: Path
    python: str
    env: Mapping[str, str]


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from error
    return values


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from error
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepseekv3-671b")
    parser.add_argument("--trace-name", default="Azure-Code")
    parser.add_argument("--trace-file", type=Path, default=DEFAULT_TRACE_FILE)
    parser.add_argument(
        "--systems", type=_csv_strings, default=DEFAULT_SYSTEMS, metavar="SYSTEM,..."
    )
    parser.add_argument(
        "--goals", type=_csv_strings, default=DEFAULT_GOALS, metavar="GOAL,..."
    )
    parser.add_argument(
        "--chip-versions",
        type=_csv_strings,
        default=DEFAULT_CHIP_VERSIONS,
        metavar="VERSION,...",
    )
    parser.add_argument(
        "--prefill-chip-versions", type=_csv_strings, metavar="VERSION,..."
    )
    parser.add_argument(
        "--decode-chip-versions", type=_csv_strings, metavar="VERSION,..."
    )
    parser.add_argument(
        "--max-chips-per-version",
        dest="chip_caps",
        action="append",
        metavar="CAPS",
        help=(
            "Fleet chip cap such as 5p=256,6e=512. Repeat for a sweep; use "
            "'unlimited' to include the unbounded setting."
        ),
    )
    parser.add_argument(
        "--num-pools",
        type=_csv_positive_ints,
        default=(3,),
        metavar="N,...",
        help="MultiPool pool count, or comma-separated counts for a sweep.",
    )
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument("--prediction-accuracy", type=float)
    prediction_group.add_argument(
        "--prediction-accuracies",
        type=_csv_floats,
        metavar="P,...",
        help="NeuScale prediction accuracies for a sensitivity sweep.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=DEFAULT_HOURS,
        help="Trace horizon in hours (default: %(default)s).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=-1,
        help="Request cap for a smoke run; -1 runs the full selected horizon.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Concurrent FleetSim processes (default: %(default)s).",
    )
    parser.add_argument(
        "--n-cpu",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="CPU workers passed to each FleetSim process.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--request-cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--backend-cache-dir", type=Path, default=DEFAULT_BACKEND_CACHE_DIR
    )
    parser.add_argument("--configs-dir", type=Path, default=REPO_ROOT / "configs")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume",
        dest="force",
        action="store_false",
        help="Reuse complete outputs (default).",
    )
    run_mode.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Rerun settings even when complete outputs exist.",
    )
    parser.set_defaults(force=False)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print selected FleetSim commands without running them.",
    )
    return parser


def _unique[T](values: Sequence[T]) -> tuple[T, ...]:
    return tuple(dict.fromkeys(values))


def _parse_caps(raw_caps: Sequence[str] | None) -> tuple[str | None, ...]:
    if not raw_caps:
        return (None,)
    parsed: list[str | None] = []
    for raw in raw_caps:
        value = raw.strip()
        if value.lower() in {"none", "unlimited"}:
            parsed.append(None)
            continue
        entries = value.split(",")
        try:
            valid = all(
                version.strip() and int(count) > 0 and count.strip() == str(int(count))
                for version, count in (entry.split("=", 1) for entry in entries)
            )
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise LauncherError(
                "--max-chips-per-version must be 'unlimited' or entries such "
                "as 5p=256,6e=512"
            )
        parsed.append(",".join(entry.strip() for entry in entries))
    return _unique(parsed)


def resolve(args: argparse.Namespace) -> Config:
    systems = _unique(tuple(args.systems))
    goals = _unique(tuple(args.goals))
    unknown_systems = set(systems) - SUPPORTED_SYSTEMS
    unknown_goals = set(goals) - SUPPORTED_GOALS
    if unknown_systems:
        raise LauncherError(f"unsupported systems: {sorted(unknown_systems)}")
    if unknown_goals:
        raise LauncherError(f"unsupported goals: {sorted(unknown_goals)}")
    if args.hours <= 0:
        raise LauncherError("--hours must be positive")
    if args.max_requests == 0 or args.max_requests < -1:
        raise LauncherError("--max-requests must be -1 or positive")
    if args.jobs <= 0 or args.n_cpu <= 0:
        raise LauncherError("--jobs and --n-cpu must be positive")

    if args.prediction_accuracies is not None:
        accuracies = _unique(tuple(args.prediction_accuracies))
    elif args.prediction_accuracy is not None:
        accuracies = (args.prediction_accuracy,)
    else:
        accuracies = (DEFAULT_PREDICTION_ACCURACY,)
    if any(not 0.0 <= accuracy <= 1.0 for accuracy in accuracies):
        raise LauncherError("prediction accuracy must be within [0, 1]")

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO_ROOT), env.get("PYTHONPATH", "")))
    )
    return Config(
        model=args.model,
        trace_name=args.trace_name,
        trace_file=args.trace_file.expanduser().resolve(),
        systems=systems,
        goals=goals,
        chip_versions=_unique(tuple(args.chip_versions)),
        prefill_chip_versions=(
            _unique(tuple(args.prefill_chip_versions))
            if args.prefill_chip_versions
            else None
        ),
        decode_chip_versions=(
            _unique(tuple(args.decode_chip_versions))
            if args.decode_chip_versions
            else None
        ),
        chip_caps=_parse_caps(args.chip_caps),
        num_pools=_unique(tuple(args.num_pools)),
        prediction_accuracies=accuracies,
        hours=args.hours,
        max_requests=args.max_requests,
        n_cpu=args.n_cpu,
        jobs=args.jobs,
        force=args.force,
        results_dir=args.results_dir.expanduser().resolve(),
        cache_dir=args.request_cache_dir.expanduser().resolve(),
        backend_cache_dir=args.backend_cache_dir.expanduser().resolve(),
        configs_dir=args.configs_dir.expanduser().resolve(),
        python=os.environ.get("NEUSIM_PYTHON") or sys.executable,
        env=env,
    )


def experiment_matrix(config: Config) -> tuple[Experiment, ...]:
    experiments: list[Experiment] = []
    for system, goal, cap in product(config.systems, config.goals, config.chip_caps):
        pools = config.num_pools if system == "MultiPool" else (None,)
        accuracies = (
            config.prediction_accuracies if system in PREDICTION_SYSTEMS else (None,)
        )
        experiments.extend(
            Experiment(system, goal, cap, num_pools, accuracy)
            for num_pools, accuracy in product(pools, accuracies)
        )
    return tuple(experiments)


def _slug(value: object) -> str:
    text = str(value).strip().replace(".", "p")
    return "".join(character if character.isalnum() else "-" for character in text)


def run_dir(config: Config, experiment: Experiment) -> Path:
    parts = [
        config.model,
        config.trace_name,
        experiment.goal,
        f"chips-{'-'.join(config.chip_versions)}",
    ]
    if config.prefill_chip_versions:
        parts.append(f"prefill-{'-'.join(config.prefill_chip_versions)}")
    if config.decode_chip_versions:
        parts.append(f"decode-{'-'.join(config.decode_chip_versions)}")
    parts.append(
        "caps-unlimited"
        if experiment.max_chips_per_version is None
        else f"caps-{experiment.max_chips_per_version}"
    )
    if experiment.num_pools is not None:
        parts.append(f"pools-{experiment.num_pools}")
    if experiment.prediction_accuracy is not None:
        parts.append(f"accuracy-{experiment.prediction_accuracy:g}")
    return (
        config.results_dir
        / _slug(experiment.system)
        / "_".join(_slug(part) for part in parts)
    )


def command(config: Config, experiment: Experiment) -> tuple[str, ...]:
    argv = [
        config.python,
        "-m",
        "neusim.run_scripts.fleetsim_main",
        f"--configs_path={config.configs_dir}",
        f"--request_results_cache_dir={config.cache_dir}",
        ("--npusim_backend_cache_dir=" f"{config.backend_cache_dir / experiment.goal}"),
        "--request_pattern=trace",
        f"--traces_dir={config.trace_file.parent}",
        f"--trace={config.trace_name}",
        f"--trace_file={config.trace_file}",
        f"--model={config.model}",
        f"--system={experiment.system}",
        f"--opt_goal={experiment.goal}",
        f"--chip_versions={','.join(config.chip_versions)}",
        "--allocation_success_rate=1.0",
        f"--max_timestamp_hours={config.hours:g}",
        f"--max_num_requests={config.max_requests}",
        f"--n_cpu={config.n_cpu}",
        f"--output_dir={run_dir(config, experiment)}",
    ]
    if config.prefill_chip_versions:
        argv.append(f"--prefill_chip_versions={','.join(config.prefill_chip_versions)}")
    if config.decode_chip_versions:
        argv.append(f"--decode_chip_versions={','.join(config.decode_chip_versions)}")
    if experiment.max_chips_per_version is not None:
        argv.append(f"--max_chips_per_version={experiment.max_chips_per_version}")
    if experiment.num_pools is not None:
        argv.append(f"--num_pools={experiment.num_pools}")
    if experiment.prediction_accuracy is not None:
        argv.append(f"--output_prediction_accuracy={experiment.prediction_accuracy:g}")
    return tuple(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(root: Path, pattern: str) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob(pattern) if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def run_contract(config: Config, experiment: Experiment, expected: int) -> dict:
    """Fingerprint inputs and the exact command before reusing a run."""
    cache_manifests = {}
    for name in CACHE_MANIFEST_NAMES:
        path = config.cache_dir / name
        if path.is_file() and not path.is_symlink():
            cache_manifests[name] = _sha256_file(path)
    fleetsim_main = REPO_ROOT / "neusim" / "run_scripts" / "fleetsim_main.py"
    return {
        "schema_version": 1,
        "command": list(command(config, experiment)),
        "expected_requests": expected,
        "trace_sha256": _sha256_file(config.trace_file),
        "cache_manifests": cache_manifests,
        "configs_sha256": _sha256_tree(config.configs_dir, "*.json"),
        "fleetsim_main_sha256": _sha256_file(fleetsim_main),
    }


def write_run_contract(directory: Path, contract: Mapping[str, object]) -> Path:
    path = directory / RUN_CONTRACT_NAME
    temporary = directory / f".{RUN_CONTRACT_NAME}.tmp-{os.getpid()}"
    try:
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_inputs(config: Config) -> None:
    required_paths = {
        "trace file": config.trace_file,
        "configuration directory": config.configs_dir,
        "request lookup cache": config.cache_dir,
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_paths.items()
        if not path.exists()
    ]
    model_config = config.configs_dir / "models" / f"{config.model}.json"
    if not model_config.is_file():
        missing.append(f"model configuration: {model_config}")
    all_versions = set(config.chip_versions)
    all_versions.update(config.prefill_chip_versions or ())
    all_versions.update(config.decode_chip_versions or ())
    for version in sorted(all_versions):
        chip_config = config.configs_dir / "chips" / f"tpuv{version}.json"
        if not chip_config.is_file():
            missing.append(f"chip configuration: {chip_config}")
    for goal in config.goals:
        cache_model = config.cache_dir / goal / config.model
        if not cache_model.is_dir():
            missing.append(f"{goal} cache for {config.model}: {cache_model}")
    if missing:
        raise LauncherError("missing required input(s): " + "; ".join(missing))


def _trace_request_count(path: Path, hours: float) -> int:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if {"TIMESTAMP", "ContextTokens", "GeneratedTokens"} <= fields:
                timestamps = [
                    datetime.fromisoformat(row["TIMESTAMP"]).timestamp()
                    for row in reader
                ]
            elif {"Timestamp", "Request tokens", "Response tokens"} <= fields:
                timestamps = [float(row["Timestamp"]) for row in reader]
            else:
                raise LauncherError(
                    f"unsupported request-trace schema: {sorted(fields)}"
                )
    except (OSError, ValueError) as error:
        raise LauncherError(f"cannot read request trace {path}: {error}") from error
    if not timestamps:
        raise LauncherError(f"request trace is empty: {path}")
    horizon_seconds = hours * 60 * 60
    origin = timestamps[0]
    return sum(timestamp - origin <= horizon_seconds for timestamp in timestamps)


def expected_requests(config: Config) -> int:
    expected = _trace_request_count(config.trace_file, config.hours)
    if config.max_requests > 0:
        expected = min(expected, config.max_requests)
    return expected


def inspect_outputs(directory: Path, expected: int) -> bool:
    try:
        stats = json.loads((directory / "stats.json").read_text(encoding="utf-8"))
        csv.field_size_limit(sys.maxsize)
        with (directory / "request_trace.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            reader = csv.reader(stream)
            next(reader)
            rows = sum(1 for _ in reader)
    except (OSError, json.JSONDecodeError, csv.Error, StopIteration):
        return False
    return stats.get("total_requests") == rows == expected


def inspect_complete(
    directory: Path, expected: int, contract: Mapping[str, object]
) -> bool:
    if not inspect_outputs(directory, expected):
        return False
    try:
        saved_contract = json.loads(
            (directory / RUN_CONTRACT_NAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return saved_contract == contract


def _run_one(config: Config, experiment: Experiment, expected: int) -> str:
    directory = run_dir(config, experiment)
    contract = run_contract(config, experiment, expected)
    if not config.force and inspect_complete(directory, expected, contract):
        print(f"reuse {directory}")
        return "reused"

    directory.mkdir(parents=True, exist_ok=True)
    (directory / RUN_CONTRACT_NAME).unlink(missing_ok=True)
    (config.backend_cache_dir / experiment.goal).mkdir(parents=True, exist_ok=True)
    print(f"run   {directory}")
    with (directory / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command(config, experiment),
            cwd=REPO_ROOT,
            env=dict(config.env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
        status = process.wait()
    if status != 0:
        raise LauncherError(f"FleetSim failed ({status}): {directory}")
    if not inspect_outputs(directory, expected):
        raise LauncherError(f"FleetSim output is incomplete: {directory}")
    write_run_contract(directory, contract)
    return "completed"


def run_experiments(
    config: Config, experiments: Sequence[Experiment], expected: int
) -> dict[Experiment, str]:
    states: dict[Experiment, str] = {}
    executor = ThreadPoolExecutor(max_workers=config.jobs)
    futures = {
        executor.submit(_run_one, config, experiment, expected): experiment
        for experiment in experiments
    }
    try:
        for future in as_completed(futures):
            states[futures[future]] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return states


def write_manifest(
    config: Config,
    experiments: Sequence[Experiment],
    expected: int,
    states: Mapping[Experiment, str],
) -> Path:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.results_dir / "ae_run_manifest.json"
    document = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "model": config.model,
        "trace_name": config.trace_name,
        "trace_file": str(config.trace_file),
        "window_hours": config.hours,
        "max_requests": config.max_requests,
        "expected_requests_per_setting": expected,
        "request_cache_dir": str(config.cache_dir),
        "backend_cache_dir": str(config.backend_cache_dir),
        "configs_dir": str(config.configs_dir),
        "jobs": config.jobs,
        "n_cpu": config.n_cpu,
        "experiments": [
            {
                "system": experiment.system,
                "goal": experiment.goal,
                "chip_versions": list(config.chip_versions),
                "prefill_chip_versions": (
                    list(config.prefill_chip_versions)
                    if config.prefill_chip_versions
                    else None
                ),
                "decode_chip_versions": (
                    list(config.decode_chip_versions)
                    if config.decode_chip_versions
                    else None
                ),
                "max_chips_per_version": experiment.max_chips_per_version,
                "num_pools": experiment.num_pools,
                "prediction_accuracy": experiment.prediction_accuracy,
                "output_dir": str(run_dir(config, experiment)),
                "status": states.get(experiment, "planned"),
                "command": list(command(config, experiment)),
            }
            for experiment in experiments
        ],
    }
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve(args)
        experiments = experiment_matrix(config)
        if args.list:
            for index, experiment in enumerate(experiments, 1):
                print(f"[{index}/{len(experiments)}] {run_dir(config, experiment)}")
                print(f"  {shlex.join(command(config, experiment))}")
            return 0

        validate_inputs(config)
        expected = expected_requests(config)
        write_manifest(config, experiments, expected, {})
        states = run_experiments(config, experiments, expected)
        manifest = write_manifest(config, experiments, expected, states)
        print(f"complete: {len(experiments)} settings; manifest: {manifest}")
        return 0
    except (LauncherError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
