#!/usr/bin/env python3
"""Run the two Stage-A passes needed to build a FleetSim lookup cache.

Pass one evaluates NPU configurations and stores simulation results under
``OUTPUT/.cache``.  Pass two ranks those results for both energy and monetary
objectives and writes the FleetSim lookup tree under ``OUTPUT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results" / "fleetsim" / "request_lookup_cache_generated"
)
STAGE_A_MODULE = "neusim.run_scripts.run_sim_find_optimal"
GENERATION_MANIFEST = "fleetsim_cache_generation_manifest.json"


class GenerationError(ValueError):
    """Raised when a two-pass generation request is invalid."""


@dataclass(frozen=True)
class Config:
    traces: tuple[Path, ...]
    models: tuple[str, ...]
    versions: tuple[str, ...]
    output_dir: Path
    configs_dir: Path
    python: str
    num_chips: tuple[int, ...]
    inference_batch_sizes: tuple[int, ...]
    slo_scale: float
    slo_baseline_version: str
    max_pp: int
    top_k: int
    skip_existing: bool
    debug: bool
    dry_run: bool


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in _csv_strings(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be integers") from error
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive")
    return result


def _top_k(value: str) -> int:
    """Parse a positive rank limit or the all-alternatives sentinel."""
    if value.lower() == "all":
        return -1
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected 'all' or a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected 'all' or a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        action="append",
        type=Path,
        required=True,
        help="Input request trace; repeat for multiple traces.",
    )
    parser.add_argument("--models", type=_csv_strings, default=("deepseekv3-671b",))
    parser.add_argument("--versions", type=_csv_strings, default=("5p", "6e"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--configs-dir", type=Path, default=DEFAULT_CONFIGS_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--num-chips",
        type=_csv_positive_ints,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )
    parser.add_argument(
        "--inference-batch-sizes",
        type=_csv_positive_ints,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
    )
    parser.add_argument("--slo-scale", type=float, default=2.0)
    parser.add_argument("--slo-baseline-version", default="5p")
    parser.add_argument(
        "--max-pp",
        type=int,
        default=-1,
        help="-1 leaves pipeline parallelism unrestricted; 1 disables it.",
    )
    parser.add_argument(
        "--top-k",
        type=_top_k,
        default=-1,
        metavar="N|all",
        help="Ranked configs retained per version/phase (default: all).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print both commands without running them.",
    )
    return parser


def resolve(args: argparse.Namespace) -> Config:
    traces = tuple(path.expanduser().resolve() for path in args.trace)
    missing = [path for path in traces if not path.is_file()]
    if missing:
        raise GenerationError(f"trace file does not exist: {missing[0]}")
    configs_dir = args.configs_dir.expanduser().resolve()
    if not configs_dir.is_dir():
        raise GenerationError(f"configs directory does not exist: {configs_dir}")
    if args.slo_scale <= 0:
        raise GenerationError("--slo-scale must be greater than zero")
    if args.max_pp == 0 or args.max_pp < -1:
        raise GenerationError("--max-pp must be -1 or a positive integer")
    versions = tuple(args.versions)
    if args.slo_baseline_version not in versions:
        raise GenerationError("--slo-baseline-version must be in --versions")
    return Config(
        traces=traces,
        models=tuple(args.models),
        versions=versions,
        output_dir=args.output_dir.expanduser().resolve(),
        top_k=args.top_k,
        configs_dir=configs_dir,
        python=args.python,
        num_chips=tuple(args.num_chips),
        inference_batch_sizes=tuple(args.inference_batch_sizes),
        slo_scale=args.slo_scale,
        slo_baseline_version=args.slo_baseline_version,
        max_pp=args.max_pp,
        skip_existing=args.skip_existing,
        debug=args.debug,
        dry_run=args.dry_run,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_configs(configs_dir: Path) -> str:
    """Hash every JSON config with its repository-relative name."""
    digest = hashlib.sha256()
    files = sorted(path for path in configs_dir.rglob("*.json") if path.is_file())
    if not files:
        raise GenerationError(f"configs directory has no JSON files: {configs_dir}")
    for path in files:
        digest.update(path.relative_to(configs_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def generation_contract(config: Config) -> dict:
    """Describe every input that can change cached Stage-A results/ranking."""
    stage_a_path = REPO_ROOT.joinpath(*STAGE_A_MODULE.split(".")).with_suffix(".py")
    return {
        "schema_version": 1,
        "stage_a_module": STAGE_A_MODULE,
        "stage_a_sha256": _sha256_file(stage_a_path),
        "traces": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in config.traces
        ],
        "optimal_top_k": config.top_k,
        "models": list(config.models),
        "versions": list(config.versions),
        "configs_sha256": _sha256_configs(config.configs_dir),
        "num_chips": list(config.num_chips),
        "inference_batch_sizes": list(config.inference_batch_sizes),
        "slo_scale": config.slo_scale,
        "slo_baseline_version": config.slo_baseline_version,
        "max_pp": config.max_pp,
        "debug": config.debug,
    }


def ensure_generation_contract(config: Config) -> Path:
    """Create or verify the output contract before any reusable cache is read."""
    output = config.output_dir
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise GenerationError(f"output is not a regular directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / GENERATION_MANIFEST
    contract = generation_contract(config)
    entries = list(output.iterdir())
    if entries:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise GenerationError(
                f"output directory is not empty and has no generation contract: "
                f"{output}; choose a fresh --output-dir"
            )
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise GenerationError(
                f"cannot read generation contract {manifest_path}: {error}"
            ) from error
        if existing != contract:
            raise GenerationError(
                f"generation contract does not match cached output {output}; "
                "choose a fresh --output-dir"
            )
        return manifest_path

    temporary = output / f".{GENERATION_MANIFEST}.tmp-{os.getpid()}"
    try:
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def command(config: Config, *, generate_trace: bool) -> tuple[str, ...]:
    argv = [
        config.python,
        "-m",
        STAGE_A_MODULE,
        *(f"--request_trace_file={path}" for path in config.traces),
        f"--models={','.join(config.models)}",
        f"--versions={','.join(config.versions)}",
        f"--num_chips={','.join(str(value) for value in config.num_chips)}",
        (
            "--inference_batch_sizes="
            + ",".join(str(value) for value in config.inference_batch_sizes)
        ),
        "--workload=inference",
        f"--output_dir={config.output_dir}",
        f"--configs_path={config.configs_dir}",
        f"--skip_exist={'true' if config.skip_existing else 'false'}",
        f"--debug={'true' if config.debug else 'false'}",
        f"--slo_scale={config.slo_scale:g}",
        f"--slo_baseline_version={config.slo_baseline_version}",
        f"--max_pp={config.max_pp}",
        f"--optimal_top_k={config.top_k}",
        f"--generate_trace={'true' if generate_trace else 'false'}",
        f"--generate_opt_results={'false' if generate_trace else 'true'}",
    ]
    return tuple(os.fspath(value) for value in argv)


def run(
    config: Config,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> None:
    commands = (
        command(config, generate_trace=True),
        command(config, generate_trace=False),
    )
    if not config.dry_run:
        manifest = ensure_generation_contract(config)
        print(f"Generation contract: {manifest}", flush=True)
    for index, argv in enumerate(commands, start=1):
        print(f"Stage-A pass {index}/2: {shlex.join(argv)}", flush=True)
        if not config.dry_run:
            runner(argv, check=True, cwd=REPO_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    config = resolve(build_parser().parse_args(argv))
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
