#!/usr/bin/env python3
"""Package the compact lookup cache used by the MICRO'26 sample replay.

The package contains only rank-one JSON records for the provided three-hour
sampled Azure trace, DeepSeek-V3-671B, NPU v5p/v6e, energy/monetary objectives,
and prefill/decode. ``--source`` may be either a full cache directory or an
existing cache archive. The output is a deterministic ZIP with the cache
directory as its single top-level entry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from neusim.fleetsim.util import pad_seqlen
from neusim.fleetsim.vPodAutoScaler_lib import get_ordered_seqlen_fallbacks

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifact_eval" / "micro26"
DATA_DIR = ARTIFACT_DIR / "data"
DEFAULT_TRACE = DATA_DIR / "AzureLLMInferenceTrace_code_3h_sampled.csv"
TRACE_ARCHIVE_NAME = "AzureLLMInferenceTrace_code_3h_sampled.zip"
CACHE_ARCHIVE_NAME = "request_lookup_cache_deepseekv3_azure_3h_v5p_v6e.zip"
CACHE_DIR_NAME = "request_lookup_cache_deepseekv3_azure_3h"
MANIFEST_NAME = "micro26ae_sample_cache_manifest.json"

WINDOW_HOURS = 3.0
MODEL = "deepseekv3-671b"
SLO_SCALE = 2.0
GOALS = ("energy", "monetary")
VERSIONS = ("5p", "6e")
PHASES = ("prefill", "decode")
INPUT_FACTORS = (32, 64, 128, 512, 1024, 4096, 8192, 16384, 32768)
INPUT_STEPS = (128, 512, 1024, 8192, 16384, 65536, 131072, 262144)
OUTPUT_FACTORS = (4, 16, 32, 64, 128, 256, 512, 1024)
OUTPUT_STEPS = (32, 64, 128, 512, 1024, 2048, 8192)


class PackagingError(ValueError):
    """Raised when the source cannot provide a safe, runnable sample cache."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pad_pair(input_length: int, output_length: int) -> tuple[int, int]:
    return (
        pad_seqlen(input_length, INPUT_FACTORS, INPUT_STEPS),
        pad_seqlen(max(2, output_length), OUTPUT_FACTORS, OUTPUT_STEPS),
    )


def _seconds(
    value: str, reference: datetime | int | None
) -> tuple[float, datetime | int]:
    try:
        timestamp = int(value)
    except ValueError as integer_error:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as datetime_error:
            raise PackagingError(
                f"invalid trace timestamp {value!r}"
            ) from datetime_error
        origin = parsed if reference is None else reference
        if not isinstance(origin, datetime):
            raise PackagingError(
                "mixed timestamp formats in one trace"
            ) from integer_error
        return (parsed - origin).total_seconds(), origin
    origin = timestamp if reference is None else reference
    if not isinstance(origin, int):
        raise PackagingError("mixed timestamp formats in one trace")
    return float(timestamp - origin), origin


def load_sequence_pairs(
    path: Path, window_hours: float = WINDOW_HOURS
) -> tuple[list[tuple[int, int]], int]:
    """Return padded runtime/allocation pairs and requests in the time window."""
    if window_hours <= 0:
        raise PackagingError("window_hours must be greater than zero")
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as error:
        raise PackagingError(f"cannot read Azure trace {path}: {error}") from error
    with stream:
        reader = csv.DictReader(stream)
        required = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        if not required <= set(reader.fieldnames or ()):
            raise PackagingError(f"unsupported Azure trace schema: {path}")
        reference: datetime | int | None = None
        observed: list[tuple[int, int]] = []
        for row in reader:
            offset, reference = _seconds(row["TIMESTAMP"], reference)
            if offset > window_hours * 3600:
                continue
            observed.append(
                _pad_pair(int(row["ContextTokens"]), int(row["GeneratedTokens"]))
            )
    if not observed:
        raise PackagingError(f"trace contains no requests in the window: {path}")

    augmented = list(observed)
    augmented.append(
        _pad_pair(
            math.ceil(np.mean([pair[0] for pair in observed])),
            math.ceil(np.mean([pair[1] for pair in observed])),
        )
    )
    augmented.append(
        _pad_pair(
            math.ceil(
                np.mean([pair[0] for pair in augmented])
                + 0.25 * np.std([pair[0] for pair in augmented])
            ),
            math.ceil(
                np.mean([pair[1] for pair in augmented])
                + 0.25 * np.std([pair[1] for pair in augmented])
            ),
        )
    )
    pairs = set(augmented)
    pairs.update((input_length, 4) for input_length, _ in tuple(pairs))
    return sorted(pairs), len(observed)


def _relative_leaf(
    goal: str, pair: tuple[int, int], version: str, phase: str
) -> PurePosixPath:
    return PurePosixPath(goal, MODEL, f"{pair[0]}_{pair[1]}", version, phase, "1.json")


class _Source(Protocol):
    kind: str

    def prime(self, paths: Iterable[PurePosixPath]) -> None:
        ...

    def read(self, path: PurePosixPath) -> bytes | None:
        ...

    def close(self) -> None:
        ...


class _DirectorySource:
    kind = "directory"

    def __init__(self, root: Path):
        if root.is_symlink() or not root.is_dir():
            raise PackagingError(f"source is not a regular directory: {root}")
        self.root = root

    def prime(self, paths: Iterable[PurePosixPath]) -> None:
        return None

    def read(self, path: PurePosixPath) -> bytes | None:
        candidate = self.root.joinpath(*path.parts)
        if not candidate.exists():
            return None
        if candidate.is_symlink() or not candidate.is_file():
            raise PackagingError(f"cache leaf is not a regular file: {candidate}")
        return candidate.read_bytes()

    def close(self) -> None:
        return None


def _source_member_name(name: str) -> PurePosixPath:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts:
        raise PackagingError(f"unsafe source archive member: {name}")
    if pure.parts and pure.parts[0] == CACHE_DIR_NAME:
        pure = PurePosixPath(*pure.parts[1:])
    if not pure.parts:
        raise PackagingError(f"invalid source archive member: {name}")
    return pure


class _TarSource:
    kind = "archive"

    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_file():
            raise PackagingError(f"source archive is missing: {path}")
        try:
            self.archive = tarfile.open(path, "r:*")
        except (OSError, tarfile.TarError) as error:
            raise PackagingError(
                f"cannot open source archive {path}: {error}"
            ) from error
        self.members: dict[str, tarfile.TarInfo] = {}
        self.payloads: dict[str, bytes] = {}
        try:
            for member in self.archive.getmembers():
                if member.isdir():
                    directory = PurePosixPath(member.name)
                    if directory.is_absolute() or ".." in directory.parts:
                        raise PackagingError(
                            f"unsafe source archive member: {member.name}"
                        )
                    continue
                pure = _source_member_name(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise PackagingError(
                        f"source archive links/special files are forbidden: "
                        f"{member.name}"
                    )
                key = pure.as_posix()
                if key in self.members:
                    raise PackagingError(
                        f"duplicate source archive member: {member.name}"
                    )
                self.members[key] = member
        except Exception:
            self.archive.close()
            raise

    def prime(self, paths: Iterable[PurePosixPath]) -> None:
        """Read selected gzip members in archive order to avoid repeated seeks."""
        wanted = {path.as_posix() for path in paths}
        for key, member in self.members.items():
            if key not in wanted:
                continue
            stream = self.archive.extractfile(member)
            if stream is None:
                raise PackagingError(
                    f"cannot read source archive member: {member.name}"
                )
            self.payloads[key] = stream.read()

    def read(self, path: PurePosixPath) -> bytes | None:
        return self.payloads.get(path.as_posix())

    def close(self) -> None:
        self.archive.close()


class _ZipSource:
    kind = "archive"

    def __init__(self, path: Path):
        if path.is_symlink() or not path.is_file():
            raise PackagingError(f"source archive is missing: {path}")
        try:
            self.archive = zipfile.ZipFile(path)
        except (OSError, zipfile.BadZipFile) as error:
            raise PackagingError(
                f"cannot open source archive {path}: {error}"
            ) from error
        self.members: dict[str, zipfile.ZipInfo] = {}
        self.payloads: dict[str, bytes] = {}
        try:
            for member in self.archive.infolist():
                file_type = stat.S_IFMT(member.external_attr >> 16)
                if member.is_dir():
                    directory = PurePosixPath(member.filename)
                    if (
                        directory.is_absolute()
                        or ".." in directory.parts
                        or file_type not in (0, stat.S_IFDIR)
                    ):
                        raise PackagingError(
                            f"unsafe source archive member: {member.filename}"
                        )
                    continue
                pure = _source_member_name(member.filename)
                if file_type not in (0, stat.S_IFREG):
                    raise PackagingError(
                        "source archive links/special files are forbidden: "
                        f"{member.filename}"
                    )
                key = pure.as_posix()
                if key in self.members:
                    raise PackagingError(
                        f"duplicate source archive member: {member.filename}"
                    )
                self.members[key] = member
        except Exception:
            self.archive.close()
            raise

    def prime(self, paths: Iterable[PurePosixPath]) -> None:
        wanted = {path.as_posix() for path in paths}
        for key, member in self.members.items():
            if key in wanted:
                self.payloads[key] = self.archive.read(member)

    def read(self, path: PurePosixPath) -> bytes | None:
        return self.payloads.get(path.as_posix())

    def close(self) -> None:
        self.archive.close()


def open_source(path: Path) -> _Source:
    if path.is_dir():
        return _DirectorySource(path)
    if zipfile.is_zipfile(path):
        return _ZipSource(path)
    return _TarSource(path)


def _positive_number(
    document: Mapping[str, Any], key: str, path: PurePosixPath
) -> bool:
    value = document.get(key)
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def validate_document(
    payload: bytes,
    path: PurePosixPath,
    *,
    pair: tuple[int, int],
    version: str,
    goal: str,
    phase: str,
) -> dict[str, Any] | None:
    """Validate one source record; return ``None`` for an infeasible leaf."""
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PackagingError(f"invalid cache JSON {path}: {error}") from error
    if not isinstance(document, dict):
        raise PackagingError(f"cache JSON is not an object: {path}")
    sim_config = document.get("sim_config")
    if not isinstance(sim_config, Mapping):
        raise PackagingError(f"cache JSON lacks sim_config: {path}")
    expected = (MODEL, version, pair[0], pair[1])
    actual = (
        sim_config.get("model_name"),
        sim_config.get("name"),
        sim_config.get("input_seqlen"),
        sim_config.get("output_seqlen"),
    )
    if actual != expected:
        raise PackagingError(
            f"cache identity mismatch in {path}: {actual} != {expected}"
        )

    slo_key = "slo_TTFT_sec" if phase == "prefill" else "slo_TPOT_ms_request"
    objective_key = (
        "avg_power_efficiency_tkn_per_joule"
        if goal == "energy"
        else "monetary_cost_tkn_per_dollar"
    )
    if document.get("out_of_memory") is True:
        return None
    if document.get("slo_scale") != SLO_SCALE:
        raise PackagingError(
            f"cache SLO scale mismatch in {path}: "
            f"{document.get('slo_scale')!r} != {SLO_SCALE!r}"
        )
    if not _positive_number(document, slo_key, path) or not _positive_number(
        document, objective_key, path
    ):
        return None
    return document


def _assert_runtime_coverage(
    leaves: set[tuple[str, tuple[int, int], str, str]],
    pairs: Sequence[tuple[int, int]],
) -> int:
    fallback_lookups = 0
    available_pairs = {
        (goal, phase): sorted(
            {
                pair
                for leaf_goal, pair, _version, leaf_phase in leaves
                if leaf_goal == goal and leaf_phase == phase
            }
        )
        for goal in GOALS
        for phase in PHASES
    }
    failures: list[str] = []
    for goal in GOALS:
        for phase in PHASES:
            for requested in pairs:
                candidates = (
                    requested,
                    *get_ordered_seqlen_fallbacks(
                        available_pairs[(goal, phase)],
                        requested[0],
                        requested[1],
                        phase,
                    ),
                )
                if not any(
                    (goal, candidate, version, phase) in leaves
                    for candidate in candidates
                    for version in VERSIONS
                ):
                    failures.append(f"{goal}/{requested}/{phase}")
                elif not any(
                    (goal, requested, version, phase) in leaves for version in VERSIONS
                ):
                    fallback_lookups += 1
    if failures:
        raise PackagingError(
            "source cache cannot serve required sample lookups: "
            + ", ".join(failures[:10])
        )
    return fallback_lookups


def build_tree(
    target: Path, *, source: _Source, trace: Path = DEFAULT_TRACE
) -> dict[str, Any]:
    pairs, request_count = load_sequence_pairs(trace)
    source.prime(
        _relative_leaf(goal, pair, version, phase)
        for goal in GOALS
        for pair in pairs
        for version in VERSIONS
        for phase in PHASES
    )
    copied = 0
    unavailable = 0
    leaves: set[tuple[str, tuple[int, int], str, str]] = set()
    by_dimension: dict[str, int] = {}
    for goal in GOALS:
        for pair in pairs:
            for version in VERSIONS:
                for phase in PHASES:
                    relative = _relative_leaf(goal, pair, version, phase)
                    payload = source.read(relative)
                    if payload is None:
                        unavailable += 1
                        continue
                    document = validate_document(
                        payload,
                        relative,
                        pair=pair,
                        version=version,
                        goal=goal,
                        phase=phase,
                    )
                    if document is None:
                        unavailable += 1
                        continue
                    destination = target.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(
                        json.dumps(document, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    leaves.add((goal, pair, version, phase))
                    key = f"{goal}/{version}/{phase}"
                    by_dimension[key] = by_dimension.get(key, 0) + 1
                    copied += 1

    fallback_lookups = _assert_runtime_coverage(leaves, pairs)
    manifest = {
        "schema_version": 1,
        "purpose": "MICRO 2026 Figures 18 and 19 three-hour sample replay",
        "trace": {
            "path": trace.name,
            "sha256": sha256_file(trace),
            "window_hours": WINDOW_HOURS,
            "requests": request_count,
            "padded_sequence_pairs": len(pairs),
        },
        "model": MODEL,
        "goals": list(GOALS),
        "versions": list(VERSIONS),
        "phases": list(PHASES),
        "rank": 1,
        "operator_csv_required": False,
        "coverage": {
            "json_files": copied,
            "unavailable_or_infeasible_leaves": unavailable,
            "lookups_requiring_smaller_sequence_fallback": fallback_lookups,
            "by_dimension": dict(sorted(by_dimension.items())),
        },
    }
    (target / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _archive_members(root: Path) -> Iterable[Path]:
    yield from sorted(path for path in root.rglob("*") if path.is_file())


def write_deterministic_archive(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in _archive_members(root):
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(
                    f"{CACHE_DIR_NAME}/{relative}",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def package_cache(source_path: Path, trace: Path, output: Path) -> dict[str, Any]:
    source_path = source_path.expanduser().resolve()
    trace = trace.expanduser().resolve()
    output = output.expanduser().resolve()
    if source_path == output:
        raise PackagingError("--source and --output must be different paths")
    source = open_source(source_path)
    try:
        with tempfile.TemporaryDirectory(prefix="micro26ae-sample-cache-") as temporary:
            root = Path(temporary)
            manifest = build_tree(root, source=source, trace=trace)
            write_deterministic_archive(root, output)
    finally:
        source.close()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Full cache directory or existing tar/ZIP cache archive.",
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=f"Output ZIP (release name: {CACHE_ARCHIVE_NAME}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = package_cache(args.source, args.trace, args.output)
    output = args.output.expanduser().resolve()
    print(
        f"Wrote {output}: {manifest['coverage']['json_files']} rank-one JSON files, "
        f"sha256={sha256_file(output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
