#!/usr/bin/env python3
"""Extract and validate the MICRO'26 Azure/DeepSeek sample lookup cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from neusim.run_scripts.package_micro26ae_sample_cache import (
    ARTIFACT_DIR,
    CACHE_ARCHIVE_NAME,
    CACHE_DIR_NAME,
    DEFAULT_TRACE,
    GOALS,
    MANIFEST_NAME,
    MODEL,
    PHASES,
    VERSIONS,
    WINDOW_HOURS,
    _assert_runtime_coverage,
    _relative_leaf,
    load_sequence_pairs,
    sha256_file,
    validate_document,
)

DEFAULT_TARGET = ARTIFACT_DIR / "request_lookup_cache_deepseekv3_azure_3h"


class PreparationError(ValueError):
    """Raised when the sample cache fails structural or content validation."""


def _read_manifest(cache_dir: Path) -> dict:
    path = cache_dir / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise PreparationError(f"cache manifest is not an object: {path}")
    return manifest


def validate_cache(cache_dir: Path, trace: Path = DEFAULT_TRACE) -> dict:
    """Validate exact dimensions, trace identity, rank, and runtime fields."""
    cache_dir = cache_dir.expanduser().resolve()
    trace = trace.expanduser().resolve()
    if cache_dir.is_symlink() or not cache_dir.is_dir():
        raise PreparationError(f"cache is not a regular directory: {cache_dir}")
    manifest = _read_manifest(cache_dir)
    expected_contract = {
        "schema_version": 1,
        "model": MODEL,
        "goals": list(GOALS),
        "versions": list(VERSIONS),
        "phases": list(PHASES),
        "rank": 1,
        "operator_csv_required": False,
    }
    for key, expected in expected_contract.items():
        if manifest.get(key) != expected:
            raise PreparationError(
                f"cache manifest has invalid {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )

    pairs, request_count = load_sequence_pairs(trace)
    trace_record = manifest.get("trace")
    if not isinstance(trace_record, Mapping):
        raise PreparationError("cache manifest has no trace record")
    expected_trace = {
        "path": trace.name,
        "sha256": sha256_file(trace),
        "window_hours": WINDOW_HOURS,
        "requests": request_count,
        "padded_sequence_pairs": len(pairs),
    }
    for key, expected in expected_trace.items():
        if trace_record.get(key) != expected:
            raise PreparationError(
                f"cache trace mismatch for {key}: "
                f"{trace_record.get(key)!r} != {expected!r}"
            )

    coverage = manifest.get("coverage")
    if not isinstance(coverage, Mapping):
        raise PreparationError("cache manifest has no coverage record")
    manifest_path = cache_dir / MANIFEST_NAME
    json_files = sorted(cache_dir.glob("*/*/*/*/*/1.json"))
    expected_files = coverage.get("json_files")
    if expected_files != len(json_files):
        raise PreparationError(
            f"cache JSON count mismatch: expected {expected_files}, "
            f"found {len(json_files)}"
        )
    allowed = {manifest_path, *json_files}
    for path in cache_dir.rglob("*"):
        if path.is_symlink():
            raise PreparationError(f"cache contains a symlink: {path}")
        if path.is_file() and path not in allowed:
            raise PreparationError(f"cache contains an unexpected file: {path}")

    expected_pairs = set(pairs)
    dimensions: set[tuple[str, str, str]] = set()
    leaves: set[tuple[str, tuple[int, int], str, str]] = set()
    by_dimension: dict[str, int] = {}
    for path in json_files:
        relative = path.relative_to(cache_dir)
        parts = relative.parts
        if len(parts) != 6:
            raise PreparationError(f"invalid cache path: {relative}")
        goal, model, pair_text, version, phase, rank_file = parts
        try:
            pair_values = tuple(int(value) for value in pair_text.split("_"))
        except ValueError as error:
            raise PreparationError(f"invalid sequence pair path: {relative}") from error
        if len(pair_values) != 2:
            raise PreparationError(f"invalid sequence pair path: {relative}")
        pair = (pair_values[0], pair_values[1])
        if (
            goal not in GOALS
            or model != MODEL
            or pair not in expected_pairs
            or version not in VERSIONS
            or phase not in PHASES
            or rank_file != "1.json"
        ):
            raise PreparationError(f"cache path is outside sample contract: {relative}")
        try:
            document = validate_document(
                path.read_bytes(),
                _relative_leaf(goal, pair, version, phase),
                pair=pair,
                version=version,
                goal=goal,
                phase=phase,
            )
        except ValueError as error:
            raise PreparationError(str(error)) from error
        if document is None:
            raise PreparationError(f"cache contains an infeasible record: {relative}")
        dimensions.add((goal, version, phase))
        leaves.add((goal, pair, version, phase))
        dimension = f"{goal}/{version}/{phase}"
        by_dimension[dimension] = by_dimension.get(dimension, 0) + 1
    expected_dimensions = {
        (goal, version, phase)
        for goal in GOALS
        for version in VERSIONS
        for phase in PHASES
    }
    if dimensions != expected_dimensions:
        raise PreparationError(
            f"cache dimensions mismatch: {dimensions} != {expected_dimensions}"
        )
    try:
        fallback_lookups = _assert_runtime_coverage(leaves, pairs)
    except ValueError as error:
        raise PreparationError(str(error)) from error
    expected_coverage = {
        "unavailable_or_infeasible_leaves": (
            len(pairs) * len(GOALS) * len(VERSIONS) * len(PHASES) - len(json_files)
        ),
        "lookups_requiring_smaller_sequence_fallback": fallback_lookups,
        "by_dimension": dict(sorted(by_dimension.items())),
    }
    for key, expected in expected_coverage.items():
        if coverage.get(key) != expected:
            raise PreparationError(
                f"cache coverage mismatch for {key}: "
                f"{coverage.get(key)!r} != {expected!r}"
            )
    return manifest


def _validate_members(archive: zipfile.ZipFile) -> None:
    seen: set[str] = set()
    required_manifest = f"{CACHE_DIR_NAME}/{MANIFEST_NAME}"
    for member in archive.infolist():
        pure = PurePosixPath(member.filename)
        if (
            not pure.parts
            or pure.is_absolute()
            or ".." in pure.parts
            or pure.parts[0] != CACHE_DIR_NAME
        ):
            raise PreparationError(f"unsafe archive member: {member.filename}")
        file_type = stat.S_IFMT(member.external_attr >> 16)
        if member.is_dir():
            if file_type not in (0, stat.S_IFDIR):
                raise PreparationError(
                    f"unsafe archive directory member: {member.filename}"
                )
            continue
        if len(pure.parts) < 2 or file_type not in (0, stat.S_IFREG):
            raise PreparationError(
                f"unsafe or non-file archive member: {member.filename}"
            )
        if member.filename in seen:
            raise PreparationError(f"duplicate archive member: {member.filename}")
        seen.add(member.filename)
    if required_manifest not in seen:
        raise PreparationError(f"cache archive is missing {required_manifest}")


def prepare_cache(
    archive_path: Path,
    target: Path = DEFAULT_TARGET,
    trace: Path = DEFAULT_TRACE,
) -> dict:
    """Reuse a valid target or atomically publish a validated ZIP extraction."""
    archive_path = archive_path.expanduser().resolve()
    target = target.expanduser().resolve()
    trace = trace.expanduser().resolve()
    if target.exists():
        manifest = validate_cache(target, trace)
        print(f"Reusing validated sample cache: {target}")
        return manifest
    if archive_path.is_symlink() or not archive_path.is_file():
        raise PreparationError(f"cache archive is missing: {archive_path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                _validate_members(archive)
                archive.extractall(temporary)
        except zipfile.BadZipFile as error:
            raise PreparationError(
                f"cannot open cache ZIP {archive_path}: {error}"
            ) from error
        extracted = temporary / CACHE_DIR_NAME
        manifest = validate_cache(extracted, trace)
        os.replace(extracted, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    print(
        f"Prepared {target}: {manifest['coverage']['json_files']} "
        "rank-one JSON files"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help=(
            "Cache ZIP to extract. Omit after the archive has already been "
            f"unpacked to --target (release name: {CACHE_ARCHIVE_NAME})."
        ),
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --target without extracting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        manifest = validate_cache(args.target, args.trace)
        print(f"Valid: {manifest['coverage']['json_files']} JSON files")
    elif args.archive is not None:
        prepare_cache(args.archive, args.target, args.trace)
    elif args.target.exists():
        manifest = validate_cache(args.target, args.trace)
        print(f"Valid: {manifest['coverage']['json_files']} JSON files")
    else:
        parser.error(
            "--archive is required until the external cache ZIP has been "
            "extracted to --target"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
