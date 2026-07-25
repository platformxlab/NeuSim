from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from neusim.run_scripts import generate_fleetsim_optimal_cache as generate
from neusim.run_scripts import package_micro26ae_sample_cache as package
from neusim.run_scripts import prepare_micro26ae_sample_cache as prepare


def _trace(path: Path) -> Path:
    path.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n" "0,17,3\n" "22000,100000,1000\n",
        encoding="utf-8",
    )
    return path


def _document(goal: str, pair: tuple[int, int], version: str, phase: str) -> dict:
    document = {
        "sim_config": {
            "model_name": package.MODEL,
            "name": version,
            "input_seqlen": pair[0],
            "output_seqlen": pair[1],
            "num_chips": 1,
        },
        "out_of_memory": False,
        "slo_scale": package.SLO_SCALE,
        "avg_power_efficiency_tkn_per_joule": 2.0,
        "monetary_cost_tkn_per_dollar": 3.0,
    }
    document["slo_TTFT_sec" if phase == "prefill" else "slo_TPOT_ms_request"] = 1.0
    return document


def _source_tree(root: Path, trace: Path) -> None:
    pairs, _ = package.load_sequence_pairs(trace)
    for goal in package.GOALS:
        for pair in pairs:
            for version in package.VERSIONS:
                for phase in package.PHASES:
                    relative = package._relative_leaf(goal, pair, version, phase)
                    path = root.joinpath(*relative.parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        json.dumps(_document(goal, pair, version, phase)),
                        encoding="utf-8",
                    )


@pytest.mark.skipif(
    not package.DEFAULT_TRACE.is_file(),
    reason="external AE trace archive has not been extracted",
)
def test_external_sample_contract_has_expected_three_hour_coverage() -> None:
    pairs, requests = package.load_sequence_pairs(package.DEFAULT_TRACE)
    assert len(pairs) == 368
    assert requests == 3460


def test_package_from_directory_and_prepare_round_trip(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    source = tmp_path / "full-cache"
    _source_tree(source, trace)
    archive = tmp_path / "sample.zip"

    manifest = package.package_cache(source, trace, archive)

    assert manifest["model"] == package.MODEL
    assert manifest["coverage"]["json_files"] == 8
    assert manifest["coverage"]["unavailable_or_infeasible_leaves"] == 0
    with zipfile.ZipFile(archive) as stream:
        names = stream.namelist()
    assert f"{package.CACHE_DIR_NAME}/{package.MANIFEST_NAME}" in names
    assert {name.split("/", 1)[0] for name in names} == {package.CACHE_DIR_NAME}
    assert len(names) == 9

    target = tmp_path / "prepared"
    prepared = prepare.prepare_cache(archive, target, trace)
    assert prepared == manifest
    assert prepare.validate_cache(target, trace) == manifest


def test_packager_accepts_an_existing_archive_source(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    source = tmp_path / "full-cache"
    _source_tree(source, trace)
    source_archive = tmp_path / "source.zip"
    package.write_deterministic_archive(source, source_archive)
    output = tmp_path / "sample.zip"

    manifest = package.package_cache(source_archive, trace, output)

    assert manifest["coverage"]["json_files"] == 8


def test_preparer_rejects_wrong_zip_root(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    archive = tmp_path / "wrong-root.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(f"wrong/{package.MANIFEST_NAME}", "{}")

    with pytest.raises(prepare.PreparationError, match="unsafe archive member"):
        prepare.prepare_cache(archive, tmp_path / "prepared", trace)


def test_preparer_rejects_content_outside_sample_contract(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    source = tmp_path / "full-cache"
    _source_tree(source, trace)
    archive = tmp_path / "sample.zip"
    package.package_cache(source, trace, archive)
    target = tmp_path / "prepared"
    prepare.prepare_cache(archive, target, trace)
    extra = target / "energy" / "other-model" / "32_4" / "5p" / "prefill"
    extra.mkdir(parents=True)
    (extra / "1.json").write_text("{}")

    with pytest.raises(prepare.PreparationError, match="count mismatch|outside"):
        prepare.validate_cache(target, trace)


def test_preparer_rejects_missing_runtime_pair_with_edited_manifest(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "Azure_multi_pair.csv"
    trace.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n" "0,17,3\n" "1,100000,1000\n",
        encoding="utf-8",
    )
    source = tmp_path / "full-cache"
    _source_tree(source, trace)
    archive = tmp_path / "sample.zip"
    package.package_cache(source, trace, archive)
    target = tmp_path / "prepared"
    prepare.prepare_cache(archive, target, trace)

    pairs, _ = package.load_sequence_pairs(trace)
    removed_pair = pairs[0]
    for version in package.VERSIONS:
        package_path = target.joinpath(
            *package._relative_leaf("energy", removed_pair, version, "prefill").parts
        )
        package_path.unlink()

    manifest_path = target / package.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["coverage"]["json_files"] -= len(package.VERSIONS)
    manifest["coverage"]["unavailable_or_infeasible_leaves"] += len(package.VERSIONS)
    for version in package.VERSIONS:
        manifest["coverage"]["by_dimension"][f"energy/{version}/prefill"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(prepare.PreparationError, match="cannot serve required"):
        prepare.validate_cache(target, trace)


def test_two_pass_launcher_builds_complementary_stage_commands(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    parser = generate.build_parser()
    config = generate.resolve(
        parser.parse_args(
            [
                "--trace",
                str(trace),
                "--models",
                "deepseekv3-671b,llama3-70b",
                "--versions",
                "5p,6e",
                "--output-dir",
                str(tmp_path / "cache"),
                "--configs-dir",
                str(generate.DEFAULT_CONFIGS_DIR),
                "--max-pp",
                "1",
            ]
        )
    )
    first = generate.command(config, generate_trace=True)
    second = generate.command(config, generate_trace=False)

    assert "--generate_trace=true" in first
    assert "--generate_opt_results=false" in first
    assert "--generate_trace=false" in second
    assert "--generate_opt_results=true" in second
    assert f"--request_trace_file={trace.resolve()}" in first
    assert "--models=deepseekv3-671b,llama3-70b" in first
    assert "--versions=5p,6e" in first
    assert "--max_pp=1" in first
    assert "--slo_scale=2" in first
    assert config.top_k == -1
    assert "--optimal_top_k=-1" in first
    assert "--optimal_top_k=-1" in second


def test_two_pass_launcher_runs_in_order(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    config = generate.resolve(
        generate.build_parser().parse_args(
            [
                "--trace",
                str(trace),
                "--output-dir",
                str(tmp_path / "cache"),
            ]
        )
    )
    observed: list[tuple[str, ...]] = []

    def fake_runner(
        argv: tuple[str, ...], *, check: bool, cwd: Path
    ) -> subprocess.CompletedProcess:
        assert check is True
        assert cwd == generate.REPO_ROOT
        observed.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    generate.run(config, runner=fake_runner)

    contract_path = config.output_dir / generate.GENERATION_MANIFEST
    assert generate.generation_contract(config)["optimal_top_k"] == -1
    assert json.loads(contract_path.read_text(encoding="utf-8")) == (
        generate.generation_contract(config)
    )
    assert len(observed) == 2
    assert "--generate_trace=true" in observed[0]
    assert "--generate_trace=false" in observed[1]


def test_generator_rejects_stale_output_with_different_contract(
    tmp_path: Path,
) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    output = tmp_path / "cache"
    first = generate.resolve(
        generate.build_parser().parse_args(
            ["--trace", str(trace), "--output-dir", str(output)]
        )
    )
    generate.run(
        first,
        runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
    )
    changed = generate.resolve(
        generate.build_parser().parse_args(
            [
                "--trace",
                str(trace),
                "--output-dir",
                str(output),
                "--num-chips",
                "1,2,4",
            ]
        )
    )

    with pytest.raises(generate.GenerationError, match="contract does not match"):
        generate.run(
            changed,
            runner=lambda argv, **kwargs: subprocess.CompletedProcess(argv, 0),
        )


def test_generator_dry_run_does_not_create_output(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    output = tmp_path / "cache"
    config = generate.resolve(
        generate.build_parser().parse_args(
            [
                "--trace",
                str(trace),
                "--output-dir",
                str(output),
                "--dry-run",
            ]
        )
    )

    generate.run(config)

    assert not output.exists()


def test_generator_accepts_explicit_top_k(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")
    config = generate.resolve(
        generate.build_parser().parse_args(["--trace", str(trace), "--top-k", "7"])
    )

    assert config.top_k == 7
    assert "--optimal_top_k=7" in generate.command(config, generate_trace=False)
    assert generate.generation_contract(config)["optimal_top_k"] == 7


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_generator_rejects_invalid_top_k(value: str, tmp_path: Path) -> None:
    trace = _trace(tmp_path / "Azure_sample.csv")

    with pytest.raises(SystemExit):
        generate.build_parser().parse_args(["--trace", str(trace), "--top-k", value])
