from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from neusim.run_scripts import run_micro26ae_sample as runner


def _config(tmp_path: Path, *arguments: str) -> runner.Config:
    args = runner.build_parser().parse_args(
        [
            "--trace-file",
            str(tmp_path / "trace.csv"),
            "--results-dir",
            str(tmp_path / "results"),
            "--request-cache-dir",
            str(tmp_path / "request-cache"),
            "--backend-cache-dir",
            str(tmp_path / "backend-cache"),
            "--configs-dir",
            str(tmp_path / "configs"),
            *arguments,
        ]
    )
    return runner.resolve(args)


def test_defaults_are_the_three_hour_figures18_and19_matrix(tmp_path: Path) -> None:
    config = _config(tmp_path)
    experiments = runner.experiment_matrix(config)

    assert config.model == "deepseekv3-671b"
    assert config.trace_name == "Azure-Code"
    assert config.hours == 3
    assert config.chip_versions == ("5p", "6e")
    assert len(experiments) == 6
    assert {(item.system, item.goal) for item in experiments} == {
        (system, goal)
        for system in ("Base-Max", "NeuScale", "Ideal")
        for goal in ("energy", "monetary")
    }
    assert {
        item.prediction_accuracy for item in experiments if item.system == "NeuScale"
    } == {0.6}
    assert all(
        item.prediction_accuracy is None
        for item in experiments
        if item.system != "NeuScale"
    )


def test_command_passes_explicit_paths_and_sample_controls(tmp_path: Path) -> None:
    config = _config(tmp_path, "--max-requests", "100", "--n-cpu", "7")
    experiment = next(
        item
        for item in runner.experiment_matrix(config)
        if item.system == "NeuScale" and item.goal == "monetary"
    )

    argv = runner.command(config, experiment)

    assert f"--configs_path={config.configs_dir}" in argv
    assert f"--request_results_cache_dir={config.cache_dir}" in argv
    assert f"--npusim_backend_cache_dir={config.backend_cache_dir / 'monetary'}" in argv
    assert f"--trace_file={config.trace_file}" in argv
    assert f"--traces_dir={config.trace_file.parent}" in argv
    assert "--chip_versions=5p,6e" in argv
    assert "--max_timestamp_hours=3" in argv
    assert "--max_num_requests=100" in argv
    assert "--output_prediction_accuracy=0.6" in argv
    assert "--n_cpu=7" in argv
    assert not any("plot" in argument for argument in argv)


def test_sensitivity_sweeps_expand_only_relevant_systems(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        "--systems",
        "Base-Max,NeuScale,Ideal,MultiPool",
        "--goals",
        "energy",
        "--prediction-accuracies",
        "0.5,0.6,0.8,0.9,1",
        "--num-pools",
        "2,3,4",
        "--max-chips-per-version",
        "unlimited",
        "--max-chips-per-version",
        "5p=256,6e=512",
    )
    experiments = runner.experiment_matrix(config)

    # Per cap: Base-Max (1), Ideal (1), NeuScale (5), MultiPool (3).
    assert len(experiments) == 20
    assert sum(item.system == "Base-Max" for item in experiments) == 2
    assert sum(item.system == "Ideal" for item in experiments) == 2
    assert sum(item.system == "NeuScale" for item in experiments) == 10
    assert sum(item.system == "MultiPool" for item in experiments) == 6
    assert len({runner.run_dir(config, item) for item in experiments}) == 20


def test_hardware_customization_reaches_fleetsim(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        "--systems",
        "MultiPool",
        "--goals",
        "energy",
        "--chip-versions",
        "5p,6e",
        "--prefill-chip-versions",
        "6e",
        "--decode-chip-versions",
        "5p",
        "--num-pools",
        "4",
        "--max-chips-per-version",
        "5p=512,6e=1024",
    )
    experiment = runner.experiment_matrix(config)[0]
    argv = runner.command(config, experiment)

    assert "--prefill_chip_versions=6e" in argv
    assert "--decode_chip_versions=5p" in argv
    assert "--num_pools=4" in argv
    assert "--max_chips_per_version=5p=512,6e=1024" in argv
    assert not any(
        argument.startswith("--output_prediction_accuracy=") for argument in argv
    )


def test_list_prints_commands_without_requiring_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = runner.main(
        [
            "--list",
            "--trace-file",
            str(tmp_path / "missing.csv"),
            "--request-cache-dir",
            str(tmp_path / "missing-cache"),
            "--systems",
            "NeuScale",
            "--goals",
            "energy",
        ]
    )

    output = capsys.readouterr().out
    assert status == 0
    assert "[1/1]" in output
    assert "neusim.run_scripts.fleetsim_main" in output
    assert "--output_prediction_accuracy=0.6" in output


def test_expected_count_and_resume_accept_large_trace_fields(tmp_path: Path) -> None:
    config = _config(tmp_path, "--hours", "1", "--max-requests", "2")
    config.trace_file.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2026-01-01T00:00:00+00:00,32,4\n"
        "2026-01-01T00:30:00+00:00,64,8\n"
        "2026-01-01T02:00:00+00:00,128,16\n",
        encoding="utf-8",
    )
    assert runner.expected_requests(config) == 2

    experiment = runner.experiment_matrix(config)[0]
    directory = runner.run_dir(config, experiment)
    directory.mkdir(parents=True)
    (directory / "stats.json").write_text(
        json.dumps({"total_requests": 2}), encoding="utf-8"
    )
    with (directory / "request_trace.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("request_id", "config_history"))
        writer.writerow(("0", "x" * 200_000))
        writer.writerow(("1", "y"))

    contract = runner.run_contract(config, experiment, 2)
    runner.write_run_contract(directory, contract)
    assert runner.inspect_complete(directory, 2, contract)
    assert runner._run_one(config, experiment, 2) == "reused"

    config.trace_file.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n"
        "2026-01-01T00:00:00+00:00,33,4\n"
        "2026-01-01T00:30:00+00:00,65,8\n",
        encoding="utf-8",
    )
    changed_contract = runner.run_contract(config, experiment, 2)
    assert changed_contract["trace_sha256"] != contract["trace_sha256"]
    assert not runner.inspect_complete(directory, 2, changed_contract)


def test_manifest_contains_raw_run_contract_without_plotting(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "--systems", "NeuScale", "--goals", "energy")
    config.trace_file.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n" "2026-01-01T00:00:00+00:00,32,4\n",
        encoding="utf-8",
    )
    experiment = runner.experiment_matrix(config)[0]
    manifest = runner.write_manifest(
        config, (experiment,), expected=100, states={experiment: "reused"}
    )
    document = json.loads(manifest.read_text(encoding="utf-8"))

    assert document["expected_requests_per_setting"] == 100
    assert document["experiments"][0]["status"] == "reused"
    assert document["experiments"][0]["command"] == list(
        runner.command(config, experiment)
    )
    assert "plot" not in json.dumps(document).lower()
    contract = runner.run_contract(config, experiment, 100)
    assert contract["command"] == document["experiments"][0]["command"]
    assert "trace_sha256" in contract
    assert "configs_sha256" in contract


@pytest.mark.parametrize(
    "arguments,message",
    [
        (("--hours", "0"), "--hours"),
        (("--max-requests", "0"), "--max-requests"),
        (("--prediction-accuracy", "1.1"), "prediction accuracy"),
        (("--systems", "Unknown"), "unsupported systems"),
        (("--goals", "latency"), "unsupported goals"),
        (
            ("--max-chips-per-version", "5p=nope"),
            "--max-chips-per-version",
        ),
    ],
)
def test_invalid_customization_is_rejected(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    with pytest.raises(runner.LauncherError, match=message):
        _config(tmp_path, *arguments)
