import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from neusim.run_scripts import fleetsim_main

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("model", ["llama3-70b", "llama-qwen3-32b"])
def test_fleetsim_cli_validates_from_outside_checkout(tmp_path, model) -> None:
    env = os.environ.copy()
    env.pop("NEUSIM_CONFIGS_DIR", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")])
    )
    request_cache = (
        REPO_ROOT / "neusim" / "fleetsim" / "tests" / "data" / "request_lookup_cache"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "neusim.run_scripts.fleetsim_main",
            "--validate_only",
            f"--model={model}",
            "--system=Base",
            "--trace=Azure-test",
            "--output_prediction_accuracy=0.6",
            "--output_prediction_seed=17",
            f"--request_results_cache_dir={request_cache}",
            f"--output_dir={tmp_path / 'results'}",
            f"--npusim_backend_cache_dir={tmp_path / 'backend-cache'}",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FleetSim configuration validated" in completed.stderr
    assert "output_prediction_accuracy=0.6" in completed.stderr
    assert "output_prediction_seed=17" in completed.stderr


@pytest.mark.parametrize(
    "modules",
    [
        (
            "neusim.run_scripts.fleetsim_main",
            "neusim.run_scripts.run_sim_find_optimal",
        ),
        (
            "neusim.run_scripts.run_sim_find_optimal",
            "neusim.run_scripts.fleetsim_main",
        ),
    ],
)
def test_fleetsim_and_stage_a_cli_modules_can_coexist(tmp_path, modules) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")])
    )
    completed = subprocess.run(
        [sys.executable, "-c", f"import {modules[0]}; import {modules[1]}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_fleetsim_shell_defaults_write_under_current_directory(tmp_path) -> None:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("NEUSIM_")
    }
    helper = REPO_ROOT / "neusim" / "run_scripts" / "fleetsim_env.sh"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n" "$NEUSIM_RESULTS_DIR" "$NEUSIM_BACKEND_CACHE_DIR"',
            "fleetsim-default-test",
            str(helper),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(tmp_path / "results" / "fleetsim"),
        str(tmp_path / "results" / "fleetsim" / ".cache" / "npusim_backend"),
    ]


def test_fleetsim_shell_defaults_use_repository_configs(tmp_path) -> None:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("NEUSIM_")
    }
    helper = REPO_ROOT / "neusim" / "run_scripts" / "fleetsim_env.sh"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$NEUSIM_CONFIGS_DIR"',
            "fleetsim-config-default-test",
            str(helper),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(REPO_ROOT / "configs")


def test_fleetsim_smoke_script_runs_exact_decode_work(tmp_path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "NEUSIM_REPO_ROOT": str(REPO_ROOT),
            "NEUSIM_RESULTS_DIR": str(tmp_path / "results"),
            "NEUSIM_BACKEND_CACHE_DIR": str(tmp_path / "backend-cache"),
            "NEUSIM_REQUEST_CACHE_DIR": str(
                REPO_ROOT
                / "neusim"
                / "fleetsim"
                / "tests"
                / "data"
                / "request_lookup_cache"
            ),
            "NEUSIM_PYTHON": sys.executable,
        }
    )

    completed = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "neusim" / "run_scripts" / "run_fleetsim_smoke.sh"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output_dir = tmp_path / "results" / "smoke"
    with (output_dir / "request_trace.csv").open(newline="") as trace_file:
        rows = list(csv.DictReader(trace_file))
    assert len(rows) == 1
    assert rows[0]["output_seqlen"] == "4"
    assert len(rows[0]["config_decode_batch_sizes"].split("/")) == 3
    assert rows[0]["decode_queuing_delay_per_iteration_ns"] == "0"

    checkpoints = list(output_dir.glob("checkpoint_*"))
    assert len(checkpoints) == 1
    assert int(checkpoints[0].name.removeprefix("checkpoint_")) < int(60 * 1e9)


def test_static_vpod_configs_live_under_repository_configs() -> None:
    config_names = {
        "static_mixed_llama3_70b_4pv4tp8_6dv6etp8.json",
        "static_mixed_llama3_8b_4pv4tp8_6dv6etp8.json",
        "static_tpuv4_llama3_8b_tp4.json",
        "static_tpuv4_llama3_8b_tp4_2p6d.json",
        "static_tpuv4_llama3_8b_tp4_6p2d.json",
        "static_tpuv4_llama3_8b_tp4_8p8d.json",
    }
    config_dir = REPO_ROOT / "configs" / "fleetsim"

    assert {path.name for path in config_dir.glob("*.json")} == config_names
    old_config_dir = REPO_ROOT / "neusim" / "run_scripts" / "fleetsim_configs"
    assert not list(old_config_dir.glob("*.json"))
    for config_name in config_names:
        config = json.loads((config_dir / config_name).read_text())
        assert set(config) == {"prefill", "decode"}


def test_static_vpod_smoke_launcher_uses_config_root() -> None:
    config_name = "static_tpuv4_llama3_8b_tp4.json"
    launcher = REPO_ROOT / "neusim" / "run_scripts" / "run_fleetsim_smoke.sh"
    contents = launcher.read_text()

    assert f"${{NEUSIM_CONFIGS_DIR}}/fleetsim/{config_name}" in contents
    assert "fleetsim_configs/" not in contents


@pytest.mark.parametrize(
    ("use_mmap", "expected_kwargs"),
    [(False, {}), (True, {"mmap_mode": "r"})],
)
def test_ideal_backend_uses_configured_disk_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_mmap: bool,
    expected_kwargs: dict[str, str],
) -> None:
    cache_dir = tmp_path / "ideal-backend-cache"
    configured: list[tuple[Path, dict[str, str]]] = []
    profiling: list[bool] = []
    config = SimpleNamespace(
        npusim_backend_cache_dir=str(cache_dir),
        npusim_backend_cache_use_mmap=use_mmap,
        enable_profile=True,
    )

    monkeypatch.setattr(
        fleetsim_main.npusim_backend,
        "set_npusim_backend_cache_dir",
        lambda path, **kwargs: configured.append((Path(path), kwargs)),
    )
    monkeypatch.setattr(
        fleetsim_main.npusim_backend,
        "set_enable_profile",
        profiling.append,
    )

    fleetsim_main._initialize_ideal_backend(config)

    assert cache_dir.is_dir()
    assert configured == [(cache_dir, expected_kwargs)]
    assert profiling == [True]
