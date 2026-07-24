import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_ALLOCATION = REPO_ROOT / "configs" / "fleetsim" / "smoke_llama3_8b_tpuv4.json"


def cli_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("NEUSIM_CONFIGS_DIR", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), env.get("PYTHONPATH", "")])
    )
    return env


def test_fleetsim_cli_validates_from_outside_checkout(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "neusim.run_scripts.fleetsim_main",
            "--validate_only",
            "--model=llama3-8b",
            "--request_pattern=synthetic",
            "--synthetic_num_requests=1",
            f"--static_vpod_allocation={SMOKE_ALLOCATION}",
            f"--output_dir={tmp_path / 'results'}",
            f"--npusim_backend_cache_dir={tmp_path / 'backend-cache'}",
        ],
        cwd=tmp_path,
        env=cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "FleetSim configuration validated" in completed.stderr


def test_fleetsim_cli_rejects_missing_dvfs_slo_file(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "neusim.run_scripts.fleetsim_main",
            "--validate_only",
            "--model=llama3-8b",
            "--request_pattern=synthetic",
            "--synthetic_num_requests=1",
            "--enable_dvfs=true",
            "--enable_dvfs_power_model=true",
            f"--slo_json_path={tmp_path / 'missing-slo.json'}",
            f"--static_vpod_allocation={SMOKE_ALLOCATION}",
            f"--output_dir={tmp_path / 'results'}",
            f"--npusim_backend_cache_dir={tmp_path / 'backend-cache'}",
        ],
        cwd=tmp_path,
        env=cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "SLO JSON not found" in completed.stderr


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
            "NEUSIM_PYTHON": sys.executable,
        }
    )
    completed = subprocess.run(
        ["bash", str(REPO_ROOT / "neusim" / "run_scripts" / "run_fleetsim_smoke.sh")],
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
    assert rows[0]["decode_queuing_delay_per_iteration_ns"] == "0"


def test_only_final_figure5_and_smoke_static_configs_are_packaged() -> None:
    expected = {
        "figure_05_llama3_70b_tpuv5p_p20d8.json",
        "smoke_llama3_8b_tpuv4.json",
    }
    config_dir = REPO_ROOT / "configs" / "fleetsim"
    assert {path.name for path in config_dir.glob("*.json")} == expected
    for config_name in expected:
        config = json.loads((config_dir / config_name).read_text())
        assert set(config) == {"prefill", "decode"}


def test_smoke_launcher_uses_static_config_and_no_search_cache() -> None:
    launcher = REPO_ROOT / "neusim" / "run_scripts" / "run_fleetsim_smoke.sh"
    helper = REPO_ROOT / "neusim" / "run_scripts" / "fleetsim_env.sh"
    launcher_text = launcher.read_text()
    helper_text = helper.read_text()
    assert "${NEUSIM_CONFIGS_DIR}/fleetsim/smoke_llama3_8b_tpuv4.json" in launcher_text
    assert "chip_versions" not in launcher_text
    assert "REQUEST_CACHE" not in helper_text
    assert "request_results_cache" not in helper_text


@pytest.mark.parametrize(
    "removed_flag",
    [
        "--opt_goal",
        "--chip_versions",
        "--allocation_success_rate",
        "--hs_interval_minutes",
        "--vs_interval_minutes",
        "--request_results_cache_dir",
        "--num_pools",
        "--max_chips_per_version",
    ],
)
def test_neuscale_cli_flags_are_removed(tmp_path, removed_flag) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "neusim.run_scripts.fleetsim_main",
            "--validate_only",
            f"--static_vpod_allocation={SMOKE_ALLOCATION}",
            f"{removed_flag}=unused",
        ],
        cwd=tmp_path,
        env=cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Unknown command line flag" in completed.stderr
