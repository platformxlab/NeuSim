import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.fleetsim.NPUClusterManager import (
    NPUClusterManager,
    NPUPodAllocationRequest,
)


class SequenceRandom:
    """Deterministic, pickle-friendly random source with call accounting."""

    def __init__(self, *values: float):
        self.values = values
        self.calls = 0

    def random(self) -> float:
        if self.calls >= len(self.values):
            raise AssertionError("random() called more times than expected")
        value = self.values[self.calls]
        self.calls += 1
        return value


def scheduler_config(
    *,
    npu_types: list[str] | None = None,
    probabilities: list[float] | None = None,
    max_chips: dict[str, int] | None = None,
    prefill_npu_types: list[str] | None = None,
    decode_npu_types: list[str] | None = None,
    chip_config_path: str = "",
):
    return SimpleNamespace(
        npu_types=["4"] if npu_types is None else npu_types,
        satisfaction_probability=[1.0] if probabilities is None else probabilities,
        max_chips_per_version=max_chips,
        prefill_npu_types=prefill_npu_types,
        decode_npu_types=decode_npu_types,
        chip_config_path=chip_config_path,
    )


def simulator(config):
    return SimpleNamespace(
        config=SimpleNamespace(cluster_scheduler_config=config),
    )


def manager(config=None, random_values=(0.0,)) -> NPUClusterManager:
    config = config or scheduler_config()
    chip_configs = {npu_type: object() for npu_type in config.npu_types}
    with patch.object(
        NPUClusterManager,
        "_load_chip_configs",
        return_value=chip_configs,
    ):
        return NPUClusterManager(
            simulator(config),
            random_generator=SequenceRandom(*random_values),
        )


@pytest.mark.parametrize(
    ("num_chips", "pod_shape"),
    [
        (None, None),
        (1, (1, 1, 1)),
    ],
)
def test_request_requires_exactly_one_size(num_chips, pod_shape):
    with pytest.raises(ValueError, match="exactly one"):
        NPUPodAllocationRequest(
            timestamp=0,
            npu_type="4",
            num_chips=num_chips,
            pod_shape=pod_shape,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestamp": -1, "npu_type": "4", "num_chips": 1},
        {"timestamp": 0, "npu_type": "", "num_chips": 1},
        {"timestamp": 0, "npu_type": "4", "num_chips": 0},
        {"timestamp": 0, "npu_type": "4", "num_chips": -1},
        {"timestamp": 0, "npu_type": "4", "pod_shape": ()},
        {"timestamp": 0, "npu_type": "4", "pod_shape": (1, 0, 2)},
    ],
)
def test_request_rejects_invalid_metadata(kwargs):
    with pytest.raises(ValueError):
        NPUPodAllocationRequest(**kwargs)


def test_request_normalizes_shape_and_accepts_configurable_npu_versions():
    request = NPUPodAllocationRequest(
        timestamp=17,
        npu_type="6p",
        pod_shape=(2, 3, 4),
    )

    assert request.timestamp == 17
    assert request.npu_type == "6p"
    assert request.pod_shape == (2, 3, 4)
    assert request.num_chips == 24


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (scheduler_config(npu_types=[]), "must not be empty"),
        (scheduler_config(npu_types=["4", "4"]), "must be unique"),
        (scheduler_config(npu_types=["4", ""]), "non-empty strings"),
        (scheduler_config(prefill_npu_types=[]), "prefill_npu_types must not be empty"),
        (scheduler_config(decode_npu_types=[]), "decode_npu_types must not be empty"),
        (
            scheduler_config(npu_types=["4", "5p"], prefill_npu_types=["4", "4"]),
            "prefill_npu_types must contain unique",
        ),
        (
            scheduler_config(npu_types=["4", "5p"], decode_npu_types=["5p", "5p"]),
            "decode_npu_types must contain unique",
        ),
        (
            scheduler_config(npu_types=["4", "5p"], prefill_npu_types=["6e"]),
            "prefill_npu_types contains types not present in npu_types",
        ),
        (
            scheduler_config(npu_types=["4", "5p"], decode_npu_types=["6e"]),
            "decode_npu_types contains types not present in npu_types",
        ),
        (
            scheduler_config(
                npu_types=["4", "5p"],
                probabilities=[0.1, 0.2, 0.3],
            ),
            "one shared value or one value per NPU type",
        ),
        (scheduler_config(probabilities=[-0.1]), r"must be in \[0, 1\]"),
        (scheduler_config(probabilities=[1.1]), r"must be in \[0, 1\]"),
        (
            scheduler_config(max_chips={"5p": 8}),
            "types not present in npu_types",
        ),
        (
            scheduler_config(max_chips={"4": -1}),
            "must be non-negative",
        ),
    ],
)
def test_scheduler_configuration_is_validated_before_loading_chips(config, message):
    with (
        patch.object(NPUClusterManager, "_load_chip_configs") as load_configs,
        pytest.raises(ValueError, match=message),
    ):
        NPUClusterManager(simulator(config))

    load_configs.assert_not_called()


def test_loads_each_configured_chip_version_from_config_path(tmp_path):
    (tmp_path / "tpuv4.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "tpuv6p.json").write_text(json.dumps({}), encoding="utf-8")
    config = scheduler_config(
        npu_types=["4", "6p"],
        probabilities=[1.0],
        chip_config_path=str(tmp_path),
    )

    cluster_manager = NPUClusterManager(
        simulator(config),
        random_generator=SequenceRandom(0.0),
    )

    assert set(cluster_manager.chip_configs) == {"4", "6p"}
    assert all(
        isinstance(chip_config, ChipConfig)
        for chip_config in cluster_manager.chip_configs.values()
    )


def test_accepts_distinct_phase_specific_npu_type_subsets():
    config = scheduler_config(
        npu_types=["4", "5p", "6e"],
        prefill_npu_types=["5p", "6e"],
        decode_npu_types=["4"],
    )

    cluster_manager = manager(config)

    assert cluster_manager.scheduler_config.prefill_npu_types == ["5p", "6e"]
    assert cluster_manager.scheduler_config.decode_npu_types == ["4"]


def test_probability_mapping_checks_membership_even_for_shared_probability():
    shared = manager(
        scheduler_config(npu_types=["4", "5p"], probabilities=[0.25]),
    )
    assert shared.get_allocation_satisfaction_probability("4") == 0.25
    assert shared.get_allocation_satisfaction_probability("5p") == 0.25
    with pytest.raises(ValueError, match="not supported"):
        shared.get_allocation_satisfaction_probability("6p")

    per_type = manager(
        scheduler_config(npu_types=["4", "5p"], probabilities=[0.25, 0.75]),
    )
    assert per_type.get_allocation_satisfaction_probability("4") == 0.25
    assert per_type.get_allocation_satisfaction_probability("5p") == 0.75


def test_success_is_cached_through_interval_and_refreshed_afterward():
    cluster_manager = manager(
        scheduler_config(probabilities=[0.5]),
        random_values=(0.25, 0.75),
    )
    interval = cluster_manager.allocation_update_interval_ns

    assert cluster_manager.allocate(NPUPodAllocationRequest(100, "4", num_chips=1))
    assert cluster_manager.last_allocation_request_timestamp == 100
    assert cluster_manager.allocate(
        NPUPodAllocationRequest(100 + interval, "4", num_chips=1)
    )
    assert cluster_manager.random_generator.calls == 1

    assert not cluster_manager.allocate(
        NPUPodAllocationRequest(101 + interval, "4", num_chips=1)
    )
    assert cluster_manager.random_generator.calls == 2
    assert cluster_manager.last_allocation_request_timestamp == 101 + interval


def test_failure_is_cached_through_interval_and_refreshed_afterward():
    cluster_manager = manager(
        scheduler_config(probabilities=[0.5]),
        random_values=(0.75, 0.25),
    )
    interval = cluster_manager.allocation_update_interval_ns

    assert not cluster_manager.allocate(NPUPodAllocationRequest(20, "4", num_chips=1))
    assert not cluster_manager.allocate(NPUPodAllocationRequest(21, "4", num_chips=1))
    assert cluster_manager.random_generator.calls == 1

    assert cluster_manager.allocate(
        NPUPodAllocationRequest(21 + interval, "4", num_chips=1)
    )
    assert cluster_manager.random_generator.calls == 2


def test_probability_cache_is_independent_per_npu_type():
    cluster_manager = manager(
        scheduler_config(
            npu_types=["4", "5p"],
            probabilities=[1.0, 0.0],
        ),
        random_values=(0.5, 0.5),
    )

    assert cluster_manager.allocate(NPUPodAllocationRequest(10, "4", num_chips=1))
    assert not cluster_manager.allocate(NPUPodAllocationRequest(11, "5p", num_chips=1))
    assert cluster_manager.allocate(NPUPodAllocationRequest(12, "4", num_chips=1))
    assert cluster_manager.random_generator.calls == 2


def test_capacity_is_checked_before_cached_probability():
    cluster_manager = manager(
        scheduler_config(probabilities=[1.0], max_chips={"4": 4}),
        random_values=(0.0,),
    )
    cluster_manager.track_allocate("4", 4)

    assert not cluster_manager.allocate(NPUPodAllocationRequest(0, "4", num_chips=1))
    assert cluster_manager.random_generator.calls == 0

    cluster_manager.track_deallocate("4", 2)
    assert cluster_manager.allocate(NPUPodAllocationRequest(1, "4", num_chips=2))
    assert cluster_manager.random_generator.calls == 1


def test_capacity_accounting_validates_inputs_and_returns_defensive_copy():
    cluster_manager = manager()

    cluster_manager.track_allocate("4", 3)
    snapshot = cluster_manager.get_allocated_chips()
    snapshot["4"] = 999
    assert cluster_manager.get_allocated_chips() == {"4": 3}

    cluster_manager.track_deallocate("4", 2)
    assert cluster_manager.get_allocated_chips() == {"4": 1}

    with pytest.raises(ValueError, match="not supported"):
        cluster_manager.track_allocate("5p", 1)
    with pytest.raises(ValueError, match="positive"):
        cluster_manager.track_allocate("4", 0)
    with pytest.raises(ValueError, match="only 1"):
        cluster_manager.track_deallocate("4", 2)


def test_checkpoint_round_trip_preserves_cache_rng_and_capacity_state(tmp_path):
    config = scheduler_config(
        npu_types=["4", "5p"],
        probabilities=[0.5, 0.5],
        max_chips={"4": 8, "5p": 8},
    )
    cluster_manager = manager(config, random_values=(0.25, 0.75, 0.1))
    cluster_manager.allocation_update_interval_ns = 123

    assert cluster_manager.allocate(NPUPodAllocationRequest(10, "4", num_chips=1))
    assert not cluster_manager.allocate(NPUPodAllocationRequest(20, "5p", num_chips=1))
    cluster_manager.track_allocate("4", 2)
    checkpoint = tmp_path / "cluster.pkl.gz"
    cluster_manager.save_to_checkpoint(checkpoint)

    restored = manager(config, random_values=(0.99,))
    restored.load_from_checkpoint(checkpoint)

    assert restored.get_allocated_chips() == {"4": 2, "5p": 0}
    assert restored.allocation_update_interval_ns == 123
    assert restored.last_allocation_success is False
    assert restored.last_allocation_request_timestamp == 20
    assert restored.last_allocation_npu_type == "5p"
    assert restored._allocation_outcomes == {
        "4": (True, 10),
        "5p": (False, 20),
    }

    assert not restored.allocate(NPUPodAllocationRequest(100, "5p", num_chips=1))
    assert restored.random_generator.calls == 2
    assert restored.allocate(NPUPodAllocationRequest(134, "4", num_chips=1))
    assert restored.random_generator.calls == 3
