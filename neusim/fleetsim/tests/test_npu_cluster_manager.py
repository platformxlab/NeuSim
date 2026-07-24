from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.fleetsim.NPUClusterManager import NPUClusterManager


def simulator(
    *, npu_types: tuple[str, ...] = ("4",), chip_config_path: str = ""
):
    fleet_config = SimpleNamespace(
        npu_types=npu_types,
        cluster_scheduler_config=SimpleNamespace(chip_config_path=chip_config_path),
    )
    return SimpleNamespace(config=fleet_config)


@pytest.mark.parametrize(
    ("npu_types", "message"),
    [
        ((), "must reference"),
        (("4", "4"), "must be unique"),
        (("4", ""), "non-empty strings"),
    ],
)
def test_static_inventory_is_validated_before_loading_chips(
    npu_types, message
) -> None:
    with (
        patch.object(NPUClusterManager, "_load_chip_configs") as load_configs,
        pytest.raises(ValueError, match=message),
    ):
        NPUClusterManager(simulator(npu_types=npu_types))
    load_configs.assert_not_called()


def test_loads_only_chip_versions_named_by_static_allocation(tmp_path) -> None:
    for npu_type in ("4", "6p", "7e"):
        (tmp_path / f"tpuv{npu_type}.json").write_text("{}", encoding="utf-8")

    cluster_manager = NPUClusterManager(
        simulator(npu_types=("4", "6p"), chip_config_path=str(tmp_path))
    )

    assert set(cluster_manager.chip_configs) == {"4", "6p"}
    assert all(
        isinstance(chip_config, ChipConfig)
        for chip_config in cluster_manager.chip_configs.values()
    )
