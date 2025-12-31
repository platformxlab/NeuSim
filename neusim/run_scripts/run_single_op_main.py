### Single-op simulation script. Helpful for testing, debugging, and experimenting.


import csv
import json
import os
from typing import Any
from absl import app, flags, logging
import ray

from neusim.npusim.frontend.Operator import EinsumOperator, EinsumStatistics, Operator, to_csv_dict
from neusim.configs.chips.ChipConfig import ChipConfig
import neusim.npusim.frontend.llm_ops_lib as ops_lib
import neusim.npusim.frontend.op_analysis_lib as analysis_lib


__CHIP_CONFIG = flags.DEFINE_string(
    "chip_config",
    "../configs/chips/tpuv4.json",
    "Path to the NPU chip configuration file.",
)
CHIP_CONFIG = None
__OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "../results/single_op",
    "Path to the output directory for simulation results.",
)
OUTPUT_PATH = None
__DEBUG = flags.DEFINE_boolean(
    "debug",
    False,
    "If true, run in debug mode without ray.",
)


def get_sim_config(cfg_file_name: str | None = None) -> ChipConfig:
    global CHIP_CONFIG
    if cfg_file_name is None:
        assert CHIP_CONFIG
        cfg_file_name = CHIP_CONFIG
    with open(cfg_file_name, "r") as f:
        config = json.load(f)
    return ChipConfig.model_validate(config)


def generate_matmul_op(
    M: int, N: int, K: int
):
    return ops_lib.create_einsum_op(
        [M, K],
        [K, N],
        "MK;KN->MN",
        name=f"matmul_{M}_{N}_{K}",
    )


def generate_matmul_ops(
    M_min: int = 1,
    M_max: int = 53248,
    N_min: int = 1,
    N_max: int = 53248,
    K_min: int = 1,
    K_max: int = 53248,
    step: int = 128,
):
    for M in range(M_min, M_max + 1, step):
        for N in range(N_min, N_max + 1, step):
            for K in range(K_min, K_max + 1, step):
                op = generate_matmul_op(M, N, K)
                yield op


def run_sim_for_op(op: Operator, cfg: ChipConfig) -> Operator:
    ops = analysis_lib.fill_operators_execution_info(
        [op], cfg
    )
    return ops[0]


def __run_sim_for_op(row: dict[str, Any]):
    return { "item": to_csv_dict(run_sim_for_op(*row["item"])) }


def init_cmd_flags():
    global CHIP_CONFIG
    global OUTPUT_PATH

    CHIP_CONFIG = __CHIP_CONFIG.value
    OUTPUT_PATH = __OUTPUT_PATH.value

    os.makedirs(OUTPUT_PATH, exist_ok=True)


def main(argv):
    del argv  # Unused.

    init_cmd_flags()

    # ops_small = list(
    #     generate_matmul_ops(
    #         1, 1024, 1, 1024, 1, 1024, 128
    #     )
    # )
    # ops_medium = list(
    #     generate_matmul_ops(
    #         1024, 16384, 1024, 16384, 1024, 16384, 2048
    #     )
    # )
    # ops_large = list(
    #     generate_matmul_ops(
    #         16384, 53248, 16384, 53248, 16384, 53248, 4096
    #     )
    # )
    # ops = ops_small + ops_medium + ops_large
    # logging.info(f"Generated {len(ops_small)} small ops, {len(ops_medium)} medium ops, and {len(ops_large)} large ops. Total: {len(ops)} ops.")

    if __DEBUG.value:
        from IPython import embed; embed()
        exit(0)


    ops = [generate_matmul_op(8, 14336, 4096)]

    cfg = get_sim_config()

    input_ds = ray.data.from_items([
        (op, cfg) for op in ops
    ])
    result_ds = input_ds.map(__run_sim_for_op)
    op_dicts = [item["item"] for item in result_ds.take_all()]

    assert OUTPUT_PATH
    with open(os.path.join(OUTPUT_PATH, "matmul.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=op_dicts[0].keys())
        writer.writeheader()
        writer.writerows(op_dicts)


if __name__ == "__main__":
    app.run(main)
