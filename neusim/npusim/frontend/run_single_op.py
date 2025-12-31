### Helper library for running a single operator simulation

from neusim.npusim.frontend.Operator import Operator
from neusim.npusim.frontend import op_analysis_lib as analysis_lib
from neusim.configs.chips.ChipConfig import ChipConfig


def run_sim_single_op(op: Operator, cfg: ChipConfig) -> Operator:
    ops = analysis_lib.fill_operators_execution_info(
        [op], cfg
    )
    return ops[0]
