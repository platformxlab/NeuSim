"""Backward-compatible facade for component-wise custom DVFS policies."""

from neusim.npusim.backend.dvfs_custom_policy import (  # noqa: F401
    couple_compute_domains,
    get_compute_domain_groups,
    get_dvfs_policy_custom,
    slowdown_freq,
)
from neusim.npusim.frontend.util import compute_component_slack_for_op  # noqa: F401
