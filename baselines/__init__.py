"""Gradient-based unlearning baselines: GA, GD, NPO, SimNPO, RMU."""

from baselines.grad_ascent import GradAscent
from baselines.grad_diff import GradDiff
from baselines.npo import NPO
from baselines.simnpo import SimNPO
from baselines.rmu import RMU
from baselines.data import ForgetRetainDataset, unlearn_collator

TRAINER_REGISTRY = {
    "GradAscent": GradAscent,
    "GradDiff": GradDiff,
    "NPO": NPO,
    "SimNPO": SimNPO,
    "RMU": RMU,
}

__all__ = [
    "GradAscent", "GradDiff", "NPO", "SimNPO", "RMU",
    "ForgetRetainDataset", "unlearn_collator", "TRAINER_REGISTRY",
]
