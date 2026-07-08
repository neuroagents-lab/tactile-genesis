"""System-identification extensions."""

from eden.extensions.sysid.base import SYSID_REGISTRY, SystemIdentificationBase
from eden.extensions.sysid.deployment_recorder import DeploymentRecorder
from eden.extensions.sysid.excitation import (
    ChirpExcitation,
    Excitation,
    PRBSExcitation,
    PlaybackExcitation,
)
from eden.extensions.sysid.modifier import (
    DOF_PROPERTY_GETTERS,
    DOF_PROPERTY_SETTERS,
    apply_candidates,
    apply_parameters,
    make_parameter_from_default,
)
from eden.extensions.sysid.optimizers import CMAES, SciPyLeastSquares
from eden.extensions.sysid.parameter import Parameter, ParameterSet
from eden.extensions.sysid.residual import multi_trajectory_residual, signal_residual
from eden.extensions.sysid.rollout import (
    batched_candidate_rollout,
    replay_rollout,
    single_candidate_rollout,
)
from eden.extensions.sysid.trajectory import Trajectory

__all__ = [
    "SYSID_REGISTRY",
    "SystemIdentificationBase",
    "Parameter",
    "ParameterSet",
    "Trajectory",
    "SciPyLeastSquares",
    "CMAES",
    "DeploymentRecorder",
    "Excitation",
    "ChirpExcitation",
    "PRBSExcitation",
    "PlaybackExcitation",
    "make_parameter_from_default",
    "apply_parameters",
    "apply_candidates",
    "replay_rollout",
    "batched_candidate_rollout",
    "single_candidate_rollout",
    "signal_residual",
    "multi_trajectory_residual",
    "DOF_PROPERTY_SETTERS",
    "DOF_PROPERTY_GETTERS",
]
