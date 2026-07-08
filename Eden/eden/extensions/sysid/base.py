"""Base class and fit-result type for system identification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.residual import multi_trajectory_residual
from eden.extensions.sysid.rollout import single_candidate_rollout
from eden.extensions.sysid.trajectory import Trajectory
from eden.options.extensions.sysid import SystemIdentificationOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


SYSID_REGISTRY = Registry("SYSID")


@dataclass
class FitResult:
    """Structured diagnostics returned by :meth:`SystemIdentificationBase.fit`.

    Attributes
    ----------
    cost: float
        Final 0.5 * ||residual||^2.
    nfev: int
        Number of residual evaluations performed.
    x: np.ndarray
        Flat optimised parameter vector (matches ``ParameterSet.as_vector``).
    jacobian: np.ndarray | None
        Jacobian at the optimum, when the backend provides one.
    message: str
        Backend-specific termination message.
    history: np.ndarray | None
        Per-iteration cost history, when available.
    extras: dict[str, Any]
        Backend-specific additional diagnostics.
    """

    cost: float
    nfev: int
    x: np.ndarray
    jacobian: np.ndarray | None = None
    message: str = ""
    history: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SystemIdentificationBase(ConfigurableMixin[SystemIdentificationOptions], ABC):
    """Base class for system-identification backends.

    Subclasses override :meth:`fit` to implement the optimisation loop.
    The base class provides :meth:`evaluate` for one-shot residual evaluation
    on the current parameter vector.

    Parameters
    ----------
    entity_name: str
        Name of the entity whose parameters are being identified.
    signals: Sequence[SignalName]
        Measurement signals included in the residual.
    signal_weights: dict[str, float]
        Per-signal weights (missing signals default to 1.0).
    normalize: bool
        If True, normalise each signal block by its measured RMS.
    max_iters: int
        Maximum optimiser iterations.
    verbose: bool
        Print per-iteration status.

    Notes
    -----
    The environment passed in should be a **dedicated sysid env**: a
    standalone ``RLEnvBase`` built with ``env_options.num_envs`` equal to
    the intended batched-candidate population (or 1 for serial backends).
    The base class avoids ``RLEnvBase.step`` entirely so termination,
    reward, reset, command, event, and recorder managers do not fire
    during replay.
    """

    entity_name: str = "robot"
    signals: Sequence[str] = ("dofs_pos", "dofs_vel", "dofs_torque")
    signal_weights: dict[str, float] = {}
    normalize: bool = True
    max_iters: int = 200
    verbose: bool = True

    def __init__(self, env: "EnvBase", options: SystemIdentificationOptions) -> None:
        super().__init__(options)
        self._env = env
        if self.entity_name not in env.entities:
            raise ValueError(f"Entity '{self.entity_name}' not in scene.")

    def evaluate(
        self,
        params: ParameterSet,
        trajectories: Sequence[Trajectory],
    ) -> np.ndarray:
        """Rollout once per trajectory and return the flattened residual."""
        preds = [
            single_candidate_rollout(
                self._env,
                params,
                traj,
                entity_name=self.entity_name,
                signals=self.signals,
            )
            for traj in trajectories
        ]
        return multi_trajectory_residual(
            preds,
            trajectories,
            signals=self.signals,
            weights=self.signal_weights,
            normalize=self.normalize,
        )

    @abstractmethod
    def fit(
        self,
        params: ParameterSet,
        trajectories: Sequence[Trajectory],
    ) -> tuple[ParameterSet, FitResult]:
        """Run the optimisation loop and return the best ParameterSet + FitResult."""
        raise NotImplementedError
