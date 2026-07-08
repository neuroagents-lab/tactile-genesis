"""Base class and registry for real-robot deployment backends."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from abc import ABC, abstractmethod
import threading

import numpy as np
import torch

import eden as en
from eden.options.entities import RobotOptions
from eden.options.extensions.deployment import DeploymentOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry
from eden.utils.string import resolve_matching_names_values
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.extensions.deployment.utils.state import RobotCommand, RobotState, RobotStateEntity


if TYPE_CHECKING:
    from eden.envs.base import RLEnvBase
    from eden.utils.configs import EdenRLConfig


DEPLOYMENT_REGISTRY = Registry("DEPLOYMENT")


# Robot → deployment binding. Populated at import time by each deployment
# module (e.g. ``robotera_xhand.py`` declares ``bind_deployment(XHand1_R, ...)``).
# ``eden deploy --task <name>`` resolves the deployment class from the robot
# bound to the task's scene entity via :func:`resolve_deployment_for`.
_ROBOT_DEPLOYMENT_BINDINGS: dict[type, str] = {}


def bind_deployment(robot_cls: type, deployment_name: str) -> None:
    """Register ``robot_cls`` as driven by the named deployment in :data:`DEPLOYMENT_REGISTRY`.

    Resolution walks the robot's MRO, so subclasses inherit the binding
    unless they register their own override. Call this once per robot class
    in the deployment module that supports it; collisions are not detected
    (the last binding wins).
    """
    _ROBOT_DEPLOYMENT_BINDINGS[robot_cls] = deployment_name


def resolve_deployment_for(robot_options: RobotOptions) -> type[DeploymentBase]:
    """Return the deployment class registered for ``type(robot_options)``.

    Walks the MRO to honor inheritance (a subclass with no binding falls
    back to its parent's). Raises ``KeyError`` if no ancestor is bound.
    """
    cls = type(robot_options)
    for ancestor in cls.__mro__:
        if ancestor in _ROBOT_DEPLOYMENT_BINDINGS:
            name = _ROBOT_DEPLOYMENT_BINDINGS[ancestor]
            return DEPLOYMENT_REGISTRY.get(name)
    available = sorted({k.__name__ for k in _ROBOT_DEPLOYMENT_BINDINGS}) or ["<none>"]
    raise KeyError(
        f"No deployment registered for robot {cls.__name__}. "
        f"Available bindings: {available}. Add one with "
        f"``bind_deployment({cls.__name__}, '<deployment_name>')``."
    )


class DeploymentBase(ConfigurableMixin[DeploymentOptions], ABC):
    """
    Base class for real-robot deployment backends.

    Parameters
    ----------
    entity_name: str
        The name of the entity to deploy
    control_dt: float | None
        The control time step
    control_freq: float | None
        The control frequency
    decimation: int | None
        The decimation factor
    sync: bool
        Whether to synchronize the control loop
    connect_timeout_s: float
        The timeout for connecting to the robot
    deploy_kp_kd_scale: float
        The scale of the Kp and Kd for the deployment.
        Recommend to try from 0.2 (20%), 0.5 (50%), 0.8 (80%), 1.0 (100%)
    auto_yaw_align: bool
        If True (default), capture the robot's IMU yaw at the end of
        :meth:`reset` and install it on the state wrapper so subsequent
        observations are computed in the yaw-aligned world frame the policy
        was trained in. Disable when an external estimator (e.g. mocap)
        already provides a yaw-consistent base orientation.
    """

    entity_name: str = "robot"
    control_dt: float | None = None
    control_freq: float | None = None
    decimation: int | None = None
    sync: bool = True
    connect_timeout_s: float = 5.0
    deploy_kp_kd_scale: float = 1.0
    auto_yaw_align: bool = True

    def __init__(self, env: RLEnvBase, options: DeploymentOptions) -> None:
        super().__init__(options)
        self._env = env
        self.cfg = env.config
        self.robot_options = self._resolve_robot_options(self.cfg, self.entity_name)
        self.dofs_name = list(self.robot_options.dofs_name)
        self.num_dofs = len(self.dofs_name)

        self.control_dt = self._resolve_control_dt()
        if self.control_freq is None and self.control_dt > 0.0:
            self.control_freq = 1.0 / self.control_dt

        _, _, resolved = resolve_matching_names_values(
            self.robot_options.default_dofs_pos, self.dofs_name, preserve_order=True
        )
        self.default_dof_pos = np.asarray(resolved, dtype=np.float64)
        _, _, resolved = resolve_matching_names_values(
            self.robot_options.default_dofs_kp, self.dofs_name, preserve_order=True
        )
        self.default_dof_kp = np.asarray(resolved, dtype=np.float64) * self.deploy_kp_kd_scale
        _, _, resolved = resolve_matching_names_values(
            self.robot_options.default_dofs_kd, self.dofs_name, preserve_order=True
        )
        self.default_dof_kd = np.asarray(resolved, dtype=np.float64) * self.deploy_kp_kd_scale
        self._processed_action = torch.zeros(self.num_dofs, device=self._env.device)
        self._term_action_maps = self._build_term_action_maps()

        self._rate = None
        if self.sync and self.control_freq is not None:
            self._rate = RateLimiter(self.control_freq)

        # Replace the sim entity with a lightweight state wrapper so that
        # observation terms read from real robot state instead of the sim.
        sim_entity = self._env.entities[self.entity_name]
        self._state_entity = RobotStateEntity(sim_entity, device=self._env.device)
        self._env.entities[self.entity_name] = self._state_entity
        # Update cached entity references in class-based observation terms.
        for group_terms in self._env.observation_manager._group_obs_terms.values():
            for term in group_terms:
                if hasattr(term, "entity") and getattr(term, "entity_name", None) == self.entity_name:
                    term.entity = self._state_entity

    def _resolve_control_dt(self) -> float:
        if self.control_dt is not None:
            return self.control_dt
        if self.control_freq is not None:
            return 1.0 / self.control_freq
        decimation = self.decimation or self.cfg.env_options.decimation
        return self.cfg.env_options.sim_dt * decimation

    @staticmethod
    def _resolve_robot_options(cfg: EdenRLConfig, entity_name: str) -> RobotOptions:
        if not hasattr(cfg.scene_options, entity_name):
            raise ValueError(f"Scene has no entity named '{entity_name}'.")
        robot_options = getattr(cfg.scene_options, entity_name)
        if not isinstance(robot_options, RobotOptions):
            raise TypeError(f"Entity '{entity_name}' is not a RobotOptions instance: {type(robot_options)}")
        return robot_options

    def _build_term_action_maps(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Precompute index mappings from each action term's local DOF order to deployment DOF order."""
        dofs_name_to_idx = {name: i for i, name in enumerate(self.dofs_name)}
        maps = []
        for _, term in self._env.action_manager._terms.items():
            src_idx = []
            dst_idx = []
            for j, name in enumerate(term.dofs_name):
                if name in dofs_name_to_idx:
                    src_idx.append(j)
                    dst_idx.append(dofs_name_to_idx[name])
            maps.append(
                (
                    torch.tensor(src_idx, dtype=torch.long, device=self._env.device),
                    torch.tensor(dst_idx, dtype=torch.long, device=self._env.device),
                )
            )
        return maps

    def action_to_payload(self, action: Optional[torch.Tensor | dict[str, torch.Tensor]] = None) -> RobotCommand:
        if action is not None:
            self._env.action_manager.compute(action)

            for (src_idx, dst_idx), (_, term) in zip(self._term_action_maps, self._env.action_manager._terms.items()):
                self._processed_action[dst_idx] = term._processed_action[0, src_idx]

        return RobotCommand(
            dofs_pos=self._processed_action.cpu().numpy(),
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp,
            dofs_kd=self.default_dof_kd,
        )

    def step(self, action: Optional[torch.Tensor | dict[str, torch.Tensor]] = None) -> dict[str, Any]:
        if action is not None:
            payload = self.action_to_payload(action)
            self.send_payload(payload)
        if self._rate is not None:
            self._rate.sleep()
        if self._env.command_manager is not None:
            self._env.command_manager.compute(dt=self.control_dt)
            self._env.command_manager.draw_vis()
        self.state = self.read_state()
        return self.state_to_observation(self.state)

    def reset(self) -> dict[str, Any]:
        start_event = threading.Event()

        def _wait_for_enter() -> None:
            input("Press Enter to continue...")
            start_event.set()

        threading.Thread(target=_wait_for_enter, daemon=True).start()

        while not start_event.is_set():
            self.init_sequence()
            if self._rate is not None:
                self._rate.sleep()

        self.state = self.read_state()
        if self.auto_yaw_align and self.state.base_quat is not None:
            yaw = self._state_entity.capture_yaw_offset(self.state.base_quat)
            en.logger.info(f"[deployment] Captured robot yaw offset: {np.degrees(yaw):.2f} deg")
        obs = self.state_to_observation(self.state)
        return obs

    def state_to_observation(self, state: RobotState) -> dict[str, Any]:
        self._state_entity.update(state)
        return self._env.observation_manager.compute(update_history=True)

    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_sequence(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_state(self) -> RobotState:
        raise NotImplementedError

    @abstractmethod
    def send_payload(self, payload: RobotCommand) -> None:
        raise NotImplementedError

    @staticmethod
    def _to_numpy(value: np.ndarray | Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        try:
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(value, dtype=np.float64)
