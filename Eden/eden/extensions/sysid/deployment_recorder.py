"""Recorder that captures real-robot trajectories for system identification."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

import eden as en
from eden.extensions.deployment.utils.rate import RateLimiter
from eden.extensions.deployment.utils.state import RobotCommand
from eden.extensions.sysid.excitation import Excitation
from eden.extensions.sysid.trajectory import Trajectory

if TYPE_CHECKING:
    from eden.extensions.deployment.base import DeploymentBase


class DeploymentRecorder:
    """Drive a ``DeploymentBase`` with an ``Excitation`` and record a ``Trajectory``.

    The loop writes the excitation **offset** on top of a center pose at
    every control tick, captures the returned ``RobotState``, and after the
    duration serialises everything to a ``Trajectory`` compatible with the
    sysid rollout loader.

    Parameters
    ----------
    deployer: DeploymentBase
        Active deployment backend; must be connected and post-init_sequence.
    excitation: Excitation
        Excitation generator. Called once per control tick with the elapsed
        seconds since ``run`` was invoked.
    kp_scale, kd_scale: float
        Multipliers on ``deployer.default_dof_kp`` / ``..._kd`` used for
        this recording. Lower kp is safer and surfaces compliance/damping
        in the response; higher kp stiffens the reference-tracking.
    safety_limits: np.ndarray | None
        Optional per-DOF symmetric clamp on the excitation offset
        (radians). Defaults to no additional clamp beyond whatever the
        backend already enforces.
    center_dofs_pos: np.ndarray | None
        Optional absolute joint-position center for the excitation. Defaults
        to the robot's actual position at recording start.
    dofs_limit: tuple[np.ndarray, np.ndarray] | None
        Optional per-DOF lower and upper position limits used for diagnostics.
    """

    def __init__(
        self,
        deployer: "DeploymentBase",
        excitation: Excitation,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        safety_limits: np.ndarray | None = None,
        center_dofs_pos: np.ndarray | None = None,
        dofs_limit: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if excitation.num_dofs != deployer.num_dofs:
            raise ValueError(f"Excitation has {excitation.num_dofs} DOFs but deployer expects {deployer.num_dofs}.")
        self.deployer = deployer
        self.excitation = excitation
        self.kp = deployer.default_dof_kp * float(kp_scale)
        self.kd = deployer.default_dof_kd * float(kd_scale)
        self.safety_limits = np.asarray(safety_limits, dtype=np.float64) if safety_limits is not None else None
        if self.safety_limits is not None and self.safety_limits.size != deployer.num_dofs:
            raise ValueError("safety_limits size must equal deployer.num_dofs.")
        self.center_dofs_pos = np.asarray(center_dofs_pos, dtype=np.float64) if center_dofs_pos is not None else None
        if self.center_dofs_pos is not None and self.center_dofs_pos.size != deployer.num_dofs:
            raise ValueError("center_dofs_pos size must equal deployer.num_dofs.")
        if dofs_limit is None:
            self.dofs_limit = None
        else:
            lo, hi = dofs_limit
            self.dofs_limit = (np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
            if self.dofs_limit[0].size != deployer.num_dofs or self.dofs_limit[1].size != deployer.num_dofs:
                raise ValueError("dofs_limit arrays must equal deployer.num_dofs.")
        # Own our own rate limiter rather than reusing the deployer's, so that
        # data collection can run at its own cadence and does not race with
        # whatever other loop may hold the deployer's limiter.
        self._rate = RateLimiter(deployer.control_freq) if deployer.control_freq else None

    def run(self) -> Trajectory:
        """Execute the excitation for ``excitation.duration`` seconds.

        By default, the chirp / PRBS offset is added on top of the hand's
        **actual resting pose** (``read_state().dofs_pos``). Callers can pass
        ``center_dofs_pos`` to use a known center pose instead, such as the
        midpoint of each joint's position limits.

        Returns the recorded Trajectory. The timestamp column starts at
        zero (relative to loop start); the caller can shift externally if
        absolute alignment with other logs is needed.
        """
        deployer = self.deployer
        duration = self.excitation.duration

        stamps: list[float] = []
        actions: list[np.ndarray] = []
        dofs_pos: list[np.ndarray] = []
        dofs_vel: list[np.ndarray] = []
        dofs_torque: list[np.ndarray] = []
        base_quat: list[np.ndarray] = []
        base_ang_vel: list[np.ndarray] = []
        base_lin_acc: list[np.ndarray] = []

        initial_state = deployer.read_state()
        q0 = np.asarray(initial_state.dofs_pos, dtype=np.float64).copy()
        center = q0 if self.center_dofs_pos is None else self.center_dofs_pos

        # Warn if the requested excitation range leaves the known joint limits.
        try:
            if self.dofs_limit is None:
                entity = deployer._env.entities[deployer.entity_name]
                q_lim_lo, q_lim_hi = entity.get_dofs_limit()
                q_lim_lo = q_lim_lo[0].detach().cpu().numpy() if q_lim_lo.ndim == 2 else q_lim_lo.detach().cpu().numpy()
                q_lim_hi = q_lim_hi[0].detach().cpu().numpy() if q_lim_hi.ndim == 2 else q_lim_hi.detach().cpu().numpy()
            else:
                q_lim_lo, q_lim_hi = self.dofs_limit
            # Peak excitation magnitude per DOF (best-effort; safe-fallback to a
            # scan of a short time window if the Excitation doesn't expose it).
            peak = np.zeros(deployer.num_dofs, dtype=np.float64)
            for t in np.linspace(0.0, duration, 64):
                peak = np.maximum(peak, np.abs(self.excitation(t)))
            if self.safety_limits is not None:
                peak = np.minimum(peak, self.safety_limits)
            below = (center - peak) < q_lim_lo
            above = (center + peak) > q_lim_hi
            if np.any(below | above):
                names = list(deployer.dofs_name)
                for i in range(deployer.num_dofs):
                    if below[i] or above[i]:
                        en.logger.warning(
                            f"DOF '{names[i]}' excitation range "
                            f"[{center[i] - peak[i]:+.3f}, {center[i] + peak[i]:+.3f}] "
                            f"exceeds URDF limits [{q_lim_lo[i]:+.3f}, {q_lim_hi[i]:+.3f}]. "
                            f"The real hand will clip at the mechanical stop and this joint "
                            f"will be hard to identify. Lower --dofs-range-ratio."
                        )
        except Exception as exc:  # pragma: no cover — diagnostic, don't crash collection
            en.logger.debug(f"joint-limit check skipped: {exc!r}")

        t0 = time.monotonic()
        while True:
            t = time.monotonic() - t0
            if t >= duration:
                break

            offset = self.excitation(t)
            if self.safety_limits is not None:
                offset = np.clip(offset, -self.safety_limits, self.safety_limits)
            target = center + offset

            cmd = RobotCommand(
                dofs_pos=target,
                dofs_vel=np.zeros(deployer.num_dofs),
                dofs_torque=np.zeros(deployer.num_dofs),
                dofs_kp=self.kp,
                dofs_kd=self.kd,
            )
            deployer.send_payload(cmd)
            if self._rate is not None:
                self._rate.sleep()
            state = deployer.read_state()

            stamps.append(t)
            actions.append(target.astype(np.float64, copy=True))
            dofs_pos.append(np.asarray(state.dofs_pos, dtype=np.float64))
            dofs_vel.append(np.asarray(state.dofs_vel, dtype=np.float64))
            dofs_torque.append(np.asarray(state.dofs_torque, dtype=np.float64))
            if state.base_quat is not None:
                base_quat.append(np.asarray(state.base_quat, dtype=np.float64))
            if state.base_ang_vel is not None:
                base_ang_vel.append(np.asarray(state.base_ang_vel, dtype=np.float64))
            if state.base_lin_acc is not None:
                base_lin_acc.append(np.asarray(state.base_lin_acc, dtype=np.float64))

        traj = Trajectory(
            times=np.asarray(stamps),
            action=_stack_or_none(actions),
            dofs_pos=_stack_or_none(dofs_pos),
            dofs_vel=_stack_or_none(dofs_vel),
            dofs_torque=_stack_or_none(dofs_torque),
            base_quat=_stack_or_none(base_quat),
            base_ang_vel=_stack_or_none(base_ang_vel),
            base_lin_acc=_stack_or_none(base_lin_acc),
            dof_names=tuple(deployer.dofs_name),
            initial_state={"qpos": q0, "dofs_vel": np.zeros_like(q0)},
        )
        return traj


def _stack_or_none(rows: list[np.ndarray]) -> np.ndarray | None:
    return np.stack(rows, axis=0) if rows else None
