from __future__ import annotations

from typing import Any, Literal

import numpy as np
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY
from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
from eden.extensions.deployment.utils.state import RobotCommand, RobotState

from tactile_compare import load_tactile_scales


@DEPLOYMENT_REGISTRY.register()
class RoboTeraXHandTactileDeployment(RoboTeraXHandDeployment):
    tactile_sensor_type: Literal["bool", "agg_bool", "agg_force"] = "bool"
    tactile_bool_threshold: float = 0.1
    # Path to the real-tactile calibration YAML; None -> conf/sensor/xhand1_deploy_sensor_params.yaml.
    sensor_param_path: str | None = None
    dofs_kp: int = 22
    dofs_kd: int = 11

    @staticmethod
    def _resolve_fingertip_order(fingertip_sensors: dict[int, dict[str, Any]]) -> list[int]:
        return sorted(fingertip_sensors.keys())

    def _scale_fingertip_sensors(self, fingertip_sensors: dict[int, dict[str, Any]]) -> None:
        """Calibrate each finger's raw ``calc_pressure`` so the real reading matches sim.

        Applies a per-finger ``[fx, fy, fz]`` multiplier (magnitude scale plus the
        fx/fy sign flip) from ``conf/sensor/xhand1_deploy_sensor_params.yaml`` (loaded
        once). Mutating ``calc_pressure`` here -- before the observation is
        computed -- means both the policy observation and the deploy recording/
        plot see the calibrated value.
        """
        scales = getattr(self, "_tactile_scales_cache", None)
        if scales is None:
            scales = load_tactile_scales(self.sensor_param_path)
            self._tactile_scales_cache = scales
            print(
                f"[tactile] real-hand calc_pressure [fx,fy,fz] multiplier per finger id: "
                f"{ {fid: mult.tolist() for fid, mult in scales.items()} }"
            )
        for finger_id, reading in fingertip_sensors.items():
            if not isinstance(reading, dict) or "calc_pressure" not in reading:
                continue
            mult = scales.get(int(finger_id))
            if mult is not None:
                reading["calc_pressure"] = np.asarray(reading["calc_pressure"], dtype=np.float64) * mult

    def _augment_state_extra(self, state: RobotState) -> None:
        fingertip_sensors = state.extra.get("fingertip_sensors")
        if not isinstance(fingertip_sensors, dict):
            raise RuntimeError("Deploy tactile mode expects fingertip_sensors in RobotState.extra.")
        self._scale_fingertip_sensors(fingertip_sensors)
        state.extra["fingertip_sensor_order"] = self._resolve_fingertip_order(fingertip_sensors)
        state.extra["tactile_sensor_type"] = self.tactile_sensor_type
        state.extra["tactile_bool_threshold"] = float(self.tactile_bool_threshold)

    def action_to_payload(self, action=None) -> RobotCommand:
        # TODO: Fix in eden. DeploymentBase swaps the robot entity for observations, but action terms keep their cached sim-robot entity.
        if action is None:
            return super().action_to_payload(action)

        action_terms = list(self._env.action_manager._terms.values())
        previous_entities = [getattr(term, "_entity", None) for term in action_terms]
        for term in action_terms:
            if getattr(term, "entity_name", None) == self.entity_name:
                term._entity = self._state_entity
        try:
            return super().action_to_payload(action)
        finally:
            for term, entity in zip(action_terms, previous_entities, strict=True):
                term._entity = entity

    def state_to_observation(self, state: RobotState) -> dict[str, Any]:
        self._augment_state_extra(state)
        self._state_entity.update(state)
        # Keep deploy extras accessible for deploy-aware observation terms.
        self._state_entity.extra = dict(state.extra)
        return self._env.observation_manager.compute(update_history=True)
