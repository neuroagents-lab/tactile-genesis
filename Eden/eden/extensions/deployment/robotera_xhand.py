"""RobotEra XHand real-robot deployment backend."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Literal

import time

import numpy as np

import eden as en
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY, DeploymentBase
from eden.extensions.deployment.utils.state import RobotCommand, RobotState

try:
    from xhand_controller import xhand_control
except ImportError:
    xhand_control = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.extensions.deployment import DeploymentOptions


# Maps each URDF joint name (right or left) to its 0-based finger command index in the
# xhand_control API.  Both hands share the same ordinal scheme (0-11); the hand side is
# selected at the device level via ``hand_id``.
XHAND_FINGER_IDX: dict[str, int] = {
    # Right hand
    "right_hand_thumb_bend_joint": 0,
    "right_hand_thumb_rota_joint1": 1,
    "right_hand_thumb_rota_joint2": 2,
    "right_hand_index_bend_joint": 3,
    "right_hand_index_joint1": 4,
    "right_hand_index_joint2": 5,
    "right_hand_mid_joint1": 6,
    "right_hand_mid_joint2": 7,
    "right_hand_ring_joint1": 8,
    "right_hand_ring_joint2": 9,
    "right_hand_pinky_joint1": 10,
    "right_hand_pinky_joint2": 11,
    # Left hand
    "left_hand_thumb_bend_joint": 0,
    "left_hand_thumb_rota_joint1": 1,
    "left_hand_thumb_rota_joint2": 2,
    "left_hand_index_bend_joint": 3,
    "left_hand_index_joint1": 4,
    "left_hand_index_joint2": 5,
    "left_hand_mid_joint1": 6,
    "left_hand_mid_joint2": 7,
    "left_hand_ring_joint1": 8,
    "left_hand_ring_joint2": 9,
    "left_hand_pinky_joint1": 10,
    "left_hand_pinky_joint2": 11,
}

_NUM_JOINTS = 12
# Fingers {2, 5, 7, 9, 11} carry tip sensors; sensor_data is ordered
# by their ascending finger-ID index (sensor_data[0] -> finger 2, etc.).
_SENSOR_FINGERS = (2, 5, 7, 9, 11)


@DEPLOYMENT_REGISTRY.register()
class RoboTeraXHandDeployment(DeploymentBase):
    """Deployment backend for the RoboTera XHand1 dexterous hand.

    Supports both EtherCAT and RS485 communication protocols.  For RS485 you
    must also set ``serial_port`` and ``baud_rate`` to match your hardware.

    The hand is treated as a fixed-base robot: ``read_state`` returns an
    identity quaternion and zero angular velocity because the hand has no IMU.
    """

    protocol: Literal["RS485", "EtherCAT"] = "RS485"
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 3_000_000
    hand_id: int = 0
    dofs_kp: int = 50
    dofs_ki: int = 0
    dofs_kd: int = 25
    tor_max: int = 300
    # 0: powerless  3: position (default)  5: powerful
    finger_mode: int = 3
    crc_retries: int = 3

    def __init__(self, env: EnvBase, options: DeploymentOptions) -> None:
        super().__init__(env, options)
        self._dof_index = [XHAND_FINGER_IDX[name] for name in self.dofs_name]

        self._device: Any | None = None
        self._hand_cmd: Any | None = None
        self._active_hand_id: int = self.hand_id

    # ------------------------------------------------------------------
    # DeploymentBase interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if xhand_control is None:
            raise RuntimeError(
                "xhand_controller package is not installed. Install it before using RoboTeraXHandDeployment."
            )

        self._device = xhand_control.XHandControl()

        if self.protocol == "EtherCAT":
            ports = self._device.enumerate_devices("EtherCAT")
            if not ports:
                raise RuntimeError("No EtherCAT XHand devices found.")
            rsp = self._device.open_ethercat(ports[0])
        elif self.protocol == "RS485":
            rsp = self._device.open_serial(self.serial_port, self.baud_rate)
        else:
            raise ValueError(f"Unknown protocol '{self.protocol}'. Expected 'RS485' or 'EtherCAT'.")

        if rsp.error_code != 0:
            raise RuntimeError(f"Failed to open XHand device: {rsp.error_message}")

        # Wait until the device enumerates at least one hand, then verify that the
        # configured hand_id is present.  When hand_id is 0 (default) we auto-select
        # the first discovered hand for single-hand setups.
        start_time = time.time()
        while True:
            ids = self._device.list_hands_id()
            if ids:
                if self.hand_id in ids:
                    self._active_hand_id = self.hand_id
                    break
                if self.hand_id == 0:
                    self._active_hand_id = ids[0]
                    break
                raise RuntimeError(f"Configured hand_id {self.hand_id} not found. Available hand IDs: {ids}")
            if time.time() - start_time > self.connect_timeout_s:
                raise RuntimeError("Timed out waiting for XHand device to enumerate hand IDs.")
            time.sleep(0.05)

        self._hand_cmd = xhand_control.HandCommand_t()
        for i in range(_NUM_JOINTS):
            fc = self._hand_cmd.finger_command[i]
            fc.id = i
            fc.kp = self.dofs_kp
            fc.ki = self.dofs_ki
            fc.kd = self.dofs_kd
            fc.position = 0.0
            fc.tor_max = self.tor_max
            fc.mode = self.finger_mode

    def close(self) -> None:
        if self._device is not None:
            self._set_finger_mode(0)  # release to powerless before disconnect
        self._device = None
        self._hand_cmd = None

    def init_sequence(self) -> None:
        payload = RobotCommand(
            dofs_pos=self._state_entity.default_dofs_pos[0].cpu().numpy(),
            dofs_vel=np.zeros(self.num_dofs),
            dofs_torque=np.zeros(self.num_dofs),
            dofs_kp=self.default_dof_kp,
            dofs_kd=self.default_dof_kd,
        )
        self.send_payload(payload)

    def read_state(self) -> RobotState:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")

        for attempt in range(self.crc_retries):
            err, state = self._device.read_state(self._active_hand_id, True)
            if err.error_code == 0:
                break
            en.logger.warning(f"read_state CRC error (attempt {attempt + 1}/{self.crc_retries}): {err.error_message}")
        else:
            raise RuntimeError(f"Failed to read XHand state after {self.crc_retries} attempts: {err.error_message}")

        dofs_pos = np.zeros(self.num_dofs, dtype=np.float64)
        # The XHand hardware does not report joint velocity; downstream terms will see zeros.
        dofs_vel = np.zeros(self.num_dofs, dtype=np.float64)
        dofs_torque = np.zeros(self.num_dofs, dtype=np.float64)

        joint_diagnostics: list[dict] = []
        for i, finger_idx in enumerate(self._dof_index):
            finger = state.finger_state[finger_idx]
            dofs_pos[i] = finger.position
            dofs_torque[i] = finger.torque
            joint_diagnostics.append(
                {
                    "id": finger.id,
                    "raw_position": finger.raw_position,
                    "sensor_id": finger.sensor_id,
                    "temperature": finger.temperature,
                    "commboard_err": finger.commboard_err,
                    "jonitboard_err": finger.jonitboard_err,
                    "tipboard_err": finger.tipboard_err,
                }
            )

        # Collect fingertip tactile data for the five sensor-equipped fingers.
        fingertip_sensors: dict[int, dict] = {}
        for sensor_idx, finger_id in enumerate(_SENSOR_FINGERS):
            sensor_data = state.sensor_data[sensor_idx]
            fingertip_sensors[finger_id] = {
                "calc_pressure": np.array(
                    [
                        sensor_data.calc_force.fx,
                        sensor_data.calc_force.fy,
                        sensor_data.calc_force.fz,
                    ]
                ),
                "raw_pressure": np.array([[force.fx, force.fy, force.fz] for force in sensor_data.raw_force]),
                "sensor_temperature": sensor_data.calc_temperature,
            }

        return RobotState(
            stamp=time.time(),
            # The XHand1 is a fixed-base end-effector with no on-board IMU.
            base_quat=np.array([1.0, 0.0, 0.0, 0.0]),  # identity (wxyz)
            base_ang_vel=np.zeros(3),
            base_lin_acc=np.zeros(3),
            dofs_pos=dofs_pos,
            dofs_vel=dofs_vel,
            dofs_torque=dofs_torque,
            extra={
                "fingertip_sensors": fingertip_sensors,
                "joint_diagnostics": joint_diagnostics,
            },
        )

    def send_payload(self, payload: RobotCommand) -> None:
        if self._device is None or self._hand_cmd is None:
            raise RuntimeError("XHand deployment is not connected.")

        for i, finger_idx in enumerate(self._dof_index):
            fc = self._hand_cmd.finger_command[finger_idx]
            fc.id = finger_idx
            fc.position = float(payload.dofs_pos[i])

        for attempt in range(self.crc_retries):
            err = self._device.send_command(self._active_hand_id, self._hand_cmd)
            if err.error_code == 0:
                break
            en.logger.warning(f"send_command CRC error (attempt {attempt + 1}/{self.crc_retries}): {err.error_message}")
        else:
            raise RuntimeError(f"Failed to send XHand command after {self.crc_retries} attempts: {err.error_message}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_finger_mode(self, mode: int) -> None:
        """Broadcast a mode change to all fingers and send immediately."""
        if self._device is None or self._hand_cmd is None:
            return
        for i in range(_NUM_JOINTS):
            self._hand_cmd.finger_command[i].mode = mode
        self._device.send_command(self._active_hand_id, self._hand_cmd)
        # Restore the configured operating mode so the next send_payload is correct.
        for i in range(_NUM_JOINTS):
            self._hand_cmd.finger_command[i].mode = self.finger_mode

    def reset_sensor(self, sensor_id: int) -> None:
        """Reset a fingertip sensor by its ID."""
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err = self._device.reset_sensor(self._active_hand_id, sensor_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to reset sensor {sensor_id}: {err.error_message}")

    def set_hand_id(self, new_id: int) -> None:
        """Change the hand ID on the device."""
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err = self._device.set_hand_id(self._active_hand_id, new_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to set hand ID from {self._active_hand_id} to {new_id}: {err.error_message}")
        self._active_hand_id = new_id

    def _sdk_version(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        return self._device.get_sdk_version()

    def _serial_number(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err, serial_number = self._device.get_serial_number(self._active_hand_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to get serial number: {err.error_message}")
        return serial_number

    def _hand_type(self) -> str:
        if self._device is None:
            raise RuntimeError("XHand deployment is not connected.")
        err, hand_type = self._device.get_hand_type(self._active_hand_id)
        if err.error_code != 0:
            raise RuntimeError(f"Failed to get hand type: {err.error_message}")
        return hand_type
