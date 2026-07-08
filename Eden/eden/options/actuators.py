"""Actuator specification options and motor presets (Unitree, DAMIAO)."""

from __future__ import annotations
from typing import ClassVar

from eden.options.options import ConfigurableOptions
from eden.utils.actuator import (
    reflected_inertia,
    reflected_inertia_from_two_stage_planetary,
    raibert_heuristic_kp,
    raibert_heuristic_kd,
    raibert_heuristic_action_scale,
)


class ActuatorSpecOptions(ConfigurableOptions):
    """Specification of a single actuator's physical and control parameters.

    Parameters
    ----------
    STIFFNESS: float
        Stiffness of the actuator (different from control stiffness, i.e., kp) [N*m/rad]
    DAMPING: float
        Damping of the actuator (different from control damping, i.e., kd) [N*m*s/rad]
    FULL_TORQUE_SPEED: float
        (X1) Maximum Speed at Full Torque (T-N Curve Knee Point) [rad/s]
    NO_LOAD_SPEED: float
        (X2) No-Load Speed Test Result [rad/s]
    DRIVING_TORQUE_LIMIT: float
        (Y1) Driving Torque Limit, i.e., Torque and Speed in the Same Direction [N*m]
    BRAKING_TORQUE_LIMIT: float | None
        (Y2) Braking Torque Limit, i.e., Torque and Speed in the Opposite Direction [N*m]
    EFFORT_LIMIT: float
        Maximum torque limit to prevent mechanical damage [N*m]
    STATIC_FRICTION: float
        Static Friction Coefficient
    DYNAMIC_FRICTION: float
        Dynamic Friction Coefficient
    FRICTION_ACTIVATION_SPEED: float
        Velocity at which the friction is fully activated [rad/s]
    ROTOR_INERTIA: float | tuple[float, float, float]
        Rotor inertia of the motor [kg*m^2]
    GEAR_RATIO: float | tuple[float, float, float]
        Gear ratio of the gearbox

    NOTE
    ----
    Per-joint motor torque-speed envelope parameters.

    The torque-speed curve is defined as follows:

            Torque Limit, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2
    """

    STIFFNESS: ClassVar[float] = 0.0
    DAMPING: ClassVar[float] = 0.0
    FULL_TORQUE_SPEED: ClassVar[float] = 1e9
    NO_LOAD_SPEED: ClassVar[float] = 1e9
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 1e9
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 1e9
    EFFORT_LIMIT: ClassVar[float] = 1e9
    STATIC_FRICTION: ClassVar[float] = 0.0
    DYNAMIC_FRICTION: ClassVar[float] = 0.0
    FRICTION_ACTIVATION_SPEED: ClassVar[float] = 0.01

    ROTOR_INERTIA: ClassVar[float | tuple[float, float, float]] = 0.0
    GEAR_RATIO: ClassVar[float | tuple[float, float, float]] = 1.0

    @classmethod
    def ARMATURE(cls) -> float:
        if isinstance(cls.ROTOR_INERTIA, float):
            assert isinstance(cls.GEAR_RATIO, float), "Expected single-stage gearbox (1 gear ratio)"
            return reflected_inertia(cls.ROTOR_INERTIA, cls.GEAR_RATIO)
        elif isinstance(cls.ROTOR_INERTIA, tuple):
            assert isinstance(cls.GEAR_RATIO, tuple), "Expected two-stage planetary gearbox (3 gear ratios)"
            assert len(cls.ROTOR_INERTIA) == len(cls.GEAR_RATIO)
            assert len(cls.GEAR_RATIO) == 3, "Expected two-stage planetary gearbox (3 gear ratios)"
            return reflected_inertia_from_two_stage_planetary(cls.ROTOR_INERTIA, cls.GEAR_RATIO)
        else:
            return 0.0

    @classmethod
    def RAIBERT_HEURISTIC_KP(cls) -> float:
        """Return the control stiffness (kp) computed using the Raibert Heuristic [N*m/rad]."""
        return raibert_heuristic_kp(cls)

    @classmethod
    def RAIBERT_HEURISTIC_KD(cls, damping_ratio: float = 2.0) -> float:
        """Return the control damping (kd) computed using the Raibert Heuristic [N*m*s/rad]."""
        return raibert_heuristic_kd(cls, damping_ratio=damping_ratio)

    @classmethod
    def RAIBERT_HEURISTIC_ACTION_SCALE(cls) -> float:
        return raibert_heuristic_action_scale(cls)


class Unitree7520_22(ActuatorSpecOptions):
    """Unitree N7520-22.5 motor used in G1 robot."""

    FULL_TORQUE_SPEED: ClassVar[float] = 14.5
    NO_LOAD_SPEED: ClassVar[float] = 22.7
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 111.0
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 131.0
    EFFORT_LIMIT: ClassVar[float] = 139.0
    STATIC_FRICTION: ClassVar[float] = 2.4
    DYNAMIC_FRICTION: ClassVar[float] = 0.24
    ROTOR_INERTIA: ClassVar[tuple[float, float, float]] = (
        0.489e-4,
        0.109e-4,
        0.738e-4,
    )
    GEAR_RATIO: ClassVar[tuple[float, float, float]] = (1.0, 4.5, 5.0)


class Unitree7520_14(ActuatorSpecOptions):
    """Unitree N7520-14.3 motor used in G1 robot."""

    FULL_TORQUE_SPEED: ClassVar[float] = 22.63
    NO_LOAD_SPEED: ClassVar[float] = 35.52
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 71.0
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 83.3
    EFFORT_LIMIT: ClassVar[float] = 88.0
    STATIC_FRICTION: ClassVar[float] = 1.6
    DYNAMIC_FRICTION: ClassVar[float] = 0.16
    ROTOR_INERTIA: ClassVar[tuple[float, float, float]] = (
        0.489e-4,
        0.098e-4,
        0.533e-4,
    )
    GEAR_RATIO: ClassVar[tuple[float, float, float]] = (1.0, 4.5, 1.0 + (48 / 22))


class Unitree5020_16(ActuatorSpecOptions):
    """Unitree N5020-16 motor used in G1 robot."""

    FULL_TORQUE_SPEED: ClassVar[float] = 30.86
    NO_LOAD_SPEED: ClassVar[float] = 40.13
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 24.8
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 31.9
    EFFORT_LIMIT: ClassVar[float] = 25.0
    STATIC_FRICTION: ClassVar[float] = 0.6
    DYNAMIC_FRICTION: ClassVar[float] = 0.06
    ROTOR_INERTIA: ClassVar[tuple[float, float, float]] = (
        0.139e-4,
        0.017e-4,
        0.169e-4,
    )
    GEAR_RATIO: ClassVar[tuple[float, float, float]] = (1.0, 1.0 + (46 / 18), 1.0 + (56 / 16))


class Unitree5020_16_DOUBLE(Unitree5020_16):
    """Two Unitree N5020-16 motors driving a single joint via a 4-bar linkage.

    Used for G1 waist pitch/roll and G1 ankle pitch/roll.

    Assumes a nominal 1:1 linkage ratio. Under this assumption:
      - Joint speed limits are identical to a single motor (rotors spin
        synchronously at joint speed).
      - Joint torque, friction, and reflected inertia are the sum of both
        motors' contributions.

    Note that the true effective armature is configuration-dependent because of the linkage geometry.
    """

    DRIVING_TORQUE_LIMIT: ClassVar[float] = Unitree5020_16.DRIVING_TORQUE_LIMIT * 2
    BRAKING_TORQUE_LIMIT: ClassVar[float] = Unitree5020_16.BRAKING_TORQUE_LIMIT * 2
    EFFORT_LIMIT: ClassVar[float] = Unitree5020_16.EFFORT_LIMIT * 2
    STATIC_FRICTION: ClassVar[float] = Unitree5020_16.STATIC_FRICTION * 2
    DYNAMIC_FRICTION: ClassVar[float] = Unitree5020_16.DYNAMIC_FRICTION * 2
    ROTOR_INERTIA: ClassVar[tuple[float, float, float]] = tuple(2.0 * r for r in Unitree5020_16.ROTOR_INERTIA)


class Unitree4010_25(ActuatorSpecOptions):
    """Unitree W4010-25 motor used in G1 robot."""

    FULL_TORQUE_SPEED: ClassVar[float] = 15.3
    NO_LOAD_SPEED: ClassVar[float] = 24.76
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 4.8
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 8.6
    EFFORT_LIMIT: ClassVar[float] = 5.0
    STATIC_FRICTION: ClassVar[float] = 0.6
    DYNAMIC_FRICTION: ClassVar[float] = 0.06
    ROTOR_INERTIA: ClassVar[tuple[float, float, float]] = (
        0.068e-4,
        0.0,
        0.0,
    )
    GEAR_RATIO: ClassVar[tuple[float, float, float]] = (1.0, 5.0, 5.0)


class Unitree8010_6(ActuatorSpecOptions):
    """Unitree M8010-6 motor used in Go1 robot."""

    FULL_TORQUE_SPEED: ClassVar[float] = 13.5
    NO_LOAD_SPEED: ClassVar[float] = 30.0
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 20.2
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 23.4
    EFFORT_LIMIT: ClassVar[float] = 23.7
    ROTOR_INERTIA: ClassVar[float] = 0.000111842
    GEAR_RATIO: ClassVar[float] = 6.33


class Unitree8010_6Calf(Unitree8010_6):
    """Unitree M8010-6 motor used in Go1 robot with 1.5 gear ratio (linkage ratio) for calf joint."""

    FULL_TORQUE_SPEED: ClassVar[float] = Unitree8010_6.FULL_TORQUE_SPEED / 1.5
    NO_LOAD_SPEED: ClassVar[float] = Unitree8010_6.NO_LOAD_SPEED / 1.5
    DRIVING_TORQUE_LIMIT: ClassVar[float] = Unitree8010_6.DRIVING_TORQUE_LIMIT * 1.5
    BRAKING_TORQUE_LIMIT: ClassVar[float] = Unitree8010_6.BRAKING_TORQUE_LIMIT * 1.5
    EFFORT_LIMIT: ClassVar[float] = Unitree8010_6.EFFORT_LIMIT * 1.5
    GEAR_RATIO: ClassVar[float] = Unitree8010_6.GEAR_RATIO * 1.5


class UnitreeGo2HV(ActuatorSpecOptions):
    """Unitree Go2 HV motor used in Go2 robot (this is a best guess)."""

    FULL_TORQUE_SPEED: ClassVar[float] = 16.0
    NO_LOAD_SPEED: ClassVar[float] = 36.0
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 30.0
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 34.5
    EFFORT_LIMIT: ClassVar[float] = 35.55
    ROTOR_INERTIA: ClassVar[float] = 0.000125
    GEAR_RATIO: ClassVar[float] = 6.22


class UnitreeGo2HVCalf(UnitreeGo2HV):
    """Unitree Go2 HV motor used in Go2 robot with 1.5 gear ratio (linkage ratio) for calf joint."""

    FULL_TORQUE_SPEED: ClassVar[float] = UnitreeGo2HV.FULL_TORQUE_SPEED / 1.5
    NO_LOAD_SPEED: ClassVar[float] = UnitreeGo2HV.NO_LOAD_SPEED / 1.5
    DRIVING_TORQUE_LIMIT: ClassVar[float] = UnitreeGo2HV.DRIVING_TORQUE_LIMIT * 1.5
    BRAKING_TORQUE_LIMIT: ClassVar[float] = UnitreeGo2HV.BRAKING_TORQUE_LIMIT * 1.5
    EFFORT_LIMIT: ClassVar[float] = UnitreeGo2HV.EFFORT_LIMIT * 1.5
    GEAR_RATIO: ClassVar[float] = UnitreeGo2HV.GEAR_RATIO * 1.5


class DamiaoDM4310(ActuatorSpecOptions):
    """DAMIAO DM-J4310-2EC servo motor used as the wrist-joint and gripper motor on the i2rt YAM arm.

    24V, 10:1 internal planetary gearbox, gearbox-output peak torque 7 N·m, driving
    J4-J6 plus the fingers.

    Speed/torque values are at the gearbox output shaft per the data sheet;
    ``ROTOR_INERTIA`` is the motor-side rotor inertia and ``GEAR_RATIO``
    folds in the internal planetary, so :meth:`ARMATURE` returns the
    joint-side reflected armature.
    """

    FULL_TORQUE_SPEED: ClassVar[float] = 12.57  # 120 rpm
    NO_LOAD_SPEED: ClassVar[float] = 20.94  # 200 rpm
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 3.0
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 3.0
    EFFORT_LIMIT: ClassVar[float] = 7.0
    ROTOR_INERTIA: ClassVar[float] = 0.000018  # TODO
    GEAR_RATIO: ClassVar[float] = 10.0


class DamiaoDM4340(ActuatorSpecOptions):
    """DAMIAO DM-J4340-2EC servo motor (24V variant).

    40:1 internal planetary gearbox, joint-side nominal/peak torque 9 / 27 N·m.

    Datasheet (24V): nominal current 2.5 A, peak current 8 A,
    nominal torque 9 N·m, peak torque 27 N·m,
    nominal speed 36 rpm, max no-load speed 52 rpm.

    Speed/torque values are at the gearbox output shaft per the data sheet;
    ``ROTOR_INERTIA`` is the motor-side rotor inertia and ``GEAR_RATIO``
    folds in the internal planetary, so :meth:`ARMATURE` returns the
    joint-side reflected armature (≈ 0.032 kg·m²).
    """

    FULL_TORQUE_SPEED: ClassVar[float] = 3.77  # 36 rpm
    NO_LOAD_SPEED: ClassVar[float] = 5.45  # 52 rpm
    DRIVING_TORQUE_LIMIT: ClassVar[float] = 9.0
    BRAKING_TORQUE_LIMIT: ClassVar[float] = 9.0
    EFFORT_LIMIT: ClassVar[float] = 27.0
    ROTOR_INERTIA: ClassVar[float] = 2e-5  # TODO
    GEAR_RATIO: ClassVar[float] = 40.0
