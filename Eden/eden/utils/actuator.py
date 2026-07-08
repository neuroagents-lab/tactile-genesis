"""Actuator math helpers (reflected inertia, RPM conversions, Raibert heuristic)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from eden.options.actuators import ActuatorSpecOptions


# 10 Hz target closed-loop bandwidth for the Raibert PD heuristic; basis for
# every default kp/kd/action_scale derived via the helpers below.
NATURAL_FREQ = 10 * 2.0 * np.pi


def reflected_inertia(
    rotor_inertia: float,
    gear_ratio: float,
) -> float:
    """Compute reflected inertia of a single-stage gearbox."""
    return rotor_inertia * gear_ratio**2


def reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected inertia of a two-stage planetary gearbox."""
    assert gear_ratio[0] == 1
    r1 = rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2
    r2 = rotor_inertia[1] * gear_ratio[2] ** 2
    r3 = rotor_inertia[2]
    return r1 + r2 + r3


def rpm_to_rad_per_sec(rpm: float) -> float:
    """Convert revolutions per minute to radians per second."""
    return (rpm * 2 * np.pi) / 60


def rad_per_sec_to_rpm(rad_per_sec: float) -> float:
    """Convert radians per second to revolutions per minute."""
    return (rad_per_sec * 60) / (2 * np.pi)


def raibert_heuristic_kp(actuator: type[ActuatorSpecOptions], natural_freq: float = NATURAL_FREQ) -> float:
    """Compute ideal stiffness for a given actuator and natural frequency."""
    return actuator.ARMATURE() * natural_freq**2


def raibert_heuristic_kd(
    actuator: type[ActuatorSpecOptions], natural_freq: float = NATURAL_FREQ, damping_ratio: float = 2.0
) -> float:
    """Compute ideal damping for a given actuator and natural frequency. Defaults to over-damped damping ratio of 2.0."""
    return 2.0 * damping_ratio * actuator.ARMATURE() * natural_freq


def raibert_heuristic_action_scale(actuator: type[ActuatorSpecOptions], natural_freq: float = NATURAL_FREQ) -> float:
    """``0.25 * effort_limit / kp``, where ``kp`` is computed using the Raibert Heuristic."""
    return 0.25 * actuator.EFFORT_LIMIT / raibert_heuristic_kp(actuator, natural_freq)
