"""Built-in metric terms."""

from eden.managers.terms.metrics.error import (
    error_vel_xy,
    error_vel_yaw,
)
from eden.managers.terms.metrics.proximity import (
    object_a_is_on_b,
    entity_distance,
    entity_distance_xy,
    entity_near_target,
    entity_near_entity_xy,
    EeNearEntity,
)
from eden.managers.terms.metrics.containment import (
    object_above_height,
    object_in_region,
)
from eden.managers.terms.metrics.grasp import (
    object_lifted,
    IsGrasping,
)
from eden.managers.terms.metrics.articulation import (
    joint_angle_at_target,
)
from eden.managers.terms.metrics.sequence import (
    SequentialMetricTerm,
)
from eden.managers.terms.metrics.boolean import (
    MetricTermAND,
    MetricTermOR,
    MetricTermNOT,
)
from eden.options.managers.metrics import PhaseOptions

__all__ = [
    "object_a_is_on_b",
    "error_vel_xy",
    "error_vel_yaw",
    # proximity
    "entity_distance",
    "entity_distance_xy",
    "entity_near_target",
    "entity_near_entity_xy",
    "EeNearEntity",
    # containment
    "object_above_height",
    "object_in_region",
    # grasp
    "object_lifted",
    "IsGrasping",
    # articulation
    "joint_angle_at_target",
    # sequence
    "SequentialMetricTerm",
    "PhaseOptions",
    # boolean
    "MetricTermAND",
    "MetricTermOR",
    "MetricTermNOT",
]
