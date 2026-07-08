"""Real-robot deployment backends.

Importing this package imports each deployment module so that:

1. ``@DEPLOYMENT_REGISTRY.register()`` decorators run and populate the
   string→class registry.
2. ``bind_deployment(robot_cls, deployment_name)`` calls run and populate
   the robot→deployment binding map consulted by
   :func:`eden.extensions.deployment.base.resolve_deployment_for`.

``eden deploy --task <name>`` relies on (2) to pick the right backend
automatically from the task's bound robot.
"""

from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    bind_deployment,
    resolve_deployment_for,
)

# Importing each module triggers its registry + binding side-effects. The
# imports are guarded inside ``try`` so a missing optional dependency (e.g.
# the Unitree SDK) doesn't break the whole package import; the missing
# backend simply won't be available at resolve time.
try:
    from eden.extensions.deployment import robotera_xhand  # noqa: F401
except ImportError:
    pass


__all__ = [
    "DEPLOYMENT_REGISTRY",
    "DeploymentBase",
    "bind_deployment",
    "resolve_deployment_for",
]
