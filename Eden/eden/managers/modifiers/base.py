"""Base classes and registries for manager term modifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.constants import NoiseOperation
from eden.managers.base import ManagerTermBase
from eden.options.managers.modifiers import ActionModifierOptions, NoiseOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


class NoiseModel(ManagerTermBase, ConfigurableMixin[NoiseOptions]):
    """Base class for noise models applied to observation terms."""

    operation: NoiseOperation = NoiseOperation.ADD

    def __init__(self, env: EnvBase, options: NoiseOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        pass


NOISE_MODEL_REGISTRY = Registry("NOISE_MODEL")


class ActionModifier(ManagerTermBase, ConfigurableMixin[ActionModifierOptions]):
    """Base class for composable action modifiers.

    Action modifiers intercept and transform processed actions (target
    positions/velocities) and/or control torques. They are designed to be
    composed together using :class:`~eden.managers.modifiers.actions.actuators.Compose`.

    Subclasses should override one or both of:
    - ``modify_processed_action``: transform target positions/velocities before PD control.
    - ``modify_ctrl_torque``: transform computed torques after PD control.
    """

    def __init__(self, env: EnvBase, options: ActionModifierOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

    def build(
        self,
        num_envs: int,
        device: str,
        entity: "Entity" | None = None,
        dofs_idx_local: torch.Tensor | None = None,
    ) -> None:
        """Initialize internal buffers, called by the owning action term after the entity is resolved.

        Parameters
        ----------
        num_envs : int
            Number of parallel environments.
        device : str
            Torch device string.
        entity : Entity, optional
            The entity being controlled. Needed by state-dependent modifiers.
        dofs_idx_local : torch.Tensor, optional
            Local DOF indices being controlled.
        """
        pass

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        pass

    def compute(self) -> None:
        """No-op by default. Override if the modifier needs per-step computation."""
        pass

    def modify_processed_action(self, processed_action: torch.Tensor) -> torch.Tensor:
        """Transform target positions/velocities before PD control. Default is identity."""
        return processed_action

    def modify_ctrl_torque(
        self,
        ctrl_torque: torch.Tensor,
        dofs_vel: torch.Tensor,
        pos_err: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transform computed torques after PD control. Default is identity.

        ``pos_err`` (target - current joint position) is supplied by PD-position
        controllers for modifiers that need it (e.g. position deadbands); it is
        ``None`` for velocity-only controllers.
        """
        return ctrl_torque

    def get(self, cls: type) -> ActionModifier | None:
        """Return self if it is an instance of ``cls``, else ``None``.

        :class:`~eden.managers.modifiers.actions.actuators.Compose` overrides this
        to search its children. Callers that require the modifier to exist should
        check for ``None`` and raise their own actionable error.
        """
        return self if isinstance(self, cls) else None


ACTION_MODIFIER_REGISTRY = Registry("ACTION_MODIFIER")
