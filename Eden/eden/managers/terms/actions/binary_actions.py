"""Binary (open/close) gripper action term."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


@ACTION_TERM_REGISTRY.register()
class BinaryJointController(ActionTerm):
    """A binary joint controller for a parallel-jaw gripper.

    When the action is applied, the joints are opened or closed.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    dofs_name: list[str]
        The names of the DOFs to control.
    open_action: float | dict[str, float]
        The action to open the joints.
    close_action: float | dict[str, float]
        The action to close the joints.
    scale: float | dict[str, float]
        The scale to apply to the actions.

    Note
    ----
    action is a boolean tensor, True: close, False: open
    if float is used, >0: close, <=0: open
    """

    open_action: float | dict[str, float] = 1.0
    close_action: float | dict[str, float] = 0.0
    scale: float = 1.0

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._open_action: torch.Tensor | None = None
        self._close_action: torch.Tensor | None = None

    def build(self) -> None:
        super().build()

        self._open_action = torch.zeros(1, self.num_dofs, device=self.device)
        self._close_action = torch.zeros(1, self.num_dofs, device=self.device)

        if isinstance(self.open_action, float):
            self._open_action[:] = self.open_action
        elif isinstance(self.open_action, dict):
            for dofs_name, open_action in self.open_action.items():
                _, dofs_idx = self.entity.find_named_dofs_idx_local(dofs_name, name_scope=self.entity.dofs_name)
                dofs_idx = [self.dofs_idx_map[idx] for idx in dofs_idx]
                self._open_action[:, dofs_idx] = open_action
        else:
            raise ValueError(f"Unsupported open_action type: {type(self.open_action)}.")

        if isinstance(self.close_action, float):
            self._close_action[:] = self.close_action
        elif isinstance(self.close_action, dict):
            for dofs_name, close_action in self.close_action.items():
                _, dofs_idx = self.entity.find_named_dofs_idx_local(dofs_name, name_scope=self.entity.dofs_name)
                dofs_idx = [self.dofs_idx_map[idx] for idx in dofs_idx]
                self._close_action[:, dofs_idx] = close_action
        else:
            raise ValueError(f"Unsupported close_action type: {type(self.close_action)}.")

    @property
    def action_dim(self) -> int:
        return 1

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if actions.dtype == torch.bool:
            close_mask = actions
        else:
            close_mask = (actions * self.scale) > 0
        self._processed_action = (
            self._open_action.repeat(self.num_envs, 1) * (~close_mask)
            + self._close_action.repeat(self.num_envs, 1) * close_mask
        )

    def reset(self, envs_idx: torch.Tensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = slice(None)
        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

    def apply_actions(self) -> None:
        self.entity.control_dofs_pos(self._processed_action, dofs_idx_local=self.dofs_idx_local)
