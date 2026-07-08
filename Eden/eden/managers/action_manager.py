"""Action manager and ActionTerm base: map policy actions to actuator commands.

The manager slices the policy action vector across its active :class:`ActionTerm`
instances and applies them every control step (re-applied across ``decimation``
physics sub-steps). ``ActionManager.dofs_order`` maps dof-name -> action index; mind
the DOF-ordering caveat in :mod:`eden.entities.rigid` when wiring actions to joints.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

import genesis as gs
import torch

from eden.managers.base import ManagerBase, ManagerTermBase
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.entities.base import Entity
    from eden.envs.base import EnvBase


class ActionTerm(ManagerTermBase, ConfigurableMixin[ActionTermOptions]):
    """Base class for action terms."""

    entity_name: str = ""
    dofs_name: str | list[str] = []

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

        self._entity = None
        self.dofs_name_map = {}
        self.dofs_idx_map = {}
        self.dofs_idx_local = None
        self._n_dofs = 0
        self._raw_action = None
        self._prev_action = None
        self._processed_action = None

    def build(self) -> None:
        self._entity = self._env.entities[self.entity_name]
        self.dofs_name, dofs_idx_local = self.entity.find_named_dofs_idx_local(
            self.dofs_name, name_scope=self.entity.dofs_name, preserve_order=True
        )
        # mapping from dofs global index to its local index.
        self.dofs_name_map = {name: i for i, name in enumerate(self.dofs_name)}
        self.dofs_idx_map = {idx: i for i, idx in enumerate(dofs_idx_local)}
        self.dofs_idx_local = torch.as_tensor(dofs_idx_local, dtype=gs.tc_int, device=self.device).contiguous()
        self._n_dofs = len(self.dofs_idx_local)
        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_action = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def action(self) -> torch.Tensor:
        return self._raw_action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    @property
    def dofs_order(self) -> dict[str, int]:
        """Get the mapping from dofs name to its local index."""
        return {dof_name: i for i, dof_name in enumerate(self.dofs_name)}

    @property
    def action_dim(self) -> int:
        return self._n_dofs

    @property
    def num_dofs(self) -> int:
        return self._n_dofs

    @abstractmethod
    def apply_actions(self) -> None:
        raise NotImplementedError


ACTION_TERM_REGISTRY = Registry("ACTION_TERM")


class ActionManager(ManagerBase[ActionManagerOptions]):
    """Action manager for processing actions sent to the environment."""

    def __init__(self, env: EnvBase, options: ActionManagerOptions):
        super().__init__(env=env, options=options)

        # Create action slices after terms are prepared
        self._term_slice = {}
        idx = 0
        for term_name, term in self._terms.items():
            self._term_slice[term_name] = slice(idx, idx + term.action_dim)
            idx += term.action_dim

        # Create buffers to store actions.
        self._action = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_action = torch.zeros_like(self._action)

    def summary(self) -> str:
        return self._format_summary_table(
            title=f"Active Action Terms (shape: {self.total_action_dim})",
            field_names=["Index", "Name", "Dimension", "Slice"],
            rows=(
                [index, name, term.action_dim, self._term_slice[name]]
                for index, (name, term) in enumerate(self._terms.items())
            ),
            align={"Name": "l", "Dimension": "r", "Slice": "r"},
        )

    @cached_property
    def total_action_dim(self) -> int:
        return sum(self.action_term_dim)

    @cached_property
    def action_term_dim(self) -> list[int]:
        """The order corresponds to the .active_terms property."""
        return [term.action_dim for term in self._terms.values()]

    @cached_property
    def total_dofs_dim(self) -> int:
        return sum(self.dofs_term_dim)

    @cached_property
    def dofs_term_dim(self) -> list[int]:
        """The order corresponds to the .active_terms property."""
        return [term.num_dofs for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        return self._prev_action

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> dict[str, float]:
        if envs_idx is None:
            envs_idx = slice(None)

        # Reset action history.
        # NOTE: use assignment here (it works for both slice and tensor index), not .zero_()
        self._prev_action[envs_idx] = 0.0
        self._action[envs_idx] = 0.0

        # Reset action terms.
        for term in self._terms.values():
            term.reset(envs_idx=envs_idx)
        return {}

    def compute(self, action: torch.Tensor | dict[str, torch.Tensor]) -> None:
        if isinstance(action, dict):
            for term_name, term_action in action.items():
                self._terms[term_name].compute(term_action)
                self._prev_action[:, self._term_slice[term_name]] = self._action[:, self._term_slice[term_name]]
                self._action[:, self._term_slice[term_name]] = term_action
        else:
            if self.total_action_dim != action.shape[1]:
                raise ValueError(
                    f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}."
                )
            self._prev_action[:] = self._action
            self._action[:] = action
            for term_name, term in self._terms.items():
                term_actions = self._action[:, self._term_slice[term_name]]
                term.compute(term_actions)

    def apply_actions(self) -> None:
        for term in self._terms.values():
            term.apply_actions()

    def _prepare_terms(self):
        self._terms: dict[str, ActionTerm] = dict()

        for term_name in self._options.term_keys():
            term: ActionTerm = self._build_term(term_name, ACTION_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
