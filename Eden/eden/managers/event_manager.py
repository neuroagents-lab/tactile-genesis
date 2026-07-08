"""Event manager for applying randomization and other event-driven changes."""

from __future__ import annotations

from collections import defaultdict
from types import FunctionType
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

from eden.constants import EventMode
from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.misc import sanitize_envs_idx
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


# Define proxies for fast lookup
_STARTUP, _RESET, _INTERVAL = EventMode


class EventTerm(ManagerTermBase, ConfigurableMixin[EventTermOptions]):
    """Base class for event terms."""

    mode: EventMode = _RESET
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    priority: int = 0

    def __init__(self, env: EnvBase, options: EventTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)


class EventTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[EventTermOptions]):
    """Base class for event terms defined as function."""

    mode: EventMode = _RESET
    interval_range_s: tuple[float, float] | None = None
    is_global_time: bool = False
    min_step_count_between_reset: int = 0
    priority: int = 0

    def __init__(self, func: FunctionType, env: EnvBase, options: EventTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)


EVENT_TERM_REGISTRY = TermRegistry("EVENT_TERM", EventTerm, EventTermFuncWrapper)


class EventManager(ManagerBase[EventManagerOptions]):
    def __init__(self, env: EnvBase, options: EventManagerOptions):
        super().__init__(env=env, options=options)

    def summary(self) -> str:
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"
        for mode in self._mode_term_names.keys():
            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"
            if mode == _INTERVAL:
                table.field_names = [
                    "Index",
                    "Name",
                    "Priority",
                    "Interval time range (s)",
                ]
                table.align["Name"] = "l"
                for index, (name, term) in enumerate(
                    zip(
                        self._mode_term_names[mode],
                        self._mode_terms[mode],
                        strict=False,
                    )
                ):
                    table.add_row([index, name, term.priority, term.interval_range_s])
            else:
                table.field_names = ["Index", "Name", "Priority"]
                table.align["Name"] = "l"
                for index, (name, term) in enumerate(
                    zip(
                        self._mode_term_names[mode],
                        self._mode_terms[mode],
                        strict=False,
                    )
                ):
                    table.add_row([index, name, term.priority])
            msg += table.get_string()
            msg += "\n"
        if self._domain_randomization_fields:
            table = PrettyTable()
            table.title = "Domain Randomization Fields"
            table.field_names = ["Index", "Field Name"]
            table.align["Field Name"] = "l"
            for index, field in enumerate(self._domain_randomization_fields):
                table.add_row([index, field])
            msg += table.get_string()
            msg += "\n"
        return msg

    @property
    def active_terms(self) -> dict[EventMode, list[str]]:
        return self._mode_term_names

    @property
    def available_modes(self) -> list[EventMode]:
        return list(self._mode_term_names.keys())

    @property
    def domain_randomization_fields(self) -> list[str]:
        return self._domain_randomization_fields

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None):
        for mode_terms in self._mode_reset_terms.values():
            for term in mode_terms:
                term.reset(envs_idx=envs_idx)
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        if _INTERVAL in self._mode_terms:
            for index, term in enumerate(self._mode_terms[_INTERVAL]):
                if not isinstance(term, EventTerm):
                    continue
                if not term.is_global_time:
                    assert term.interval_range_s is not None
                    lower, upper = term.interval_range_s
                    sampled_interval = (torch.rand(self._env.num_envs, device=self.device) * (upper - lower) + lower)[
                        envs_idx
                    ]
                    self._interval_term_time_left[index][envs_idx] = sampled_interval
        return {}

    def compute(
        self,
        mode: EventMode,
        envs_idx: (slice | torch.Tensor) | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
    ):
        if mode == _INTERVAL:
            if dt is None:
                raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
            if envs_idx is not None:
                raise ValueError(
                    f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                    " as the environment indices are computed based on the time left for each environment."
                )
        elif mode == _RESET:
            if global_env_step_count is None:
                raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")
            envs_idx, n_envs = sanitize_envs_idx(envs_idx, self._env.num_envs, return_n_envs=True)
            if n_envs == 0:
                return

        for index, term in enumerate(self._mode_terms[mode]):
            if mode == _INTERVAL:
                time_left = self._interval_term_time_left[index]
                time_left -= dt
                if term.is_global_time:
                    if time_left < 1e-6:
                        assert term.interval_range_s is not None
                        lower, upper = term.interval_range_s
                        sampled_interval = torch.rand(1, device=self.device) * (upper - lower) + lower
                        self._interval_term_time_left[index][:] = sampled_interval
                        term.compute()
                else:
                    valid_envs_mask = time_left < 1e-6
                    # Bool-mask ops are no-ops for all-False — skip .any() GPU sync
                    assert term.interval_range_s is not None
                    lower, upper = term.interval_range_s
                    sampled_time = torch.rand(self._env.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left[index][valid_envs_mask] = sampled_time[valid_envs_mask]
                    term.compute(envs_idx=valid_envs_mask)
            elif mode == _RESET:
                assert global_env_step_count is not None
                min_step_count = term.min_step_count_between_reset
                if min_step_count == 0:
                    self._reset_term_last_triggered_step_id[index][envs_idx] = global_env_step_count
                    self._reset_term_last_triggered_once[index][envs_idx] = True
                    term.compute(envs_idx=envs_idx)
                else:
                    last_triggered_step = self._reset_term_last_triggered_step_id[index][envs_idx]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][envs_idx]
                    steps_since_triggered = global_env_step_count - last_triggered_step
                    valid_trigger = steps_since_triggered >= min_step_count
                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once

                    # index tensor or slice -> scatter into pre-allocated full bool mask
                    self._valid_envs_buf.zero_()
                    self._valid_envs_buf[envs_idx] = valid_trigger
                    valid_envs_mask = self._valid_envs_buf

                    # Always execute — bool-mask ops are no-ops for all-False
                    self._reset_term_last_triggered_once[index][valid_envs_mask] = True
                    self._reset_term_last_triggered_step_id[index][valid_envs_mask] = global_env_step_count
                    term.compute(envs_idx=valid_envs_mask)
            elif mode == _STARTUP:
                term.compute(envs_idx=envs_idx)
            else:
                raise ValueError(f"Invalid event mode: {mode}")

    def _prepare_terms(self) -> None:
        # Temporary storage for terms with their priorities before sorting
        temp_mode_terms: dict[EventMode, list[tuple[int, EventTerm | EventTermFuncWrapper, str]]] = defaultdict(list)
        temp_mode_reset_terms: dict[EventMode, list[tuple[int, EventTerm, str]]] = defaultdict(list)

        # Temporary buffers with indices
        temp_interval_buffers: list[tuple[int, torch.Tensor, str]] = []
        temp_reset_buffers: list[tuple[int, torch.Tensor, torch.Tensor, str]] = []

        self._domain_randomization_fields: list[str] = list()

        for term_name in self._options.term_keys():
            term_option: EventTermOptions = getattr(self._options, term_name)
            term: EventTerm | EventTermFuncWrapper = self._build_term(term_name, EVENT_TERM_REGISTRY)

            # Store terms with priority for sorting
            temp_mode_terms[term_option.mode].append((term_option.priority, term, term_name))
            if isinstance(term, EventTerm):
                temp_mode_reset_terms[term_option.mode].append((term_option.priority, term, term_name))

            if term_option.mode == _INTERVAL:
                if term_option.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )
                if term_option.is_global_time:
                    lower, upper = term_option.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    temp_interval_buffers.append((term_option.priority, time_left, term_name))
                else:
                    lower, upper = term_option.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    temp_interval_buffers.append((term_option.priority, time_left, term_name))
            elif term_option.mode == _RESET:
                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                temp_reset_buffers.append((term_option.priority, step_count, no_trigger, term_name))

        # Sort terms by priority (higher priority first) and create final structures
        self._mode_term_names: dict[EventMode, list[str]] = defaultdict(list)
        self._mode_terms: dict[EventMode, list[EventTerm | EventTermFuncWrapper]] = defaultdict(list)
        self._mode_reset_terms: dict[EventMode, list[EventTerm]] = defaultdict(list)

        self._interval_term_time_left: list[torch.Tensor] = []
        self._reset_term_last_triggered_step_id: list[torch.Tensor] = []
        self._reset_term_last_triggered_once: list[torch.Tensor] = []

        for mode in temp_mode_terms:
            # Sort by priority ascending (lower numbers = higher priority = runs first)
            sorted_terms = sorted(temp_mode_terms[mode], key=lambda x: x[0])
            for priority, term, name in sorted_terms:
                self._mode_term_names[mode].append(name)
                self._mode_terms[mode].append(term)

        for mode in temp_mode_reset_terms:
            # Sort by priority ascending (lower numbers = higher priority = runs first)
            sorted_reset_terms = sorted(temp_mode_reset_terms[mode], key=lambda x: x[0])
            for priority, term, name in sorted_reset_terms:
                self._mode_reset_terms[mode].append(term)

        # Sort interval buffers by priority ascending (lower numbers = higher priority = runs first)
        sorted_interval_buffers = sorted(temp_interval_buffers, key=lambda x: x[0])
        for priority, buffer, name in sorted_interval_buffers:
            self._interval_term_time_left.append(buffer)

        # Sort reset buffers by priority ascending (lower numbers = higher priority = runs first)
        sorted_reset_buffers = sorted(temp_reset_buffers, key=lambda x: x[0])
        for priority, step_buffer, trigger_buffer, name in sorted_reset_buffers:
            self._reset_term_last_triggered_step_id.append(step_buffer)
            self._reset_term_last_triggered_once.append(trigger_buffer)

        # Pre-allocate buffer for RESET mode index-tensor/slice → bool mask conversion
        self._valid_envs_buf = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device)
