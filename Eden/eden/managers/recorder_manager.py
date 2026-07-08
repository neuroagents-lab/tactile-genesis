"""Recorder manager and RecorderTerm base for episode data capture and export."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from prettytable import PrettyTable

import eden as en
import eden.utils.file_handler  # noqa: F401 — ensure all file handlers are registered
from eden.constants import DatasetExportMode
from eden.managers.base import ManagerBase, ManagerTermBase
from eden.options.managers.recorders import RecorderManagerOptions, RecorderTermOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.file_handler.base import FILE_HANDLER_REGISTRY
from eden.utils.file_handler.episode_data import EpisodeData
from eden.utils.misc import sanitize_envs_idx
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class RecorderTerm(ManagerTermBase, ConfigurableMixin[RecorderTermOptions]):
    def __init__(self, env: EnvBase, options: RecorderTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)

    def compute(self, *args, **kwargs):
        """RecorderTerm does not use compute(); recording is driven by record_* hooks."""
        raise NotImplementedError("RecorderTerm uses record_* methods instead of compute().")

    def record_pre_reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict]:
        """Record data at the beginning of env.reset() before reset is effective."""
        return {}

    def record_post_reset(self, envs_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict]:
        """Record data at the end of env.reset()."""
        return {}

    def record_pre_step(self) -> dict[str, torch.Tensor | dict]:
        """Record data at the beginning of env.step() before all managers are processed."""
        return {}

    def record_post_step(self) -> dict[str, torch.Tensor | dict]:
        """Record data at the end of env.step() after rewards/terminations are computed."""
        return {}

    def close(self, file_path: str):
        """Finalize and "clean up" the recorder term."""
        pass


RECORDER_TERM_REGISTRY = TermRegistry(name="RECORDER_TERM", term_class=RecorderTerm)


class _EpisodeStore:
    """Recorder runtime state for all environments.

    This keeps one current `EpisodeData` per env plus one pending step payload
    per env. Pending step data is committed only when a step/reset boundary is
    explicitly closed by `RecorderManager`.
    """

    def __init__(self, num_envs: int):
        self._episodes = {env_id: self._create_episode(env_id) for env_id in range(num_envs)}
        self._pending: dict[int, dict[str, torch.Tensor]] = {}

    def get_episode(self, env_idx: int) -> EpisodeData:
        if env_idx not in self._episodes:
            raise IndexError(f"Environment index {env_idx} is out of range for recorder episodes.")
        return self._episodes[env_idx]

    def add_step_data(self, key: str, value: torch.Tensor | dict, env_ids: list[int]) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add_step_data(f"{key}/{sub_key}", sub_value, env_ids)
            return

        for value_index, env_id in enumerate(env_ids):
            pending = self._pending.setdefault(env_id, {})
            if key in pending:
                raise RuntimeError(f"Duplicate pending transition key '{key}' for env {env_id}.")
            pending[key] = value[value_index].clone()

    def add_episode_data(self, key: str, value: torch.Tensor | dict, env_ids: list[int]) -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                self.add_episode_data(f"{key}/{sub_key}", sub_value, env_ids)
            return

        for value_index, env_id in enumerate(env_ids):
            self.get_episode(env_id).add(key, value[value_index])

    def commit_steps(self, env_ids: list[int]) -> None:
        for env_id in env_ids:
            pending = self._pending.pop(env_id, None)
            if not pending:
                continue
            episode = self.get_episode(env_id)
            for key, value in pending.items():
                # Pending values were already cloned by `add_step_data`; avoid
                # cloning them a second time on commit.
                episode._append_owned(key, value)

    def assert_no_open_steps(self, env_ids: list[int] | None = None) -> None:
        if env_ids is None:
            env_ids = list(self._pending)
        pending_envs = [env_id for env_id in env_ids if env_id in self._pending]
        if pending_envs:
            raise RuntimeError(f"Pending recorder transitions remain for envs {pending_envs}.")

    def set_episode_success(self, env_ids: list[int], success_values: torch.Tensor) -> None:
        success_list = success_values.tolist() if isinstance(success_values, torch.Tensor) else list(success_values)
        if len(env_ids) != len(success_list):
            raise ValueError(
                f"set_episode_success got {len(env_ids)} env ids but {len(success_list)} success values; "
                "they must align 1:1."
            )
        for env_id, success in zip(env_ids, success_list):
            self.get_episode(env_id).success = success

    def collect_export_episodes(self, env_ids: list[int]) -> list[tuple[int, EpisodeData]]:
        self.assert_no_open_steps(env_ids)
        exportable_episodes = []
        for env_id in env_ids:
            episode = self.get_episode(env_id)
            if not episode.is_empty():
                exportable_episodes.append((env_id, episode.pre_export_copy()))
        return exportable_episodes

    def reset(self, env_ids: list[int]) -> None:
        self.assert_no_open_steps(env_ids)
        for env_id in env_ids:
            self._episodes[env_id] = self._create_episode(env_id)

    @staticmethod
    def _create_episode(env_id: int) -> EpisodeData:
        episode = EpisodeData()
        episode.env_id = env_id
        return episode


class _EpisodeExporter:
    """Owns export policy, file handlers, and per-env export counters."""

    def __init__(self, options: RecorderManagerOptions, env_cfg: dict):
        self._options = options
        self._env_cfg = env_cfg
        self._dataset_file_handler = None
        self._failed_episode_dataset_file_handler = None

        self._exported_successful_episode_count: dict[int, int] = {}
        self._exported_failed_episode_count: dict[int, int] = {}

    @property
    def dataset_file_handler(self):
        return self._dataset_file_handler

    @property
    def failed_episode_dataset_file_handler(self):
        return self._failed_episode_dataset_file_handler

    @property
    def exported_successful_episode_count(self) -> int:
        return sum(self._exported_successful_episode_count.values())

    def get_exported_successful_episode_count(self, env_idx: int) -> int:
        return self._exported_successful_episode_count.get(env_idx, 0)

    @property
    def exported_failed_episode_count(self) -> int:
        return sum(self._exported_failed_episode_count.values())

    def get_exported_failed_episode_count(self, env_idx: int) -> int:
        return self._exported_failed_episode_count.get(env_idx, 0)

    @property
    def demo_count(self) -> int:
        if self._dataset_file_handler is None:
            return 0
        return self._dataset_file_handler.demo_count

    def load_episode(self, demo_name, device):
        if self._options.dataset_export_mode == DatasetExportMode.EXPORT_NONE:
            raise RuntimeError(
                "Cannot load episodes: no dataset file handler is configured"
                f" (dataset_export_mode={self._options.dataset_export_mode!r})."
            )
        # `load_episode` must not have side effects on disk. If no handler is
        # initialized yet (no recording session has started), refuse to silently
        # create an empty dataset on a read API.
        if self._dataset_file_handler is None:
            raise RuntimeError(
                "Cannot load episodes: dataset file handler has not been initialized. "
                "Start a recording session first, or call load_episode after at least one "
                "successful export."
            )
        return self._dataset_file_handler.load_episode(demo_name, device=device)

    def prepare_for_recording(self) -> None:
        """Validate export dependencies and initialize handlers before recording starts."""
        mode = self._options.dataset_export_mode
        if mode == DatasetExportMode.EXPORT_NONE:
            return
        self._ensure_dataset_file_handler()
        if mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            self._ensure_failed_episode_dataset_file_handler()

    def export_episodes(self, episodes: list[tuple[int, EpisodeData]], env_ids: list[int]) -> None:
        if not episodes:
            return

        need_to_flush = False

        for env_id, save_episode in episodes:
            episode_succeeded = save_episode.success
            target = self._get_or_create_target_handler(episode_succeeded)
            # Only count episodes that were actually written; e.g. EXPORT_SUCCEEDED_ONLY
            # silently drops failed episodes, so they should not bump the failed counter.
            if target is None:
                continue
            target.write_episode(save_episode)
            need_to_flush = True

            if episode_succeeded:
                self._exported_successful_episode_count[env_id] = (
                    self._exported_successful_episode_count.get(env_id, 0) + 1
                )
            else:
                self._exported_failed_episode_count[env_id] = self._exported_failed_episode_count.get(env_id, 0) + 1

        if need_to_flush:
            self.flush()

    def flush(self) -> None:
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.flush()
        if self._failed_episode_dataset_file_handler is not None:
            self._failed_episode_dataset_file_handler.flush()

    def close(self) -> None:
        if self._dataset_file_handler is not None:
            self._dataset_file_handler.close()
        if self._failed_episode_dataset_file_handler is not None:
            self._failed_episode_dataset_file_handler.close()

    def _get_or_create_target_handler(self, episode_succeeded: bool):
        mode = self._options.dataset_export_mode
        if mode == DatasetExportMode.EXPORT_NONE:
            return None
        if mode == DatasetExportMode.EXPORT_ALL:
            self._ensure_dataset_file_handler()
            return self._dataset_file_handler
        if mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY:
            if not episode_succeeded:
                return None
            self._ensure_dataset_file_handler()
            return self._dataset_file_handler
        if mode == DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES:
            if episode_succeeded:
                self._ensure_dataset_file_handler()
                return self._dataset_file_handler
            self._ensure_failed_episode_dataset_file_handler()
            return self._failed_episode_dataset_file_handler
        return None

    def _ensure_dataset_file_handler(self) -> None:
        if self._dataset_file_handler is None:
            self._dataset_file_handler = self._create_file_handler(self._options.dataset_filename, self._env_cfg)

    def _ensure_failed_episode_dataset_file_handler(self) -> None:
        if self._failed_episode_dataset_file_handler is None:
            self._failed_episode_dataset_file_handler = self._create_file_handler(
                f"{self._options.dataset_filename}_failed", self._env_cfg
            )

    def _create_file_handler(self, filename: str, env_cfg: dict):
        handler = FILE_HANDLER_REGISTRY.get(self._options.file_handler_options.name)()
        file_path = os.path.join(self._options.dataset_export_dir_path, filename)
        resolved_path = handler.resolve_path(file_path)
        exists_before = os.path.exists(resolved_path)
        if exists_before and not self._options.override:
            handler.open(file_path, mode="a", env_cfg=env_cfg)
            en.logger.info(f"Resuming recorder dataset at '{resolved_path}'.")
        else:
            handler.create(file_path, env_cfg=env_cfg)
            if exists_before and self._options.override:
                en.logger.info(f"Overriding recorder dataset at '{resolved_path}'.")
            else:
                en.logger.info(f"Creating recorder dataset at '{resolved_path}'.")
        return handler


class RecorderManager(ManagerBase[RecorderManagerOptions]):
    """Lifecycle orchestrator for transition-first recording.

    Public methods expose recording/export control to users. Internal `on_*`
    methods are called from env lifecycle events and translate those events into
    recorder-term collection, step closure, episode rotation, and export.
    """

    def __init__(self, env: EnvBase, options: RecorderManagerOptions):
        super().__init__(env=env, options=options)

        self._recording = False
        self._all_env_ids = list(range(env.num_envs))

        self._episode_store = _EpisodeStore(env.num_envs)
        self._exporter = _EpisodeExporter(
            options,
            self._env.config._to_dict(),
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def exported_successful_episode_count(self) -> int:
        return self._exporter.exported_successful_episode_count

    @property
    def exported_failed_episode_count(self) -> int:
        return self._exporter.exported_failed_episode_count

    @property
    def demo_count(self) -> int:
        return self._exporter.demo_count

    @property
    def dataset_file_handler(self):
        return self._exporter.dataset_file_handler

    @property
    def failed_episode_dataset_file_handler(self):
        return self._exporter.failed_episode_dataset_file_handler

    def summary(self) -> str:
        if not self.active_terms:
            return ""

        msg = f"<RecorderManager> contains {len(self._term_names)} active terms.\n"
        table = PrettyTable()
        table.title = "Active Recorder Terms"
        table.field_names = ["Index", "Name"]
        table.align["Name"] = "l"
        for index, name in enumerate(self._term_names):
            table.add_row([index, name])
        msg += table.get_string()
        msg += "\n"
        return msg

    def start_recording(self, reset_scene: bool = False) -> None:
        if not self.active_terms:
            en.logger.warning("RecorderManager.start_recording() ignored because there are no active recorder terms.")
            return
        if self._recording:
            en.logger.warning("RecorderManager.start_recording() ignored because recording is already active.")
            return

        self._exporter.prepare_for_recording()
        self._recording = True
        if reset_scene:
            self._env.reset()
        else:
            self._collect_post_reset_data(envs_idx=None)
        en.logger.info(f"Started recording. Current saved demos: {self.demo_count}.")

    def end_recording(self) -> None:
        if not self.active_terms:
            return

        demos_before = self.demo_count
        self._recording = False
        self._export_closed_episodes(self._all_env_ids)
        self._start_new_episodes(self._all_env_ids)
        saved_demos = self.demo_count - demos_before
        en.logger.info(f"Stopped recording. Saved {saved_demos} demo(s). Total saved demos: {self.demo_count}.")

    def get_exported_successful_episode_count(self, env_idx: int) -> int:
        return self._exporter.get_exported_successful_episode_count(env_idx)

    def get_exported_failed_episode_count(self, env_idx: int) -> int:
        return self._exporter.get_exported_failed_episode_count(env_idx)

    def get_episode(self, env_idx: int) -> EpisodeData:
        return self._episode_store.get_episode(env_idx)

    # TODO: Move `load_episode` to a dedicated dataset reader API.
    def load_episode(self, demo_name, device=None):
        if not self.active_terms:
            raise RuntimeError("Cannot load episodes: no active recorder terms.")
        if device is None:
            device = self._env.device
        return self._exporter.load_episode(demo_name, device=device)

    def set_episode_success(self, envs_idx: torch.Tensor | None, success_values: torch.Tensor):
        if not self.active_terms:
            return
        env_ids = self._get_env_ids(envs_idx)
        self._episode_store.set_episode_success(env_ids, success_values)

    def close(self):
        if not self.active_terms:
            return
        if self._recording:
            self.end_recording()
        # Terms must close before the exporter so they can still interact with open file handles.
        file_path = os.path.join(self._options.dataset_export_dir_path, self._options.dataset_filename)
        for term in self._terms.values():
            term.close(file_path)
        self._exporter.close()

    def _on_step_started(self) -> None:
        """Start collecting data for a new env step."""
        if not self.active_terms or not self._recording:
            return
        self._collect_pre_step_data()

    def _on_step_finished(
        self,
        terminal_envs_idx: torch.Tensor | None = None,
    ) -> None:
        """Close the current env step and rotate any terminal episodes.

        `terminal_envs_idx` means "envs whose current step ends their episode",
        not "perform the actual env reset here". The real reset still happens in
        `EnvBase` after the recorder has closed the old episode.
        """
        if not self.active_terms or not self._recording:
            return

        terminal_env_ids = [] if terminal_envs_idx is None else self._get_env_ids(terminal_envs_idx)
        self._collect_post_step_data()
        self._collect_terminal_step_data(terminal_env_ids)
        self._close_open_steps()
        self._export_closed_episodes(terminal_env_ids)
        self._start_new_episodes(terminal_env_ids)

    def _on_reset_started(self, envs_idx: torch.Tensor | None) -> None:
        """Close episodes before a manual env reset that is not tied to step end."""
        if not self.active_terms or not self._recording:
            return

        reset_env_ids = self._get_env_ids(envs_idx)
        self._collect_terminal_step_data(reset_env_ids)
        self._close_open_steps()
        self._export_closed_episodes(reset_env_ids)
        self._start_new_episodes(reset_env_ids)

    def _on_reset_finished(self, envs_idx: torch.Tensor | None) -> None:
        """Record new-episode data after the env has completed its reset."""
        if not self.active_terms or not self._recording:
            return
        self._collect_post_reset_data(envs_idx)

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def _collect_pre_step_data(self) -> None:
        """Collect term payloads that belong to the start of the current step."""
        for term in self._terms.values():
            for key, value in term.record_pre_step().items():
                self._episode_store.add_step_data(key, value, self._all_env_ids)

    def _collect_terminal_step_data(self, terminal_env_ids: list[int]) -> None:
        """Collect term payloads that belong to the last step of closing episodes."""
        if not terminal_env_ids:
            return

        envs_idx = self._to_env_idx_tensor(terminal_env_ids)
        for term in self._terms.values():
            for key, value in term.record_pre_reset(envs_idx=envs_idx).items():
                self._episode_store.add_step_data(key, value, terminal_env_ids)

    def _collect_post_reset_data(self, envs_idx: torch.Tensor | None = None) -> None:
        """Collect episode-level payloads after new episodes have begun."""
        env_ids = self._all_env_ids if envs_idx is None else self._get_env_ids(envs_idx)
        # Hot-path guard: in RL training _on_reset_finished fires every step with the
        # `reset_buf` mask, which is all-False most steps. Skip the term loop so we
        # don't query entity state on empty env-index tensors for every term.
        if not env_ids:
            return
        payload = {} if envs_idx is None else {"envs_idx": self._to_env_idx_tensor(env_ids)}
        for term in self._terms.values():
            for key, value in term.record_post_reset(**payload).items():
                self._episode_store.add_episode_data(key, value, env_ids)

    def _close_open_steps(self) -> None:
        """Commit all pending step payloads and ensure nothing remains half-open."""
        self._episode_store.commit_steps(self._all_env_ids)
        self._episode_store.assert_no_open_steps(self._all_env_ids)

    def _export_closed_episodes(self, env_ids: list[int]) -> None:
        """Export the specified closed episodes."""
        if not env_ids:
            return

        exportable_episodes = self._episode_store.collect_export_episodes(env_ids)
        self._exporter.export_episodes(exportable_episodes, env_ids)

    def _start_new_episodes(self, env_ids: list[int]) -> None:
        """Rotate recorder-side state to fresh episodes for the selected envs."""
        if not env_ids:
            return

        envs_idx = self._to_env_idx_tensor(env_ids)
        for term in self._terms.values():
            term.reset(envs_idx=envs_idx)
        self._episode_store.reset(env_ids)

    def _collect_post_step_data(self) -> None:
        """Collect term payloads emitted after rewards/terminations are computed."""
        for term in self._terms.values():
            for key, value in term.record_post_step().items():
                self._episode_store.add_step_data(key, value, self._all_env_ids)

    # NOTE: this helper accepts the full env-index surface used by public/reset-facing
    # APIs (`None`, slice, bool tensor, index tensor) and normalizes it to Python env ids.
    def _get_env_ids(self, envs_idx) -> list[int]:
        envs_idx = sanitize_envs_idx(envs_idx, self._env.num_envs)
        if isinstance(envs_idx, slice):
            return self._all_env_ids[envs_idx]
        if isinstance(envs_idx, torch.Tensor):
            if envs_idx.dtype == torch.bool:
                # Short-circuit the all-False common case to save a .nonzero() + .tolist() sync.
                if not bool(envs_idx.any()):
                    return []
                envs_idx = envs_idx.nonzero(as_tuple=False).flatten()
            return envs_idx.tolist()
        raise ValueError(f"Invalid environment indices: {envs_idx}")

    def _to_env_idx_tensor(self, env_ids: list[int]) -> torch.Tensor:
        return torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long)

    def _prepare_terms(self):
        self._terms: dict[str, RecorderTerm] = {}

        for term_name in self._options.term_keys():
            term = self._build_term(term_name, RECORDER_TERM_REGISTRY)
            self._term_names.append(term_name)
            self._terms[term_name] = term
