"""Observation manager for computing observations."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from types import FunctionType
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
import torch
from prettytable import PrettyTable

import eden as en
from eden.managers.base import ManagerBase, ManagerTermBase, ManagerTermFuncWrapperBase
from eden.managers.modifiers.base import NOISE_MODEL_REGISTRY
from eden.options.managers.observations import (
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.utils.buffers.circular_buffer import CircularBuffer
from eden.utils.common import ConfigurableFuncWrapperMixin, ConfigurableMixin
from eden.utils.misc import sanitize_envs_idx
from eden.utils.registry import TermRegistry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def _init_obs_term_shared(term: ObservationTerm | ObservationTermFuncWrapper, env: EnvBase) -> None:
    if term._options.history_length == 1:
        en.logger.warning(
            f"History length is 1 for {term.__class__.__name__}, is this intended? (use history length 0 instead)"
        )
    options = term._options
    if options.noise is not None:
        term._noise = NOISE_MODEL_REGISTRY.get(options.noise.name)(env=env, options=options.noise)
    else:
        term._noise = None


class ObservationTerm(ManagerTermBase, ConfigurableMixin[ObservationTermOptions]):
    """Base class for observation terms.

    The post-compute modifiers (noise/clip/scale/history) are options-only fields (see
    ``ObservationTermOptions.POST_COMPUTE_FIELDS``): ``configure()`` accepts them and stores them on the
    options, but they are NOT mirrored as term attributes — the manager reads them from ``self._options`` so a
    ``compute()`` can't accidentally read (and be wrongly deduped on) a modifier value.
    """

    _extra_option_params = ObservationTermOptions.POST_COMPUTE_FIELDS

    def __init__(self, env: EnvBase, options: ObservationTermOptions):
        ManagerTermBase.__init__(self, env=env)
        ConfigurableMixin.__init__(self, options=options)
        _init_obs_term_shared(self, env)


class ObservationTermFuncWrapper(ManagerTermFuncWrapperBase, ConfigurableFuncWrapperMixin[ObservationTermOptions]):
    """Base class for observation terms defined as function. See :class:`ObservationTerm` for the modifier
    fields (options-only, not mirrored as attributes).
    """

    _extra_option_params = ObservationTermOptions.POST_COMPUTE_FIELDS

    def __init__(self, func: FunctionType, env: EnvBase, options: ObservationTermOptions):
        ManagerTermFuncWrapperBase.__init__(self, func=func, env=env)
        ConfigurableFuncWrapperMixin.__init__(self, options=options)
        _init_obs_term_shared(self, env)


ObservationTermLike: TypeAlias = ObservationTerm | ObservationTermFuncWrapper
"""Union of a class-based observation term and its function-wrapper form."""


OBSERVATION_TERM_REGISTRY = TermRegistry("OBSERVATION_TERM", ObservationTerm, ObservationTermFuncWrapper)


class ObservationManager(ManagerBase[ObservationManagerOptions]):
    def __init__(self, env: EnvBase, options: ObservationManagerOptions):
        super().__init__(env=env, options=options)

        self._group_obs_dim: dict[str, tuple[int, ...] | list[tuple[int, ...]]] = dict()
        self._obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] | None = None

        for group_name, group_term_dims in self._group_obs_term_dim.items():
            if self._group_obs_options[group_name].concatenate_terms:
                try:
                    term_dims = torch.stack(
                        [torch.tensor(dims, device="cpu") for dims in group_term_dims],
                        dim=0,
                    )
                    if len(term_dims.shape) > 1:
                        dim = self._group_obs_options[group_name].concatenate_dim
                        dim_sum = torch.sum(term_dims[:, dim], dim=0)
                        term_dims[0, dim] = dim_sum
                        term_dims = term_dims[0]
                    else:
                        term_dims = torch.sum(term_dims, dim=0)
                    self._group_obs_dim[group_name] = tuple(term_dims.tolist())
                except RuntimeError:
                    raise RuntimeError(f"Unable to concatenate observation terms in group `{group_name}`.") from None
            else:
                self._group_obs_dim[group_name] = group_term_dims

        # Cross-group raw-output cache. Many tasks reuse the same observation term in multiple groups (e.g. an
        # identical `policy` and `critic` group that differ only in noise/corruption); the raw `term.compute()` is then
        # computed once per group even though it's identical. Terms whose raw output is provably identical (same
        # registered term + same compute params, ignoring post-compute noise/clip/scale/history) compute once per step
        # and the result is reused. Safe for any term whose compute() is a pure function of env state (all built-in
        # terms qualify).
        self._raw_cache: dict = {}
        # Reused output buffers for the general/history concat paths (keyed by group; see _cat_into).
        self._group_cat_out: dict[str, torch.Tensor] = {}
        self._build_dup_signatures()
        self._assign_concat_caches()

    def _assign_concat_caches(self) -> None:
        """Hand each fast-path term a view into its group's concat buffer as its output cache.

        Pre-slices each concat buffer once (the views are reused every step) and assigns the view to
        ``term._cache`` for terms eligible to write in place: those in a fast-path group that are **not**
        cross-group duplicates. A duplicated term's raw output is shared across groups via the cross-group
        cache, so it must not be written into a group-specific buffer — those keep ``_cache = None`` and go
        through the dedup path. See :meth:`compute_group` and ManagerTermBase for the protocol.
        """
        self._group_concat_slot_views: dict[str, list[torch.Tensor]] = {}
        for group_name, concat_buf in self._group_concat_buffer.items():
            slices = self._group_concat_slices[group_name]
            views = [concat_buf[:, start:end] for (start, end) in slices]
            self._group_concat_slot_views[group_name] = views
            for term, view in zip(self._group_obs_terms[group_name], views, strict=True):
                if id(term) not in self._dup_sig:
                    term._cache = view

    @staticmethod
    def _freeze(value):
        """Recursively convert ``value`` into a hashable form for signature keys."""
        if isinstance(value, dict):
            return tuple(sorted((k, ObservationManager._freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(ObservationManager._freeze(v) for v in value)
        return value

    @staticmethod
    def _raw_term_signature(term: ObservationTermLike):
        """Signature identifying terms with identical raw ``compute()`` output.

        Built from the term class plus every config field that feeds ``compute()`` — declared fields and dynamic extras
        alike (registered ``name``, ``func``/``params`` for function terms, custom config fields for class terms) —
        minus the post-compute modifiers in :attr:`ObservationTermOptions.POST_COMPUTE_FIELDS` (noise/clip/scale/
        history, applied per-group after compute()). Excluding those is what lets a noisy ``policy`` term share with a
        clean ``critic`` term.

        Including everything-but-modifiers by default (rather than extras-only) is deliberate: a field added to
        ``ObservationTermOptions`` later participates in the signature unless explicitly listed as a modifier, so a new
        compute-affecting field can't silently merge distinct terms. Returns ``None`` if the signature isn't hashable
        (term won't cache).
        """
        options = term._options
        keys = set(type(options).model_fields) | set(getattr(options, "__pydantic_extra__", None) or {})
        items = []
        for key in sorted(keys):
            if key in ObservationTermOptions.POST_COMPUTE_FIELDS:
                continue
            if key.startswith("_option_"):  # serialization metadata, not config
                continue
            items.append((key, ObservationManager._freeze(getattr(options, key))))
        sig = (term.__class__.__qualname__, tuple(items))
        try:
            hash(sig)
        except TypeError:
            return None
        return sig

    def _build_dup_signatures(self) -> None:
        """Map ``id(term) -> signature`` for terms whose raw output is duplicated.

        Only terms whose signature appears in more than one place are recorded; unique terms are never cached so they
        pay no lookup/storage overhead.
        """
        sig_by_id: dict[int, object] = {}
        counts: dict[object, int] = {}
        for terms in self._group_obs_terms.values():
            for term in terms:
                sig = self._raw_term_signature(term)
                if sig is None:
                    continue
                sig_by_id[id(term)] = sig
                counts[sig] = counts.get(sig, 0) + 1
        self._dup_sig: dict[int, object] = {term_id: sig for term_id, sig in sig_by_id.items() if counts[sig] > 1}

    def _term_raw(self, term: ObservationTermLike) -> torch.Tensor:
        """Compute a term's raw output, reusing a cached result for duplicates."""
        sig = self._dup_sig.get(id(term))
        if sig is None:
            return term.compute()
        cached = self._raw_cache.get(sig)
        if cached is None:
            cached = term.compute()
            self._raw_cache[sig] = cached
        return cached

    def summary(self) -> str:
        msg = f"<ObservationManager> contains {len(self._group_obs_term_names)} groups.\n"
        for group_name, group_dim in self._group_obs_dim.items():
            table = PrettyTable()
            table.title = f"Active Observation Terms in Group: '{group_name}'"
            if self._group_obs_options[group_name].concatenate_terms:
                table.title += f" (shape: {group_dim})"  # type: ignore
            table.field_names = ["Index", "Name", "Shape"]
            table.align["Name"] = "l"
            obs_terms = zip(
                self._group_obs_term_names[group_name],
                self._group_obs_term_dim[group_name],
                self._group_obs_terms[group_name],
                strict=False,
            )
            for index, (name, dims, term) in enumerate(obs_terms):
                if term._options.history_length > 0 and term._options.flatten_history_dim:
                    # Flattened history: show (9,) ← 3×(3,)
                    original_size = int(np.prod(dims)) // term._options.history_length
                    original_shape = (original_size,) if len(dims) == 1 else dims[1:]
                    shape_str = f"{dims}  ← {term._options.history_length}×{original_shape}"
                else:
                    shape_str = str(tuple(dims))
                table.add_row([index, name, shape_str])
            msg += table.get_string()
            msg += "\n"
        return msg

    @property
    def active_terms(self) -> dict[str, list[str]]:
        return self._group_obs_term_names

    def get_term(self, name: str) -> ObservationTermLike:
        # term_keys() (not keys()) so declared config fields aren't iterated as obs groups.
        for group_name in self._options.term_keys():
            for term_name, term in zip(
                self._group_obs_term_names[group_name],
                self._group_obs_terms[group_name],
                strict=True,
            ):
                if term_name == name:
                    return term
        raise ValueError(f"Term '{name}' not found in observation manager.")

    @property
    def group_obs_dim(self) -> dict[str, tuple[int, ...] | list[tuple[int, ...]]]:
        return self._group_obs_dim

    @property
    def group_obs_term_dim(self) -> dict[str, list[tuple[int, ...]]]:
        return self._group_obs_term_dim

    def reset(self, envs_idx: torch.Tensor | slice | None = None) -> dict[str, float]:
        history_batch_ids = (
            None if envs_idx is None else sanitize_envs_idx(envs_idx, self._env.num_envs, prefer_slice=False)
        )

        for group_name in self._group_obs_term_names:
            for term in self._group_obs_reset_terms.get(group_name, ()):
                term.reset(envs_idx=envs_idx)

            for term in self._group_obs_terms[group_name]:
                if term._noise is not None:
                    term._noise.reset(envs_idx=envs_idx)

            history = self._group_obs_term_history_buffer[group_name]
            if isinstance(history, dict):
                for buffer in history.values():
                    buffer.reset(batch_ids=history_batch_ids)
            else:
                # Group-level history: one CircularBuffer for the concatenated group.
                history.reset(batch_ids=history_batch_ids)
        return {}

    def compute(self, update_history: bool = False) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        # Reset the per-step cross-group cache so reused raw outputs reflect the current state. Cleared (not just
        # overwritten) to avoid pinning tensors.
        self._raw_cache.clear()
        obs_buffer: dict[str, torch.Tensor | dict[str, torch.Tensor]] = dict()
        for group_name in self._group_obs_term_names:
            obs_buffer[group_name] = self.compute_group(group_name, update_history)
        self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(self, group_name: str, update_history: bool = False) -> torch.Tensor | dict[str, torch.Tensor]:
        # Fast path: pre-allocated concatenation buffer (no history, 1-D obs, concatenate_terms).
        concat_buf = self._group_concat_buffer.get(group_name)
        if concat_buf is not None:
            slot_views = self._group_concat_slot_views[group_name]
            for i, (term_name, term) in enumerate(
                zip(self._group_obs_term_names[group_name], self._group_obs_terms[group_name], strict=True)
            ):
                slot = slot_views[i]
                if term._cache is slot:
                    # Eligible (unique) term: compute_cached() puts its raw into the slot (zero-copy for
                    # cache-aware terms, one copy otherwise). Noise/clip/scale are then applied in place —
                    # safe because a unique term's raw is not shared across groups.
                    term.compute_cached()
                    if term._noise is not None:
                        slot.copy_(term._noise.compute(slot))
                    if term._options.clip is not None:
                        slot.clamp_(min=term._options.clip[0], max=term._options.clip[1])
                    if term._options.scale is not None and term._options.scale != 1.0:
                        slot.mul_(term._options.scale)
                else:
                    # Cross-group duplicate: take the deduped raw and copy + modify into the slot.
                    obs = self._term_raw(term)
                    if term._noise is not None:
                        obs = term._noise.compute(obs)
                    if term._options.clip is not None:
                        obs = obs.clamp(min=term._options.clip[0], max=term._options.clip[1])
                    if term._options.scale is not None and term._options.scale != 1.0:
                        slot.copy_(obs).mul_(term._options.scale)
                    else:
                        slot.copy_(obs)
            # Return the reused buffer directly (no per-step clone). Per the compute() contract the result is
            # valid only until the next compute(); rsl_rl and _record_final_observations already copy it.
            return concat_buf

        # General path
        group_obs: dict[str, torch.Tensor] = OrderedDict()

        for term_name, term in zip(
            self._group_obs_term_names[group_name],
            self._group_obs_terms[group_name],
            strict=True,
        ):
            obs: torch.Tensor = self._term_raw(term)
            if term._noise is not None:
                obs = term._noise.compute(obs)
            if term._options.clip is not None:
                obs = obs.clamp(min=term._options.clip[0], max=term._options.clip[1])
            if term._options.scale is not None and term._options.scale != 1.0:
                obs = term._options.scale * obs
            if self._group_obs_options[group_name].history_length is not None:
                group_obs[term_name] = obs
            elif term._options.history_length > 0:
                circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
                if update_history or not circular_buffer.is_initialized:
                    circular_buffer.append(obs)
                    buf = circular_buffer.buffer
                else:
                    # Snapshot path: produce the buffer that *would* result from
                    # appending ``obs`` now, without mutating the underlying state.
                    # The canonical once-per-step advance happens elsewhere
                    # (post-reset compute in ``RLEnvBase.step``); doing it here
                    # too would skip a frame for non-done envs, while skipping
                    # it entirely would drop the freshly-computed post-physics
                    # frame from the snapshot.
                    buf = circular_buffer.peek_buffer(obs)

                if term._options.flatten_history_dim:
                    group_obs[term_name] = buf.reshape(self._env.num_envs, -1)
                else:
                    group_obs[term_name] = buf
            else:
                group_obs[term_name] = obs

        if self._group_obs_options[group_name].history_length is not None:
            group_obs = self._cat_into(
                f"{group_name}\x00hist",
                list(group_obs.values()),
                self._group_obs_options[group_name].concatenate_dim,
            )
            circular_buffer = self._group_obs_term_history_buffer[group_name]
            if update_history or not circular_buffer.is_initialized:
                circular_buffer.append(group_obs)
                buf = circular_buffer.buffer
            else:
                # Snapshot path: see term-level branch above.
                buf = circular_buffer.peek_buffer(group_obs)

            if self._group_obs_options[group_name].flatten_history_dim:
                return buf.reshape(self._env.num_envs, -1)
            else:
                return buf

        if self._group_obs_options[group_name].concatenate_terms:
            return self._cat_into(
                group_name, list(group_obs.values()), self._group_obs_options[group_name].concatenate_dim
            )
        return group_obs

    def _cat_into(self, key: str, tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        """Concatenate ``tensors`` into a reused per-key output buffer.

        Avoids a fresh allocation every step: the first call resizes the (initially empty) buffer to the
        result shape; subsequent calls write into the same storage. The returned buffer is valid only until
        the next ``compute()`` (the manager's buffer-reuse contract).
        """
        out = self._group_cat_out.get(key)
        if out is None:
            out = torch.empty(0, device=self._env.device, dtype=tensors[0].dtype)
            self._group_cat_out[key] = out
        return torch.cat(tensors, dim=dim, out=out)

    def _prepare_terms(self) -> None:
        self._group_obs_term_names: dict[str, list[str]] = defaultdict(list)
        self._group_obs_terms: dict[str, list[ObservationTermLike]] = defaultdict(list)
        self._group_obs_reset_terms: dict[str, list[ObservationTerm]] = defaultdict(list)
        self._group_obs_term_dim: dict[str, list[tuple[int, ...]]] = defaultdict(list)
        self._group_obs_options: dict[str, ObservationGroupOptions] = {}

        self._group_obs_term_history_buffer: dict[str, dict[str, CircularBuffer] | CircularBuffer] = dict()

        for group_name in self._options.term_keys():
            group_options: ObservationGroupOptions = getattr(self._options, group_name)
            self._group_obs_options[group_name] = group_options

            # Build term options from extra fields on the group options
            extra = getattr(group_options, "__pydantic_extra__", None) or {}
            terms_data: dict[str, ObservationTermOptions] = {}
            for key, value in extra.items():
                if key in group_options.__class__.model_fields:
                    continue
                if key == "terms" or key.startswith("_option_"):
                    continue
                if not isinstance(value, ObservationTermOptions):
                    if isinstance(value, str):
                        # Skip non-term extra values (e.g., option metadata)
                        continue
                    term_options = ObservationTermOptions(**value)
                else:
                    term_options = value

                if not group_options.enable_corruption and term_options.noise is not None:
                    en.logger.warning(
                        f"Noise is not allowed when `enable_corruption` is False for {key}, disabling noise."
                    )
                    term_options = term_options.model_copy(update={"noise": None})

                if group_options.history_length is not None:
                    en.logger.warning(f"History length is enforced for {key}, updating term options.")
                    term_options = term_options.model_copy(
                        update={
                            "history_length": group_options.history_length,
                            "flatten_history_dim": group_options.flatten_history_dim,
                            "backfill": group_options.backfill,
                        }
                    )

                terms_data[key] = term_options

            if group_options.terms_order is not None:
                for name in group_options.terms_order:
                    if name not in terms_data:
                        raise ValueError(f"Term `{name}` not found in the group!")
                for name in terms_data.keys():
                    if name not in group_options.terms_order:
                        raise ValueError(f"Term {name} not found in the terms order!")
            else:
                # No explicit order given: default to term declaration order.
                group_options = group_options.model_copy(update={"terms_order": list(terms_data.keys())})

            # Store terms on group options for downstream usage
            group_options = group_options.model_copy(update={"terms": terms_data})

            if group_options.history_length is None:
                # NOTE: prepare buffer for each term history separately if history length is not enforced for the group
                group_entry_history_buffer: dict[str, CircularBuffer] = dict()

            for term_name in group_options.terms_order:
                term_options = group_options.terms[term_name]

                term: ObservationTermLike = OBSERVATION_TERM_REGISTRY.get(term_options.name)(
                    env=self._env, options=term_options
                )
                term.build()
                self._group_obs_term_names[group_name].append(term_name)
                self._group_obs_terms[group_name].append(term)
                if isinstance(term, ObservationTerm):
                    self._group_obs_reset_terms[group_name].append(term)

                obs_dims = tuple(term.compute().shape)

                if group_options.history_length is not None:
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term._options.history_length)
                    obs_dims = tuple(old_dims)
                    if term._options.flatten_history_dim:
                        obs_dims = (obs_dims[0], int(np.prod(obs_dims[1:])))
                elif term._options.history_length > 0:
                    group_entry_history_buffer[term_name] = CircularBuffer(
                        max_len=term._options.history_length,
                        batch_size=self._env.num_envs,
                        device=self._env.device,
                        backfill=term._options.backfill,
                    )
                    old_dims = list(obs_dims)
                    old_dims.insert(1, term._options.history_length)
                    obs_dims = tuple(old_dims)
                    if term._options.flatten_history_dim:
                        obs_dims = (obs_dims[0], int(np.prod(obs_dims[1:])))

                self._group_obs_term_dim[group_name].append(obs_dims[1:])

            if group_options.history_length is None:
                self._group_obs_term_history_buffer[group_name] = group_entry_history_buffer
            else:
                self._group_obs_term_history_buffer[group_name] = CircularBuffer(
                    max_len=group_options.history_length,
                    batch_size=self._env.num_envs,
                    device=self._env.device,
                    backfill=group_options.backfill,
                )

        # Pre-allocate concatenation buffers for groups that qualify for the fast path:
        # concatenate_terms=True, no group-level history, all 1-D terms with no per-term history.
        self._group_concat_buffer: dict[str, torch.Tensor] = {}
        self._group_concat_slices: dict[str, list[tuple[int, int]]] = {}
        for group_name, group_options in self._group_obs_options.items():
            if not group_options.concatenate_terms or group_options.history_length is not None:
                continue
            term_dims = self._group_obs_term_dim[group_name]
            has_history = any(t._options.history_length > 0 for t in self._group_obs_terms[group_name])
            if has_history or not all(len(d) == 1 for d in term_dims):
                continue
            total_dim = sum(d[0] for d in term_dims)
            slices: list[tuple[int, int]] = []
            offset = 0
            for d in term_dims:
                slices.append((offset, offset + d[0]))
                offset += d[0]
            self._group_concat_buffer[group_name] = torch.zeros(
                self._env.num_envs,
                total_dim,
                device=self._env.device,
            )
            self._group_concat_slices[group_name] = slices
