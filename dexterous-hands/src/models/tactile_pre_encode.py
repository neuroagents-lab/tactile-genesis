"""Pre-encode models that dispatch each observation group through an encoder of a
chosen *kind* (``mlp``, ``tactile_cnn``, ``tactile_convrnn``) before the policy
MLP/RNN head. Mirrors the structure of
:class:`rsl_rl.models.pre_encode_model.PreEncodeMLPModel` / ``PreEncodeRecurrentModel``
and reuses :class:`PreEncodeMixin`'s pass-through / concat semantics.

State on the wire (for storage round-trip):
    - Non-recurrent head + recurrent encoder -> ``get_hidden_state()`` returns
      the encoder's 3D tensor.
    - Recurrent head (LSTM/GRU) + recurrent encoder -> returns a tuple whose
      last element is the encoder state; preceding elements are the policy
      RNN's hidden state(s). ``set_hidden_state`` splits by this convention.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.pre_encode_model import PreEncodeMixin, _last_linear_out_features
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict

from .tactile_encoders import (
    CanvasTactileLayout,
    GroupRNNEncoder,
    TactileCanvasCNNEncoder,
    TactileCanvasConvRNNEncoder,
    TactileCNNEncoder,
    TactileConvRNNEncoder,
    TactileLayout,
)


# ---- encoder construction ----------------------------------------------


def _build_encoder(
    kind: str,
    group: str,
    cfg: dict[str, Any],
    in_dim: int,
    default_activation: str,
) -> nn.Module:
    cfg = dict(cfg)
    if kind == "mlp":
        out_dim = cfg.pop("output_dim")
        hidden = cfg.pop("hidden_dims")
        act = cfg.pop("activation", default_activation)
        if cfg:
            raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
        return MLP(in_dim, out_dim, hidden, act)

    if kind in ("tactile_cnn", "tactile_convrnn"):
        layout = TactileLayout.from_dict(cfg.pop("tactile_layout"))
        if layout.flat_dim != in_dim:
            raise ValueError(
                f"encoder_cfg['{group}']: tactile_layout flat_dim {layout.flat_dim} "
                f"does not match observed feature dim {in_dim} for '{group}'."
            )
        output_dim = cfg.pop("output_dim")
        shared = cfg.pop("shared_per_sensor", False)
        if kind == "tactile_cnn":
            cnn_cfg = cfg.pop("cnn")
            if cfg:
                raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
            return TactileCNNEncoder(layout, cnn_cfg, output_dim, shared_per_sensor=shared)
        convrnn_cfg = cfg.pop("convrnn")
        if cfg:
            raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
        return TactileConvRNNEncoder(layout, convrnn_cfg, output_dim, shared_per_sensor=shared)

    if kind in ("tactile_canvas_cnn", "tactile_canvas_convrnn"):
        layout = CanvasTactileLayout.from_dict(cfg.pop("tactile_layout"))
        if layout.flat_dim != in_dim:
            raise ValueError(
                f"encoder_cfg['{group}']: tactile_layout flat_dim {layout.flat_dim} "
                f"does not match observed feature dim {in_dim} for '{group}'."
            )
        output_dim = cfg.pop("output_dim")
        if kind == "tactile_canvas_cnn":
            cnn_cfg = cfg.pop("cnn")
            if cfg:
                raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
            return TactileCanvasCNNEncoder(layout, cnn_cfg, output_dim)
        convrnn_cfg = cfg.pop("convrnn")
        if cfg:
            raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
        return TactileCanvasConvRNNEncoder(layout, convrnn_cfg, output_dim)

    if kind == "rnn":
        output_dim = cfg.pop("output_dim")
        hidden_dim = cfg.pop("hidden_dim")
        num_layers = cfg.pop("num_layers", 1)
        rnn_type = cfg.pop("rnn_type", "lstm")
        if cfg:
            raise ValueError(f"Unknown keys in encoder_cfg['{group}']: {list(cfg.keys())}")
        return GroupRNNEncoder(
            input_dim=in_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            num_layers=num_layers, rnn_type=rnn_type,
        )

    raise ValueError(f"encoder_cfg['{group}']: unknown kind {kind!r}")


def _encoder_output_dim(encoder: nn.Module, group: str) -> int:
    if isinstance(encoder, (
        TactileCNNEncoder,
        TactileConvRNNEncoder,
        TactileCanvasCNNEncoder,
        TactileCanvasConvRNNEncoder,
        GroupRNNEncoder,
    )):
        return encoder.output_dim
    return _last_linear_out_features(cast(nn.Sequential, encoder), group)


def _encoder_recurrent(encoder: nn.Module) -> bool:
    return bool(getattr(encoder, "is_recurrent", False))


# ---- mixin override: build encoders by kind ----------------------------


class _TactilePreEncodeBuild(PreEncodeMixin):
    """PreEncode prep that dispatches each group on a ``kind`` field."""

    def _pre_encode_prepare(  # type: ignore[override]
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        activation: str,
        encoder_cfg: dict[str, Any] | None,
        encoders: nn.ModuleDict | dict[str, nn.Module] | None,
    ) -> nn.ModuleDict:
        if encoders is None and encoder_cfg is None:
            raise ValueError("Provide encoder_cfg or encoders.")

        # Persist staging for the parent split logic.
        if encoders is not None:
            encoders_norm = encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
            object.__setattr__(self, "_encoder_cfg", {})
            object.__setattr__(self, "_encoders_arg", encoders_norm)
        else:
            object.__setattr__(self, "_encoder_cfg", dict(encoder_cfg or {}))
            object.__setattr__(self, "_encoders_arg", None)

        # Compute obs split (which groups encode vs pass through) + input dims.
        self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

        # Build encoders (per kind) if not shared.
        if self._encoders_arg is not None:
            encoders_mod = self._encoders_arg
        else:
            spec = self._encoder_cfg
            assert spec is not None
            built: dict[str, nn.Module] = {}
            for g in self.obs_groups_encode:
                gcfg = dict(spec[g])
                kind = gcfg.pop("kind", "mlp")
                built[g] = _build_encoder(kind, g, gcfg, self.encode_input_dims[g], activation)
            encoders_mod = nn.ModuleDict(built)

        # Record per-group encoder kind for later (forward, hidden-state routing).
        self._recurrent_encoder_groups: list[str] = [
            g for g in self.obs_groups_encode if _encoder_recurrent(encoders_mod[g])
        ]

        # encode_latent_dim sums each encoder's last-Linear out_features (works for tactile too).
        latent_dim = 0
        for g in self.obs_groups_encode:
            latent_dim += _encoder_output_dim(encoders_mod[g], g)
        self.encode_latent_dim = latent_dim

        return encoders_mod

    def _pre_encode_get_obs_dim(  # type: ignore[override]
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        """Same as parent but allows tactile groups whose input is 2D (B, flat_dim) or
        padded 3D (T, B, flat_dim) at training time. We measure on the *last* dim only."""
        active_obs_groups = obs_groups[obs_set]
        encoders = self._encoders_arg
        encoder_cfg = self._encoder_cfg

        if encoders is None and encoder_cfg is not None:
            extra_cfg = set(encoder_cfg.keys()).difference(active_obs_groups)
            if extra_cfg:
                raise ValueError(f"encoder_cfg keys not in obs set: {sorted(extra_cfg)}")
        encode_keys = set(encoders.keys()) if encoders is not None else set(encoder_cfg or ())
        unknown = encode_keys.difference(active_obs_groups)
        if unknown:
            raise ValueError(f"encoder_cfg references groups not in obs_groups[{obs_set!r}]: {sorted(unknown)}")

        self.obs_groups_encode = [g for g in active_obs_groups if g in encode_keys]
        pass_groups = [g for g in active_obs_groups if g not in encode_keys]

        if not self.obs_groups_encode:
            raise ValueError("At least one observation group must be in encoder_cfg.")

        self.encode_input_dims = {g: obs[g].shape[-1] for g in self.obs_groups_encode}
        obs_dim = sum(obs[g].shape[-1] for g in pass_groups)
        self.obs_groups = pass_groups
        return pass_groups, obs_dim


# ---- shared model utilities --------------------------------------------


def _bundle_to_parts(hidden_state: HiddenState) -> list[torch.Tensor]:
    """Normalize a HiddenState bundle (None/Tensor/tuple) into a flat list of tensors."""
    if hidden_state is None:
        return []
    if isinstance(hidden_state, torch.Tensor):
        return [hidden_state]
    return list(hidden_state)


def _parts_to_bundle(parts: list[torch.Tensor]) -> HiddenState:
    """Inverse of _bundle_to_parts: unwrap single element so storage treats it as GRU-style."""
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return tuple(parts)


# ---- models -------------------------------------------------------------


class TactilePreEncodeMLPModel(_TactilePreEncodeBuild, MLPModel):
    """Pre-encode MLP head + (CNN | ConvRNN | MLP) per-group encoders.

    If any encoder is recurrent, the model exposes recurrent semantics so PPO
    saves/restores hidden state.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, Any] | None = None,
        encoders: nn.ModuleDict | dict[str, nn.Module] | None = None,
    ) -> None:
        encoders_mod = self._pre_encode_prepare(
            obs, obs_groups, obs_set, activation, encoder_cfg, encoders,
        )
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, distribution_cfg)
        self.encoders = encoders_mod
        self._pre_encode_clear_staging()
        # Mark recurrent if any encoder carries persistent state.
        self.is_recurrent = bool(self._recurrent_encoder_groups)

    # ---- latent ------------------------------------------------------
    def _split_bundle_to_groups(self, hidden_state: HiddenState) -> dict[str, HiddenState]:
        """Slice the saved bundle into per-recurrent-encoder pieces (LSTM = 2 slots, others = 1)."""
        out: dict[str, HiddenState] = {g: None for g in self._recurrent_encoder_groups}
        parts = _bundle_to_parts(hidden_state)
        if not parts:
            return out
        i = 0
        for g in self._recurrent_encoder_groups:
            slots = self.encoders[g].state_slots
            chunk = parts[i:i + slots]
            if not chunk:
                out[g] = None
            elif len(chunk) == 1:
                out[g] = chunk[0]
            else:
                out[g] = tuple(chunk)  # type: ignore
            i += slots
        return out

    def _encoded_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
        per_group_state: dict[str, HiddenState],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for g in self.obs_groups_encode:
            enc = self.encoders[g]
            if _encoder_recurrent(enc):
                parts.append(enc(obs[g], masks=masks, hidden_state=per_group_state.get(g)))
            else:
                parts.append(enc(obs[g]))
        return torch.cat(parts, dim=-1)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        per_group = self._split_bundle_to_groups(hidden_state)
        pass_through = self._pass_through_latent(obs)
        encoded = self._encoded_latent(obs, masks, per_group)
        if masks is not None and self.is_recurrent:
            # Recurrent encoders already unpadded their output; unpad pass-through to match.
            if pass_through.numel() > 0:
                pass_through = unpad_trajectories(pass_through, masks)
        return torch.cat([pass_through, encoded], dim=-1)

    # ---- hidden state routing ---------------------------------------
    def get_hidden_state(self) -> HiddenState:
        if not self.is_recurrent:
            return None
        parts: list[torch.Tensor] = []
        for g in self._recurrent_encoder_groups:
            parts.extend(self.encoders[g].get_state_tuple())
        return _parts_to_bundle(parts)

    def set_hidden_state(self, hidden_state: HiddenState) -> None:
        if not self.is_recurrent:
            return
        parts = _bundle_to_parts(hidden_state)
        i = 0
        for g in self._recurrent_encoder_groups:
            enc = self.encoders[g]
            slots = enc.state_slots
            enc.set_state_tuple(tuple(parts[i:i + slots]))
            i += slots

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if hidden_state is not None:
            self.set_hidden_state(hidden_state)
            return
        for g in self._recurrent_encoder_groups:
            self.encoders[g].reset(dones)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        for g in self._recurrent_encoder_groups:
            self.encoders[g].detach_hidden_state(dones)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        # When recurrent, route saved hidden_state to each encoder's internal buffer
        # AND pass the per-group split into get_latent (used by batch-mode forward).
        if self.is_recurrent and hidden_state is not None:
            self.set_hidden_state(hidden_state)
            hs_for_latent = hidden_state
        else:
            hs_for_latent = None
        latent = self.get_latent(obs, masks, hs_for_latent)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    # ---- bookkeeping -----------------------------------------------
    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        return self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.encode_latent_dim

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("TorchScript export not implemented for TactilePreEncodeMLPModel.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("ONNX export not implemented for TactilePreEncodeMLPModel.")


class TactilePreEncodeRecurrentModel(_TactilePreEncodeBuild, MLPModel):
    """Pre-encode + tactile encoders + top-level LSTM/GRU head.

    Hidden state bundle = ``(policy_rnn_state, encoder_state)``; LSTM
    policy state is itself a ``(h, c)`` tuple, so the bundle becomes
    ``(h, c, encoder_state)`` in that case (storage iterates the tuple).
    """

    is_recurrent = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, Any] | None = None,
        encoders: nn.ModuleDict | dict[str, nn.Module] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ) -> None:
        encoders_mod = self._pre_encode_prepare(
            obs, obs_groups, obs_set, activation, encoder_cfg, encoders,
        )
        self.latent_dim = rnn_hidden_dim
        MLPModel.__init__(
            self, obs, obs_groups, obs_set, output_dim, hidden_dims,
            activation, obs_normalization, distribution_cfg,
        )
        self.encoders = encoders_mod
        self.rnn = RNN(self.obs_dim + self.encode_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self._pre_encode_clear_staging()
        self._rnn_type = rnn_type.lower()
        # LSTM has 2-tensor hidden state, GRU has 1-tensor.
        self._n_policy_slots = 2 if self._rnn_type == "lstm" else 1

    # ---- latent ------------------------------------------------------
    def _split_bundle(self, hidden_state: HiddenState) -> tuple[HiddenState, dict[str, HiddenState]]:
        """Split (policy_slots..., encoder_slots...) into (policy_state, per_group_state_dict)."""
        parts = _bundle_to_parts(hidden_state)
        per_group: dict[str, HiddenState] = {g: None for g in self._recurrent_encoder_groups}
        if not parts:
            return None, per_group
        # First _n_policy_slots tensors belong to the top RNN.
        policy_parts = parts[: self._n_policy_slots]
        if not policy_parts:
            policy_state: HiddenState = None
        elif len(policy_parts) == 1:
            policy_state = policy_parts[0]
        else:
            policy_state = tuple(policy_parts)  # type: ignore
        # Remaining tensors are split across recurrent encoders by their state_slots.
        i = self._n_policy_slots
        for g in self._recurrent_encoder_groups:
            slots = self.encoders[g].state_slots
            chunk = parts[i:i + slots]
            if not chunk:
                per_group[g] = None
            elif len(chunk) == 1:
                per_group[g] = chunk[0]
            else:
                per_group[g] = tuple(chunk)  # type: ignore
            i += slots
        return policy_state, per_group

    def _encoded_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
        per_group_state: dict[str, HiddenState],
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for g in self.obs_groups_encode:
            enc = self.encoders[g]
            if _encoder_recurrent(enc):
                parts.append(enc(obs[g], masks=masks, hidden_state=per_group_state.get(g)))
            else:
                parts.append(enc(obs[g]))
        return torch.cat(parts, dim=-1)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        policy_state, per_group = self._split_bundle(hidden_state)
        pass_through = self._pass_through_latent(obs)
        encoded = self._encoded_latent(obs, masks, per_group)
        encoder_is_recurrent = bool(self._recurrent_encoder_groups)
        if masks is not None and pass_through.numel() > 0 and encoder_is_recurrent:
            # Recurrent encoders already unpadded their output; unpad pass-through to match.
            pass_through = unpad_trajectories(pass_through, masks)
        pre = torch.cat([pass_through, encoded], dim=-1)
        if masks is not None and encoder_is_recurrent:
            # Encoders already unpadded; running the top RNN with masks would unpad again.
            # Feed the already-unpadded sequence directly.
            return self._top_rnn_unpadded(pre, policy_state)
        return self.rnn(pre, masks, policy_state).squeeze(0)

    def _top_rnn_unpadded(self, pre: torch.Tensor, policy_state: HiddenState) -> torch.Tensor:
        """Run the top RNN over an already-unpadded sequence as a single forward pass."""
        out, _ = self.rnn.rnn(pre, policy_state)  # type: ignore[arg-type]
        return out

    # ---- hidden state routing ---------------------------------------
    def get_hidden_state(self) -> HiddenState:
        parts: list[torch.Tensor] = []
        # Top RNN slots first.
        policy = self.rnn.hidden_state
        if policy is not None:
            if isinstance(policy, tuple):
                parts.extend(policy)
            else:
                parts.append(policy)
        # Then each recurrent encoder's slots.
        for g in self._recurrent_encoder_groups:
            parts.extend(self.encoders[g].get_state_tuple())
        return _parts_to_bundle(parts)

    def set_hidden_state(self, hidden_state: HiddenState) -> None:
        policy_state, per_group = self._split_bundle(hidden_state)
        self.rnn.hidden_state = policy_state  # type: ignore[assignment]
        for g in self._recurrent_encoder_groups:
            enc = self.encoders[g]
            chunk = per_group.get(g)
            if chunk is None:
                enc.set_state_tuple(())
            elif isinstance(chunk, torch.Tensor):
                enc.set_state_tuple((chunk,))
            else:
                enc.set_state_tuple(tuple(chunk))

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if hidden_state is not None:
            self.set_hidden_state(hidden_state)
            return
        self.rnn.reset(dones)
        for g in self._recurrent_encoder_groups:
            self.encoders[g].reset(dones)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)
        for g in self._recurrent_encoder_groups:
            self.encoders[g].detach_hidden_state(dones)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if hidden_state is not None:
            self.set_hidden_state(hidden_state)
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        return self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

    def _get_latent_dim(self) -> int:
        return self.latent_dim

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("TorchScript export not implemented for TactilePreEncodeRecurrentModel.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("ONNX export not implemented for TactilePreEncodeRecurrentModel.")
