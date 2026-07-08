# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any, cast

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import MLP, RNN, HiddenState


def _last_linear_out_features(encoder: nn.Sequential, group_name: str) -> int:
    for layer in reversed(encoder):
        if isinstance(layer, nn.Linear):
            return layer.out_features
    raise ValueError(f"Encoder for '{group_name}' must contain at least one Linear layer.")


class PreEncodeMixin:
    """Shared observation splitting, MLP encoders, and pass-through normalization for pre-encode models."""

    _encoder_cfg: dict[str, Any] | None
    _encoders_arg: nn.ModuleDict | None

    obs_groups: list[str]
    obs_groups_encode: list[str]
    encode_input_dims: dict[str, int]
    encode_latent_dim: int
    encoders: nn.ModuleDict
    obs_normalization: bool
    obs_normalizer: nn.Module

    def _pre_encode_prepare(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        activation: str,
        encoder_cfg: dict[str, Any] | None,
        encoders: nn.ModuleDict | dict[str, nn.Module] | None,
    ) -> nn.ModuleDict:
        """Resolve encoder config, observation groups, and build (or take) encoder modules.

        Call before :meth:`MLPModel.__init__`. Clears staging via :meth:`_pre_encode_clear_staging` after
        ``super().__init__`` and assigning ``self.encoders``.
        """
        if encoders is None and encoder_cfg is None:
            raise ValueError("Provide encoder_cfg or encoders.")

        if encoders is not None:
            if not isinstance(encoders, (nn.ModuleDict, dict)):
                raise TypeError("encoders must be a ModuleDict or dict[str, nn.Module].")
            encoders_norm = encoders if isinstance(encoders, nn.ModuleDict) else nn.ModuleDict(encoders)
            object.__setattr__(self, "_encoder_cfg", {})
            object.__setattr__(self, "_encoders_arg", encoders_norm)
            if encoder_cfg is not None:
                print("Sharing pre-encode MLP encoders between models; encoder_cfg of the receiving model is ignored.")
        else:
            if encoder_cfg is None:
                raise ValueError("encoder_cfg must be provided when encoders are not shared.")
            if not encoder_cfg:
                raise ValueError("encoder_cfg must contain at least one observation group.")
            object.__setattr__(self, "_encoder_cfg", dict(encoder_cfg))
            object.__setattr__(self, "_encoders_arg", None)

        self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

        if self._encoders_arg is not None:
            encoders_mod = self._encoders_arg
        else:
            spec = self._encoder_cfg
            assert spec is not None
            built: dict[str, nn.Module] = {}
            for obs_group in self.obs_groups_encode:
                cfg = dict(spec[obs_group])
                out_dim = cfg.pop("output_dim")
                enc_act = cfg.pop("activation", activation)
                enc_hidden = cfg.pop("hidden_dims")
                if cfg:
                    raise ValueError(f"Unknown keys in encoder_cfg['{obs_group}']: {list(cfg.keys())}")
                in_dim = self.encode_input_dims[obs_group]
                built[obs_group] = MLP(in_dim, out_dim, enc_hidden, enc_act)
            encoders_mod = nn.ModuleDict(built)

        encode_latent_dim = 0
        for g in self.obs_groups_encode:
            enc = encoders_mod[g]
            encode_latent_dim += _last_linear_out_features(cast(nn.Sequential, enc), g)
        self.encode_latent_dim = encode_latent_dim

        return encoders_mod

    def _pre_encode_clear_staging(self) -> None:
        object.__setattr__(self, "_encoder_cfg", None)
        object.__setattr__(self, "_encoders_arg", None)

    def _pre_encode_concat_latent(self, obs: TensorDict) -> torch.Tensor:
        """Build the concatenated latent: normalized pass-through features then encoder outputs."""
        latent_pass = self._pass_through_latent(obs)
        latent_enc_parts = [self.encoders[g](obs[g]) for g in self.obs_groups_encode]
        latent_enc = torch.cat(latent_enc_parts, dim=-1)
        return torch.cat([latent_pass, latent_enc], dim=-1)

    def _pass_through_latent(self, obs: TensorDict) -> torch.Tensor:
        if not self.obs_groups:
            return self.obs_normalizer(self._empty_feature(obs))
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def _empty_feature(self, obs: TensorDict) -> torch.Tensor:
        g = self.obs_groups[0] if self.obs_groups else self.obs_groups_encode[0]
        x = obs[g]
        return x.new_zeros((*x.shape[:-1], 0))

    def _pre_encode_get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        encoders = self._encoders_arg
        encoder_cfg = self._encoder_cfg

        if encoders is None and encoder_cfg is not None:
            extra_cfg = set(encoder_cfg.keys()).difference(active_obs_groups)
            if extra_cfg:
                raise ValueError(f"encoder_cfg contains keys not in this observation set: {sorted(extra_cfg)}")

        encode_keys = set(encoders.keys()) if encoders is not None else set(encoder_cfg or ())

        unknown = encode_keys.difference(active_obs_groups)
        if unknown:
            raise ValueError(
                f"encoder_cfg / encoders reference groups not in obs_groups['{obs_set}']: {sorted(unknown)}"
            )

        self.obs_groups_encode = [g for g in active_obs_groups if g in encode_keys]
        obs_groups_pass = [g for g in active_obs_groups if g not in encode_keys]

        if not self.obs_groups_encode:
            raise ValueError("At least one observation group must be listed in encoder_cfg (or shared encoders).")

        self.encode_input_dims = {}
        for g in self.obs_groups_encode:
            t = obs[g]
            if len(t.shape) != 2:
                raise ValueError(
                    f"Pre-encode models only support 1D observations (B, D); got shape {t.shape} for '{g}'."
                )
            self.encode_input_dims[g] = t.shape[-1]

        obs_dim = 0
        for g in obs_groups_pass:
            t = obs[g]
            if len(t.shape) != 2:
                raise ValueError(
                    f"Pre-encode models only support 1D observations (B, D); got shape {t.shape} for '{g}'."
                )
            obs_dim += t.shape[-1]

        self.obs_groups = obs_groups_pass
        return obs_groups_pass, obs_dim

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running stats using only the pass-through observation block."""
        if not self.obs_normalization or not self.obs_groups:
            return
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        mlp_obs = torch.cat(obs_list, dim=-1)
        self.obs_normalizer.update(mlp_obs)  # type: ignore[union-attr]


class PreEncodeMLPModel(PreEncodeMixin, MLPModel):
    """MLP model with MLP encoders on selected 1D observation groups.

    Observation groups listed in ``encoder_cfg`` are passed through a dedicated
    :class:`~rsl_rl.modules.mlp.MLP` each (unless shared ``encoders`` are supplied, same pattern as ``CNNModel.cnns``).
    Remaining groups are concatenated, optionally normalized, then concatenated **after** with the encoder outputs
    before the main policy MLP (pass-through first, then encoded), analogous to 1D + CNN layout in
    :class:`~rsl_rl.models.cnn_model.CNNModel`.
    """

    _encoder_cfg: dict[str, Any] | None
    _encoders_arg: nn.ModuleDict | None

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
        """Initialize the model.

        Args:
            obs: Observation dictionary (used to infer per-group dimensions).
            obs_groups: Maps observation sets to lists of observation group keys.
            obs_set: Which set this model uses (e.g. ``"actor"``, ``"student"``).
            output_dim: Policy / value head output dimension.
            hidden_dims: Hidden sizes of the main MLP head.
            activation: Activation for the main MLP and for encoders (unless overridden per encoder).
            obs_normalization: If True, normalize only the pass-through (non-encoded) concatenated vector.
            distribution_cfg: Optional distribution config for stochastic outputs.
            encoder_cfg: Per observation group, a dict with ``hidden_dims``, ``output_dim``, and optionally
                ``activation``. Ignored if ``encoders`` is provided.
            encoders: Optional shared :class:`~torch.nn.ModuleDict` (or ``dict`` of modules) for multi-model setups.
        """
        encoders_mod = self._pre_encode_prepare(obs, obs_groups, obs_set, activation, encoder_cfg, encoders)

        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        self.encoders = encoders_mod
        self._pre_encode_clear_staging()

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concatenate normalized pass-through features, then encoder outputs (in observation-set order)."""
        return self._pre_encode_concat_latent(obs)

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly module (pass-through tensor + one tensor per encoded group)."""
        return _TorchPreEncodeMLPModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX export wrapper (pass-through tensor + one tensor per encoded group)."""
        return _OnnxPreEncodeMLPModel(self, verbose)

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        return self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.encode_latent_dim


class PreEncodeRecurrentModel(PreEncodeMixin, RNNModel):
    r"""Recurrent model with MLP encoders on selected groups, then GRU/LSTM, then the policy MLP head.

    Does not use :class:`RNNModel`\ ``.__init__`` (which would size the RNN on pass-through dim only); instead calls
    :class:`MLPModel`\ ``.__init__`` and builds :class:`~rsl_rl.modules.rnn.RNN` with input size
    ``pass_through_dim + encode_latent_dim``.
    """

    _encoder_cfg: dict[str, Any] | None
    _encoders_arg: nn.ModuleDict | None

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
        """Initialize the pre-encode recurrent model (same encoder and RNN arguments as the respective base models)."""
        encoders_mod = self._pre_encode_prepare(obs, obs_groups, obs_set, activation, encoder_cfg, encoders)

        self.latent_dim = rnn_hidden_dim
        MLPModel.__init__(
            self,
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        self.encoders = encoders_mod
        self.rnn = RNN(self.obs_dim + self.encode_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self._pre_encode_clear_staging()

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Pre-encode observations, then run the recurrent core (see :class:`RNNModel`)."""
        latent_pre = self._pre_encode_concat_latent(obs)
        return self.rnn(latent_pre, masks, hidden_state).squeeze(0)

    def _get_obs_dim(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
    ) -> tuple[list[str], int]:
        return self._pre_encode_get_obs_dim(obs, obs_groups, obs_set)

    def as_jit(self) -> nn.Module:
        """Raise: TorchScript export is not supported for this model."""
        raise NotImplementedError("TorchScript export is not implemented for PreEncodeRecurrentModel.")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Raise: ONNX export is not supported for this model."""
        raise NotImplementedError("ONNX export is not implemented for PreEncodeRecurrentModel.")


class _TorchPreEncodeMLPModel(nn.Module):
    """TorchScript export: normalized pass-through vector + raw encoded group tensors."""

    def __init__(self, model: PreEncodeMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[g]) for g in model.obs_groups_encode])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_pass: torch.Tensor, obs_enc: list[torch.Tensor]) -> torch.Tensor:
        latent_pass = self.obs_normalizer(obs_pass)
        latent_enc_list = [enc(obs_enc[i]) for i, enc in enumerate(self.encoders)]
        latent_enc = torch.cat(latent_enc_list, dim=-1)
        latent = torch.cat([latent_pass, latent_enc], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxPreEncodeMLPModel(nn.Module):
    """ONNX export wrapper."""

    def __init__(self, model: PreEncodeMLPModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoders = nn.ModuleList([copy.deepcopy(model.encoders[g]) for g in model.obs_groups_encode])
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.obs_groups_encode = list(model.obs_groups_encode)
        self.obs_dim_pass = model.obs_dim

    def forward(self, obs_pass: torch.Tensor, *obs_enc: torch.Tensor) -> torch.Tensor:
        latent_pass = self.obs_normalizer(obs_pass)
        latent_enc_list = [enc(obs_enc[i]) for i, enc in enumerate(self.encoders)]
        latent_enc = torch.cat(latent_enc_list, dim=-1)
        latent = torch.cat([latent_pass, latent_enc], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        n_enc = len(self.obs_groups_encode)
        if n_enc == 0:
            return (torch.zeros(1, self.obs_dim_pass),)
        dummy_pass = torch.zeros(1, self.obs_dim_pass)
        dummies: list[torch.Tensor] = []
        for enc in self.encoders:
            seq = cast(nn.Sequential, enc)
            first = seq[0]
            if not isinstance(first, nn.Linear):
                raise TypeError("Encoder must start with nn.Linear for ONNX dummy inputs.")
            dummies.append(torch.zeros(1, first.in_features))
        return (dummy_pass, *dummies)

    @property
    def input_names(self) -> list[str]:
        return ["obs_pass", *self.obs_groups_encode]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
