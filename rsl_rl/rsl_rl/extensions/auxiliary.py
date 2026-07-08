from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from tensordict import TensorDict

from rsl_rl.algorithms import AuxiliaryLoss
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.storage import RolloutStorage


class StudentLatentPredictionLoss(AuxiliaryLoss):
    """Predict an observation target from the student latent during distillation."""

    def __init__(
        self,
        target_obs: str,
        *,
        name: str | None = None,
        weight: float = 1.0,
        target_scale: float = 1.0,
        hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        activation: str = "elu",
        loss_type: str = "mse",
    ) -> None:
        super().__init__()
        self.target_obs = target_obs
        self.name = name or target_obs
        self.weight = weight
        self.target_scale = target_scale
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.loss_type = loss_type
        self.head: MLP | None = None

    def setup(
        self,
        student: MLPModel,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        device: str,
    ) -> Iterable[nn.Parameter]:
        if self.target_obs not in obs:
            raise ValueError(f"Auxiliary target observation {self.target_obs!r} is not present in rollout obs.")
        for obs_set in ("student", "teacher"):
            if self.target_obs in obs_groups.get(obs_set, ()):
                raise ValueError(
                    f"Auxiliary target observation {self.target_obs!r} must not be part of obs_groups[{obs_set!r}]."
                )

        target = obs[self.target_obs]
        if target.ndim < 2:
            raise ValueError(f"Auxiliary target observation {self.target_obs!r} must have a feature dimension.")

        with torch.no_grad():
            latent = self._prediction_features(student, obs)

        self.head = MLP(
            latent.shape[-1],
            target.shape[-1],
            self.hidden_dims,
            self.activation,
        ).to(device)
        return self.head.parameters()

    def compute(self, batch: RolloutStorage.Batch, student: MLPModel) -> dict[str, torch.Tensor]:
        if self.head is None:
            raise RuntimeError("StudentLatentPredictionLoss.setup() must be called before compute().")
        if batch.observations is None:
            raise ValueError("Auxiliary loss requires batch observations.")

        latent = self._prediction_features(student, batch.observations)
        prediction = self.head(latent)
        target = batch.observations[self.target_obs].detach() * self.target_scale
        prediction_loss = self._loss(prediction, target)
        return {self.name: self.weight * prediction_loss}

    def _prediction_features(self, student: MLPModel, obs: TensorDict) -> torch.Tensor:
        if not getattr(student, "is_recurrent", False):
            return student.get_latent(obs)
        if hasattr(student, "_pre_encode_concat_latent"):
            return student._pre_encode_concat_latent(obs)
        return MLPModel.get_latent(student, obs)

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            return nn.functional.mse_loss(prediction, target)
        if self.loss_type == "huber":
            return nn.functional.huber_loss(prediction, target)
        raise ValueError(f"Unknown auxiliary loss_type {self.loss_type!r}.")
