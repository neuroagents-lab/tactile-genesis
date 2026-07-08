# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Iterable
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class AuxiliaryLoss(nn.Module, ABC):
    """Interface for task-specific auxiliary losses used during distillation."""

    @abstractmethod
    def setup(
        self,
        student: MLPModel,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        device: str,
    ) -> Iterable[nn.Parameter]:
        """Initialize any auxiliary modules and return parameters to optimize with the student."""
        raise NotImplementedError

    @abstractmethod
    def compute(self, batch: RolloutStorage.Batch, student: MLPModel) -> dict[str, torch.Tensor]:
        """Return named scalar auxiliary losses for the current distillation batch."""
        raise NotImplementedError


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    student: MLPModel
    """The student model."""

    teacher: MLPModel
    """The teacher model."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        normalize_action_targets: bool = False,
        num_actions: int | None = None,
        auxiliary_losses: list[AuxiliaryLoss] | None = None,
        auxiliary_parameters: Iterable[nn.Parameter] | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,  # handle unused config parameters
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Distillation components
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.auxiliary_losses = list(auxiliary_losses or [])
        self.auxiliary_parameters = list(auxiliary_parameters or [])
        self._optimizer_parameters = self._collect_optimizer_parameters()

        # Create the optimizer
        optimizer_parameters = self._optimizer_parameters if self.auxiliary_losses else self.student.parameters()
        self.optimizer = resolve_optimizer(optimizer)(optimizer_parameters, lr=learning_rate)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = (None, None)

        # Distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        # Initialize the loss function
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        self.num_updates = 0

        # Action-target normalization: standardizes teacher actions per-dim so the behavior loss
        # becomes an inverse-variance-weighted MSE. This keeps small, task-critical corrections
        # from being drowned out by larger idle-pose actions in precision tasks.
        self.normalize_action_targets = normalize_action_targets
        if normalize_action_targets:
            if num_actions is None:
                raise ValueError("num_actions must be provided when normalize_action_targets is enabled.")
            self.action_normalizer: nn.Module = EmpiricalNormalization(num_actions).to(self.device)
        else:
            self.action_normalizer = nn.Identity()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Compute the actions
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.privileged_actions = self.teacher(obs).detach()
        # Update running statistics of the teacher actions used for target normalization
        if self.normalize_action_targets:
            self.action_normalizer.update(self.transition.privileged_actions)
        # Record the observations
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.student.update_normalization(obs)
        # Record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op since distillation does not use return targets."""
        # Not needed for distillation
        pass

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_auxiliary_losses: dict[str, float] = {}
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                # Inference of the student for gradient computation
                actions = self.student(batch.observations)

                # Behavior cloning loss (action targets optionally standardized per-dim)
                behavior_loss = self.loss_fn(
                    self.action_normalizer(actions),
                    self.action_normalizer(batch.privileged_actions),
                )
                batch_loss = behavior_loss

                # Auxiliary losses
                for auxiliary_loss in self.auxiliary_losses:
                    auxiliary_loss_dict = auxiliary_loss.compute(batch, self.student)
                    for name, auxiliary_loss_value in auxiliary_loss_dict.items():
                        batch_loss = batch_loss + auxiliary_loss_value
                        key = f"aux/{name}"
                        mean_auxiliary_losses[key] = mean_auxiliary_losses.get(key, 0.0) + auxiliary_loss_value.item()

                # Total loss
                loss = loss + batch_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                # Gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self._optimizer_parameters, self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                # Reset dones
                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        for key in mean_auxiliary_losses:
            mean_auxiliary_losses[key] /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        # Construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}
        loss_dict.update(mean_auxiliary_losses)

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for the student and keep the teacher in eval mode."""
        self.student.train()
        self.action_normalizer.train()
        for auxiliary_loss in self.auxiliary_losses:
            auxiliary_loss.train()
        # Teacher is always in eval mode
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for student and teacher models."""
        self.student.eval()
        self.teacher.eval()
        self.action_normalizer.eval()
        for auxiliary_loss in self.auxiliary_losses:
            auxiliary_loss.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.normalize_action_targets:
            saved_dict["action_normalizer_state_dict"] = self.action_normalizer.state_dict()
        if self.auxiliary_losses:
            saved_dict["auxiliary_loss_state_dicts"] = [
                auxiliary_loss.state_dict() for auxiliary_loss in self.auxiliary_losses
            ]
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, determine what to load automatically
        if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):  # Load from RL training
            load_cfg = {"teacher": True, "iteration": False}  # Only load teacher by default
        elif load_cfg is None:  # Load from distillation training
            load_cfg = {
                "student": True,
                "teacher": True,
                "auxiliary_losses": bool(self.auxiliary_losses),
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("student"):
            self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
            # Restore action-target normalization stats if present (guarded for older checkpoints)
            if self.normalize_action_targets and "action_normalizer_state_dict" in loaded_dict:
                self.action_normalizer.load_state_dict(loaded_dict["action_normalizer_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True
        if load_cfg.get("auxiliary_losses"):
            auxiliary_loss_state_dicts = loaded_dict["auxiliary_loss_state_dicts"]
            if len(auxiliary_loss_state_dicts) != len(self.auxiliary_losses):
                raise ValueError(
                    f"Expected {len(self.auxiliary_losses)} auxiliary loss state dicts, "
                    f"got {len(auxiliary_loss_state_dicts)}."
                )
            for auxiliary_loss, auxiliary_loss_state_dict in zip(self.auxiliary_losses, auxiliary_loss_state_dicts):
                auxiliary_loss.load_state_dict(auxiliary_loss_state_dict, strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.student

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> Distillation:
        """Construct the distillation algorithm."""
        # Resolve class callables
        alg_class: type[Distillation] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["student", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Distillation is not compatible with RND and symmetry extensions
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with Distillation.")
        cfg["algorithm"]["symmetry_cfg"] = None

        # Initialize the policy
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")

        # Initialize the storage
        storage = RolloutStorage("distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize auxiliary losses before the optimizer is created
        auxiliary_loss_cfgs = cfg["algorithm"].pop("auxiliary_losses", None)
        auxiliary_losses, auxiliary_parameters = Distillation._construct_auxiliary_losses(
            auxiliary_loss_cfgs, student, obs, cfg["obs_groups"], device
        )

        # Initialize the algorithm
        alg: Distillation = alg_class(
            student,
            teacher,
            storage,
            device=device,
            num_actions=env.num_actions,
            auxiliary_losses=auxiliary_losses,
            auxiliary_parameters=auxiliary_parameters,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    @staticmethod
    def _construct_auxiliary_losses(
        auxiliary_loss_cfgs: list[dict] | None,
        student: MLPModel,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        device: str,
    ) -> tuple[list[AuxiliaryLoss], list[nn.Parameter]]:
        """Instantiate configured auxiliary losses and collect optimizer parameters."""
        if not auxiliary_loss_cfgs:
            return [], []

        auxiliary_losses: list[AuxiliaryLoss] = []
        auxiliary_parameters: list[nn.Parameter] = []
        for auxiliary_loss_cfg in auxiliary_loss_cfgs:
            if "class_name" not in auxiliary_loss_cfg:
                raise ValueError("Each auxiliary loss config must contain a 'class_name' entry.")

            auxiliary_loss_kwargs = dict(auxiliary_loss_cfg)
            auxiliary_loss_class = resolve_callable(auxiliary_loss_kwargs.pop("class_name"))
            auxiliary_loss = auxiliary_loss_class(**auxiliary_loss_kwargs)
            if not isinstance(auxiliary_loss, AuxiliaryLoss):
                raise TypeError(
                    f"Auxiliary loss '{auxiliary_loss_class}' must inherit from "
                    "rsl_rl.algorithms.distillation.AuxiliaryLoss."
                )

            auxiliary_loss.to(device)
            setup_parameters = list(auxiliary_loss.setup(student, obs, obs_groups, device))
            auxiliary_loss.to(device)
            student.to(device)

            auxiliary_losses.append(auxiliary_loss)
            auxiliary_parameters.extend(setup_parameters)

        return auxiliary_losses, auxiliary_parameters

    def _collect_optimizer_parameters(self) -> list[nn.Parameter]:
        """Return student and auxiliary parameters without duplicates."""
        parameters: list[nn.Parameter] = []
        seen: set[int] = set()
        for parameter in [*self.student.parameters(), *self.auxiliary_parameters]:
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue
            parameters.append(parameter)
            seen.add(parameter_id)
        return parameters

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.student.state_dict(), self.teacher.state_dict()]
        if self.auxiliary_losses:
            model_params.append([auxiliary_loss.state_dict() for auxiliary_loss in self.auxiliary_losses])
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.student.load_state_dict(model_params[0])
        self.teacher.load_state_dict(model_params[1])
        if self.auxiliary_losses:
            for auxiliary_loss, auxiliary_loss_state_dict in zip(self.auxiliary_losses, model_params[2]):
                auxiliary_loss.load_state_dict(auxiliary_loss_state_dict)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self._optimizer_parameters if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self._optimizer_parameters:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
