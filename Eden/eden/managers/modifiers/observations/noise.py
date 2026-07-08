"""Observation noise modifiers (constant, uniform, Gaussian)."""

import torch

from eden.envs.base import EnvBase
from eden.managers.modifiers.base import NoiseModel, NOISE_MODEL_REGISTRY
from eden.options.managers.observations import NoiseOptions
from eden.constants import NoiseOperation


def ensure_tensor_on_device(value: torch.Tensor | float, device: torch.device) -> torch.Tensor | float:
    """Ensure tensor is on the correct device, leave scalars unchanged."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return value


@NOISE_MODEL_REGISTRY.register()
class ConstantNoise(NoiseModel):
    bias: torch.Tensor | float = 0.0

    def __init__(self, env: EnvBase, options: NoiseOptions):
        super().__init__(env=env, options=options)
        self.bias = ensure_tensor_on_device(self.bias, device=self.device)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        match self.operation:
            case NoiseOperation.ADD:
                return data + self.bias
            case NoiseOperation.SCALE:
                return data * self.bias
            case _:
                return self.bias


@NOISE_MODEL_REGISTRY.register()
class UniformNoise(NoiseModel):
    n_min: torch.Tensor | float = -1.0
    n_max: torch.Tensor | float = 1.0

    def __init__(self, env: EnvBase, options: NoiseOptions):
        super().__init__(env=env, options=options)
        if isinstance(self.n_min, (int, float)) and isinstance(self.n_max, (int, float)):
            if self.n_min >= self.n_max:
                raise ValueError(f"n_min ({self.n_min}) must be less than n_max ({self.n_max})")

        self.n_min = ensure_tensor_on_device(self.n_min, device=self.device)
        self.n_max = ensure_tensor_on_device(self.n_max, device=self.device)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # Generate uniform noise in [0, 1) and scale to [n_min, n_max).
        noise = torch.rand_like(data) * (self.n_max - self.n_min) + self.n_min
        match self.operation:
            case NoiseOperation.ADD:
                return data + noise
            case NoiseOperation.SCALE:
                return data * noise
            case _:
                return noise


@NOISE_MODEL_REGISTRY.register()
class GaussianNoise(NoiseModel):
    mean: torch.Tensor | float = 0.0
    std: torch.Tensor | float = 1.0

    def __init__(self, env: EnvBase, options: NoiseOptions):
        super().__init__(env=env, options=options)
        if isinstance(self.std, (int, float)) and self.std <= 0:
            raise ValueError(f"std ({self.std}) must be positive")

        self.mean = ensure_tensor_on_device(self.mean, device=self.device)
        self.std = ensure_tensor_on_device(self.std, device=self.device)

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        # Generate standard normal noise and scale.
        noise = self.mean + self.std * torch.randn_like(data)
        match self.operation:
            case NoiseOperation.ADD:
                return data + noise
            case NoiseOperation.SCALE:
                return data * noise
            case _:
                return noise
