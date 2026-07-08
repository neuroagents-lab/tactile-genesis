"""Base class for teleoperation input devices."""

from typing import Callable
from abc import abstractmethod

from genesis.repr_base import RBC
import torch


class DeviceBase(RBC):
    """Base class for all devices."""

    def __init__(self):
        self._callbacks = {}

    def __str__(self) -> str:
        """Return: A string containing the information of joystick."""
        return f"{self.__class__.__name__}"

    def add_callback(self, key: str, callback: Callable) -> None:
        self._callbacks[key] = callback

    def remove_callback(self, key: str) -> Callable:
        return self._callbacks.pop(key, None)

    def reset(self) -> None:
        pass

    @abstractmethod
    def get_raw_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_command(self) -> torch.Tensor:
        pass
