"""Parameter and parameter-set definitions for system identification."""

from __future__ import annotations

import copy
import pathlib
from typing import Iterator, Sequence

import numpy as np
import yaml

from eden.options.extensions.sysid import DofProperty


class Parameter:
    """A single sysid parameter with nominal value, box bounds, and a DOF-property target.

    A parameter drives one of the per-DOF solver fields
    (``damping``, ``armature``, ``stiffness``, ``frictionloss``) or one of
    the per-DOF PD gains (``kp``, ``kd``) on the entity named
    ``entity_name`` for the listed ``dof_names``.

    If ``per_dof`` is True, ``value`` is a length-len(dof_names) vector.
    Otherwise ``value`` is a scalar that is broadcast to all listed DOFs
    when applied to the entity.
    """

    __slots__ = (
        "name",
        "property",
        "entity_name",
        "dof_names",
        "per_dof",
        "nominal",
        "min_value",
        "max_value",
        "value",
        "frozen",
    )

    def __init__(
        self,
        name: str,
        property: DofProperty,
        dof_names: Sequence[str],
        entity_name: str = "robot",
        per_dof: bool = False,
        nominal: float | Sequence[float] = 0.0,
        min_value: float | Sequence[float] = 0.0,
        max_value: float | Sequence[float] = 1.0,
        frozen: bool = False,
    ) -> None:
        if not dof_names:
            raise ValueError(f"Parameter '{name}' must target at least one DOF.")

        self.name = name
        self.property = property
        self.entity_name = entity_name
        self.dof_names = tuple(dof_names)
        self.per_dof = per_dof
        self.frozen = frozen

        expected_size = len(self.dof_names) if per_dof else 1
        self.nominal = _as_vector(nominal, expected_size, f"{name}.nominal")
        self.min_value = _as_vector(min_value, expected_size, f"{name}.min_value")
        self.max_value = _as_vector(max_value, expected_size, f"{name}.max_value")
        if np.any(self.min_value > self.max_value):
            raise ValueError(f"Parameter '{name}' has min_value > max_value.")
        if np.any(self.nominal < self.min_value) or np.any(self.nominal > self.max_value):
            raise ValueError(f"Parameter '{name}' nominal is outside [min, max].")
        self.value = self.nominal.copy()

    @property
    def size(self) -> int:
        return int(self.nominal.size)

    def reset(self) -> None:
        self.value = self.nominal.copy()

    def clip(self) -> None:
        np.clip(self.value, self.min_value, self.max_value, out=self.value)

    def as_dof_vector(self, n_dofs: int) -> np.ndarray:
        """Broadcast ``value`` to a length-``n_dofs`` vector for set_dofs_<prop>."""
        if n_dofs != len(self.dof_names):
            raise ValueError(f"Parameter '{self.name}' targets {len(self.dof_names)} DOFs but got n_dofs={n_dofs}.")
        if self.per_dof:
            return self.value.astype(np.float64, copy=False)
        return np.full(n_dofs, float(self.value[0]), dtype=np.float64)

    def __repr__(self) -> str:
        frozen = " [frozen]" if self.frozen else ""
        return (
            f"Parameter(name={self.name!r}, property={self.property!r}, "
            f"dofs={len(self.dof_names)}, size={self.size}{frozen})"
        )

    def __getstate__(self) -> dict:
        return {
            "name": self.name,
            "property": self.property,
            "entity_name": self.entity_name,
            "dof_names": list(self.dof_names),
            "per_dof": self.per_dof,
            "nominal": self.nominal.tolist(),
            "min_value": self.min_value.tolist(),
            "max_value": self.max_value.tolist(),
            "value": self.value.tolist(),
            "frozen": self.frozen,
        }

    def __setstate__(self, state: dict) -> None:
        self.name = state["name"]
        self.property = state["property"]
        self.entity_name = state["entity_name"]
        self.dof_names = tuple(state["dof_names"])
        self.per_dof = state["per_dof"]
        self.nominal = np.asarray(state["nominal"], dtype=np.float64)
        self.min_value = np.asarray(state["min_value"], dtype=np.float64)
        self.max_value = np.asarray(state["max_value"], dtype=np.float64)
        self.value = np.asarray(state["value"], dtype=np.float64)
        self.frozen = state["frozen"]


class ParameterSet:
    """Ordered collection of ``Parameter`` objects with vectorised access.

    Frozen parameters are skipped by ``as_vector`` / ``get_bounds`` /
    ``update_from_vector`` so the decision-variable dimension seen by the
    optimiser equals the number of *free* scalars.
    """

    def __init__(self, parameters: Sequence[Parameter] | None = None) -> None:
        self._parameters: dict[str, Parameter] = {}
        for p in parameters or ():
            self.add(p)

    def add(self, parameter: Parameter) -> None:
        if parameter.name in self._parameters:
            raise ValueError(f"Duplicate parameter name: {parameter.name!r}")
        self._parameters[parameter.name] = parameter

    def copy(self) -> "ParameterSet":
        return copy.deepcopy(self)

    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[Parameter]:
        return iter(self._parameters.values())

    def __getitem__(self, name: str) -> Parameter:
        return self._parameters[name]

    def __contains__(self, name: str) -> bool:
        return name in self._parameters

    def values(self) -> list[Parameter]:
        return list(self._parameters.values())

    @property
    def size(self) -> int:
        return sum(p.size for p in self if not p.frozen)

    def as_vector(self) -> np.ndarray:
        free = [p.value.ravel() for p in self if not p.frozen]
        return np.concatenate(free) if free else np.zeros(0, dtype=np.float64)

    def as_nominal_vector(self) -> np.ndarray:
        free = [p.nominal.ravel() for p in self if not p.frozen]
        return np.concatenate(free) if free else np.zeros(0, dtype=np.float64)

    def update_from_vector(self, vector: np.ndarray) -> None:
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if vector.size != self.size:
            raise ValueError(f"Expected vector of size {self.size}, got {vector.size}.")
        start = 0
        for p in self:
            if p.frozen:
                continue
            end = start + p.size
            p.value = vector[start:end].reshape(p.nominal.shape).copy()
            start = end

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = [], []
        for p in self:
            if p.frozen:
                continue
            lo.append(p.min_value.ravel())
            hi.append(p.max_value.ravel())
        if not lo:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(lo), np.concatenate(hi)

    def reset(self) -> None:
        for p in self:
            p.reset()

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        lo, hi = self.get_bounds()
        return rng.uniform(lo, hi)

    def save_yaml(self, path: str | pathlib.Path) -> None:
        payload = {p.name: p.__getstate__() for p in self}
        pathlib.Path(path).write_text(yaml.safe_dump(payload, sort_keys=False))

    @classmethod
    def load_yaml(cls, path: str | pathlib.Path) -> "ParameterSet":
        payload = yaml.safe_load(pathlib.Path(path).read_text())
        params: list[Parameter] = []
        for name, state in payload.items():
            p = Parameter.__new__(Parameter)
            p.__setstate__(state)
            if p.name != name:
                p.name = name
            params.append(p)
        return cls(params)


def _as_vector(value: float | Sequence[float], size: int, label: str) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.size == 1 and size > 1:
        arr = np.full(size, float(arr[0]), dtype=np.float64)
    if arr.size != size:
        raise ValueError(f"{label}: expected size {size}, got {arr.size}.")
    return arr.copy()
