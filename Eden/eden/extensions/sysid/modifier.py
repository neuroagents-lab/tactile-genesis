"""Apply identified parameters / candidates to entities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from eden.extensions.sysid.parameter import Parameter, ParameterSet

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


DOF_PROPERTY_SETTERS: dict[str, str] = {
    "damping": "set_dofs_damping",
    "armature": "set_dofs_armature",
    "stiffness": "set_dofs_stiffness",
    "frictionloss": "set_dofs_frictionloss",
    "kp": "set_dofs_kp",
    "kd": "set_dofs_kd",
}

DOF_PROPERTY_GETTERS: dict[str, str] = {
    "damping": "get_dofs_damping",
    "armature": "get_dofs_armature",
    "stiffness": "get_dofs_stiffness",
    "frictionloss": "get_dofs_frictionloss",
    "kp": "get_dofs_kp",
    "kd": "get_dofs_kd",
}


def make_parameter_from_default(
    env: "EnvBase",
    name: str,
    property: str,
    dof_names: Sequence[str],
    entity_name: str = "robot",
    min_scale: float = 0.1,
    max_scale: float = 10.0,
    per_dof: bool = False,
    frozen: bool = False,
) -> Parameter:
    """Construct a Parameter using the entity's current value as the nominal.

    Reads the current per-DOF value via the entity getter and uses it as
    ``nominal``. Bounds are expressed as multiplicative scales of nominal,
    which is the natural parameterisation for the supported strictly
    positive properties (``damping``, ``armature``, ``stiffness``,
    ``frictionloss``, ``kp``, ``kd``). When the URDF default is zero
    (e.g. ``frictionloss``), the multiplicative bounds collapse — pass an
    explicit ``Parameter`` constructor in that case.
    """
    if property not in DOF_PROPERTY_GETTERS:
        raise ValueError(f"Unknown DOF property: {property!r}. Expected one of {list(DOF_PROPERTY_GETTERS)}.")
    entity = env.entities[entity_name]
    _, dof_indices = entity.find_named_dofs_idx_local(list(dof_names), preserve_order=True)
    values = getattr(entity, DOF_PROPERTY_GETTERS[property])(dofs_idx_local=dof_indices)
    if values.ndim == 2:
        values = values[0]
    values_np = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if per_dof:
        nominal = values_np
        lo = nominal * min_scale
        hi = nominal * max_scale
    else:
        nominal_scalar = float(values_np.mean())
        nominal = np.array([nominal_scalar], dtype=np.float64)
        lo = np.array([nominal_scalar * min_scale], dtype=np.float64)
        hi = np.array([nominal_scalar * max_scale], dtype=np.float64)
    return Parameter(
        name=name,
        property=property,
        dof_names=dof_names,
        entity_name=entity_name,
        per_dof=per_dof,
        nominal=nominal,
        min_value=lo,
        max_value=hi,
        frozen=frozen,
    )


def apply_parameters(env: "EnvBase", params: ParameterSet) -> None:
    """Write every free parameter's current value to all envs.

    Broadcasts the parameter vector to ``(num_envs, n_dofs)`` so the
    solver tensors are consistent across the batch.
    """
    for param in params:
        if param.frozen:
            continue
        _apply_one(env, param, envs_idx=None, per_env_values=None)


def apply_candidates(env: "EnvBase", params: ParameterSet, candidates: np.ndarray) -> None:
    """Write K distinct candidate parameter vectors into K environments.

    ``candidates`` has shape ``(K, n_free_params)`` and requires
    ``K == env.num_envs``. Each row is written to env ``k`` as the per-env
    value of the targeted solver tensor. Frozen parameters are untouched.
    """
    K, n_free = candidates.shape
    if n_free != params.size:
        raise ValueError(f"Candidates width {n_free} != ParameterSet.size {params.size}.")
    if K != env.num_envs:
        raise ValueError(f"apply_candidates requires K == env.num_envs (got K={K}, num_envs={env.num_envs}).")

    original = params.as_vector()
    try:
        per_env = [candidates[k] for k in range(K)]
        start = 0
        for param in params:
            if param.frozen:
                continue
            end = start + param.size
            values = np.stack([vec[start:end] for vec in per_env])
            start = end
            _apply_one(env, param, envs_idx=None, per_env_values=values)
    finally:
        params.update_from_vector(original)


def _apply_one(
    env: "EnvBase",
    param: Parameter,
    envs_idx: torch.Tensor | Sequence[int] | None,
    per_env_values: np.ndarray | None,
) -> None:
    setter_name = DOF_PROPERTY_SETTERS.get(param.property)
    if setter_name is None:
        raise ValueError(f"Unknown DOF property: {param.property!r}.")

    entity = env.entities[param.entity_name]
    _, dof_indices = entity.find_named_dofs_idx_local(list(param.dof_names), preserve_order=True)
    n_dofs = len(param.dof_names)

    if per_env_values is not None:
        # Shape (K, param.size) where param.size is 1 (shared) or n_dofs (per_dof).
        # Broadcast to (num_envs, n_dofs).
        K = per_env_values.shape[0]
        if param.per_dof:
            mat = per_env_values.astype(np.float64, copy=False)
        else:
            mat = np.broadcast_to(per_env_values, (K, n_dofs)).copy()
        tensor = torch.as_tensor(mat, dtype=torch.float32, device=env.device)
    else:
        values = param.as_dof_vector(n_dofs)
        tensor = torch.as_tensor(values, dtype=torch.float32, device=env.device)
        tensor = tensor.unsqueeze(0).expand(env.num_envs, -1).contiguous()

    getattr(entity, setter_name)(tensor, dofs_idx_local=dof_indices, envs_idx=envs_idx)
