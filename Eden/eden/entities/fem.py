"""FEM (finite-element) soft-body entity wrapper."""

from __future__ import annotations

from typing import Literal

import torch

import eden as en
from eden.entities.base import Entity


class FEMEntity(Entity):
    """Entity subclass for FEM-based deformable materials (Elastic, Cloth, Muscle).

    FEM entities use vertex-based state (positions and velocities of mesh vertices)
    rather than rigid-body state. They have no DOFs, links, or joint-based control.

    The Genesis FEMEntity does not expose get_pos/get_vel/get_quat/get_ang, so this
    class overrides those methods to provide a compatible interface (mean vertex
    position, mean velocity, identity quaternion, zero angular velocity).
    """

    _identity_quat: torch.Tensor | None = None
    _zero_ang: torch.Tensor | None = None

    def get_pos(self, envs_idx=None):
        """Return mean vertex position (geometric centroid), shape (B, 3).

        Note: this is the unweighted mean of vertex positions, not the
        mass-weighted center of mass.
        """
        state = self._entity.get_state()
        pos = state.pos  # (B, n_vertices, 3)
        centroid = pos.mean(dim=1)  # (B, 3)
        if envs_idx is not None:
            return centroid[envs_idx]
        return centroid

    def get_vel(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return mean vertex velocity, shape (B, 3). Only world frame is supported."""
        if frame != "world":
            raise ValueError(f"FEMEntity only supports frame='world', got '{frame}'.")
        state = self._entity.get_state()
        vel = state.vel  # (B, n_vertices, 3)
        mean_vel = vel.mean(dim=1)  # (B, 3)
        if envs_idx is not None:
            return mean_vel[envs_idx]
        return mean_vel

    def get_quat(self, envs_idx=None):
        """Return identity quaternion (wxyz) for all envs, shape (B, 4).

        FEM entities have no rigid-body orientation. Returns a constant identity
        quaternion for interface compatibility.
        """
        if self._identity_quat is None or self._identity_quat.shape[0] != self._env.num_envs:
            self._identity_quat = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(self._env.num_envs, -1).contiguous()
            )
        if envs_idx is not None:
            return self._identity_quat[envs_idx]
        return self._identity_quat

    def get_ang(self, envs_idx=None, *, frame: Literal["world", "body"] = "world"):
        """Return zero angular velocity for all envs, shape (B, 3). Only world frame is supported."""
        if frame != "world":
            raise ValueError(f"FEMEntity only supports frame='world', got '{frame}'.")
        if self._zero_ang is None or self._zero_ang.shape[0] != self._env.num_envs:
            self._zero_ang = torch.zeros(self._env.num_envs, 3, device=self.device)
        if envs_idx is not None:
            return self._zero_ang[envs_idx]
        return self._zero_ang

    def set_pos(self, pos, envs_idx=None):
        """Set FEM vertex positions via Genesis ``set_position``.

        Forwarded directly to the Genesis FEM entity. Accepted shapes:

        - ``(3,)`` — COM offset relative to initial vertex positions, applied to all envs.
        - ``(n_vertices, 3)`` — absolute per-vertex positions, tiled to all envs.
        - ``(B, 3)`` — per-env COM offsets relative to initial vertex positions.
        - ``(B, n_vertices, 3)`` — full batched per-vertex absolute positions.

        Note that ``(3,)`` and ``(B, 3)`` are **offsets from the initial COM**,
        not absolute world positions.

        Parameters
        ----------
        pos : torch.Tensor
            Position tensor (see shapes above).
        envs_idx : None
            Not supported by Genesis FEM entities. Must be None.

        Raises
        ------
        NotImplementedError
            If envs_idx is not None.
        """
        if envs_idx is not None:
            raise NotImplementedError(
                f"FEMEntity '{self.name}' does not support per-env set_pos. Pass positions for all envs instead."
            )
        self._entity.set_position(pos)

    def set_quat(self, quat, envs_idx=None, relative=True):
        """FEM entities do not support set_quat.

        Raises
        ------
        NotImplementedError
            Always. Bake rotation into the morph (e.g. ``gs.morphs.Mesh(euler=...)``) instead.
        """
        raise NotImplementedError(
            f"FEMEntity '{self.name}' does not support set_quat. "
            "Bake rotation into the morph (e.g. gs.morphs.Mesh(euler=...)) instead."
        )

    def attach_to(self, entity, link_name: str) -> None:
        """FEM entities do not support attachment to other entities.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            f"FEMEntity '{self.name}' does not support attach_to. Only RigidEntity supports parent-child attachments."
        )

    def get_mass(self):
        """Return FEM entity mass. Falls back to 0 if Genesis does not support it."""
        try:
            return self._entity.get_mass()
        except (TypeError, AttributeError):
            en.logger.warning(f"FEMEntity '{self.name}': get_mass() not supported by Genesis FEM entity. Returning 0.")
            return 0

    # ------------------------------------------------------------------
    # FEM-specific state accessors
    # ------------------------------------------------------------------

    def get_vertex_positions(self, envs_idx=None):
        """Return per-vertex positions, shape (B, n_vertices, 3)."""
        state = self._entity.get_state()
        pos = state.pos
        if envs_idx is not None:
            return pos[envs_idx]
        return pos

    def get_vertex_velocities(self, envs_idx=None):
        """Return per-vertex velocities, shape (B, n_vertices, 3)."""
        state = self._entity.get_state()
        vel = state.vel
        if envs_idx is not None:
            return vel[envs_idx]
        return vel

    def set_actuation(self, actu):
        """Set muscle actuation signal (Muscle material only).

        Parameters
        ----------
        actu : torch.Tensor
            Actuation tensor. Accepted shapes:
            - (): scalar for all groups
            - (n_groups,): per-group actuation
            - (B, n_groups): batched per-group actuation
        """
        self._entity.set_actuation(actu)

    def set_muscle(self, muscle_group=None, muscle_direction=None):
        """Set muscle group IDs and/or fiber directions (Muscle material only).

        Parameters
        ----------
        muscle_group : array_like, optional
            Shape (n_elements,) — muscle group ID per element.
        muscle_direction : array_like, optional
            Shape (n_elements, 3) — unit direction vectors for muscle forces.
        """
        self._entity.set_muscle(muscle_group=muscle_group, muscle_direction=muscle_direction)

    def set_vertex_constraints(self, verts_idx_local, target_poss=None, link=None, **kwargs):
        """Pin vertices to target positions or to a rigid link.

        Thin wrapper around Genesis FEMEntity.set_vertex_constraints.

        Parameters
        ----------
        verts_idx_local : array_like
            Local vertex indices to constrain.
        target_poss : array_like, optional
            Target positions (len(verts_idx), 3). If None, uses current positions.
        link : RigidLink, optional
            Rigid link for vertices to follow.
        **kwargs
            Additional keyword arguments forwarded to Genesis (e.g., is_soft_constraint,
            stiffness, envs_idx).
        """
        self._entity.set_vertex_constraints(verts_idx_local, target_poss=target_poss, link=link, **kwargs)

    @property
    def n_vertices(self) -> int:
        return self._entity.n_vertices

    @property
    def n_elements(self) -> int:
        return self._entity.n_elements
