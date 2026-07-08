"""Particle entity wrapper for MPM/SPH/PBD deformables."""

from __future__ import annotations

import eden as en
from eden.entities.base import Entity


class ParticleEntity(Entity):
    """Entity subclass for particle-based materials (MPM, SPH).

    Particle entities have no DOFs, links, or orientation control.
    Position and rotation must be baked into the morph at creation time.
    """

    def set_pos(self, pos, envs_idx=None) -> None:
        """Particle entities do not support set_pos after build.

        Raises
        ------
        NotImplementedError
            Always. Bake position into the morph (e.g. ``gs.morphs.Mesh(pos=...)``) instead.
        """
        raise NotImplementedError(
            f"ParticleEntity '{self.name}' does not support set_pos. "
            "Bake position into the morph (e.g. gs.morphs.Mesh(pos=...)) instead."
        )

    def set_quat(self, quat, envs_idx=None, relative=True) -> None:
        """Particle entities do not support set_quat after build.

        Raises
        ------
        NotImplementedError
            Always. Bake rotation into the morph (e.g. ``gs.morphs.Mesh(euler=...)``) instead.
        """
        raise NotImplementedError(
            f"ParticleEntity '{self.name}' does not support set_quat. "
            "Bake rotation into the morph (e.g. gs.morphs.Mesh(euler=...)) instead."
        )

    def get_mass(self):
        """Particle entity mass is determined by material density and particle count.

        Genesis particle_entity.get_mass() has a bug with missing args,
        so we return the total mass from particle data instead.
        """
        try:
            return self._entity.get_mass()
        except TypeError:
            en.logger.warning(
                f"ParticleEntity '{self.name}': get_mass() not supported by Genesis particle entity. Returning 0."
            )
            return 0
