"""Per-link contact-force-norm observation.

Reads :meth:`RigidEntity.get_links_net_contact_force` (Genesis-aggregated
per-link net force; ``(num_envs, n_links, 3)`` in world frame), slices to
the configured link set, and returns the per-link force magnitude.

Trade-off vs. ``gs.sensors.ContactForce``: the Genesis sensor binds to
**one** link per ``SensorOptions`` declaration, so a 14-link hand needs
14 sensor entries plus 14 obs-term entries to do what
:class:`ContactForceNorm` does in a single declaration. The cost is no
``NoisySensorMixin`` noise / delay / decimation modeling; that path is
reachable today via the existing :func:`sensor_reading` term plus a
per-link sensor declaration if a config wants noise on a small subset
of links.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from eden.managers.observation_manager import OBSERVATION_TERM_REGISTRY, ObservationTerm

if TYPE_CHECKING:
    pass


@OBSERVATION_TERM_REGISTRY.register()
class ContactForceNorm(ObservationTerm):
    """Per-link contact-force magnitude (Euclidean, world frame).

    Returns ``(num_envs, K)`` where ``K = len(matched links)``, each
    entry being ``‖f_link‖`` at the most recent ``scene.step()``.

    Parameters
    ----------
    entity_name:
        Entity to read forces from (must be a ``RigidEntity``).
    links_name:
        :func:`resolve_matching_names`-compatible patterns selecting which
        links contribute. Empty list raises in :meth:`build` — this term
        is per-link by design; use :func:`self_collision_cost` for a
        global "any contact" scalar.

    Notes
    -----
    Output scaling, noise, clipping, and history are controlled via
    the inherited :class:`ObservationTermOptions` fields (``scale``,
    ``noise``, ``clip``, ``history_length``) — pass e.g. ``scale=0.01``
    on the term options when forces are in newtons and the policy
    expects roughly-unit observations.

    The underlying force is in **world frame** — Genesis's
    ``get_links_net_contact_force`` returns the world-frame net force per
    link. ``gs.sensors.ContactForce`` (the noise-aware path reachable via
    the generic :func:`sensor_reading` obs term) returns the same vector
    in the **link's local frame**. Magnitudes are equal under any choice
    of frame, so this term's per-link norm is unaffected by the choice;
    a config swapping between this term and a sensor-reading-plus-norm
    composition gets identical scalar values but watch the frame if a
    downstream consumer uses the raw force vector.
    """

    entity_name: str = "robot"
    links_name: list[str] = []

    def build(self) -> None:
        self._entity = self._env.entities[self.entity_name]
        names, link_idx = self._entity.find_named_links_idx_local(self.links_name, preserve_order=True)
        if not link_idx:
            available = [link.name for link in self._entity.links]
            raise ValueError(
                f"ContactForceNorm: links_name={self.links_name} matched no links on entity "
                f"'{self.entity_name}'. Available links: {available}"
            )
        self._link_idx = torch.tensor(link_idx, dtype=torch.long, device=self.device)

    def compute(self) -> torch.Tensor:
        # Computed term (per-link norm): the final reduction targets the manager-owned cache when set,
        # avoiding a fresh per-step allocation + the manager copy; ``out=None`` (build-time shape probe)
        # allocates. build() caches the entity ref + resolved link idx so compute() does no per-step name
        # resolution. See ManagerTermBase for the cache protocol.
        forces = self._entity.get_links_net_contact_force()  # (B, n_links, 3)
        return torch.norm(forces[:, self._link_idx], dim=-1, out=self._cache)
