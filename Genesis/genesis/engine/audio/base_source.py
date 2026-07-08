from typing import Callable

import torch


class AudioSource:
    """
    Base class for objects that radiate sound into the scene's :class:`AudioManager` registry (the synthesis stage of
    the synthesis -> propagation -> rendering pipeline).

    A source *generates* audio from physics (contacts, actuation, ...). It is **not** a sensor: it is never ``read``;
    a receiver sensor (``SpatialAudio`` / microphone) renders the registry. Subclasses implement ``build`` / ``emit``
    / ``reset`` and expose the radiation interface so receivers stay decoupled from concrete source types:

    - ``block``: ``(B, n_emit, K)`` the latest synthesized audio block per emission point.
    - ``emit_links``: ``(n_emit,)`` world rigid-link index of each emission point (``-1`` = static / world frame).
    - ``emit_offset``: ``(B, n_emit, 3)`` link-local (or world, for static points) offset of each emission point.
    """

    def build(self):
        """Allocate state once the scene is built."""

    def emit(self):
        """Synthesize this step's audio block into the source's persistent state."""

    def reset(self, envs_idx):
        """Reset persistent synthesis state for the given environments."""

    @property
    def block(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def emit_links(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def emit_offset(self) -> torch.Tensor:
        raise NotImplementedError


class PublishedSource(AudioSource):
    """
    A registry entry that exposes another object's already-synthesized block as a radiation source -- e.g. a
    ``ContactAudio`` sensor publishing its structure-borne output so the airborne microphone can render it.

    The three tensors are read through callables (not captured directly) so the entry always sees the backing object's
    current tensors, even after they are grown / reallocated at build time. ``emit`` is a no-op because the backing
    object does the synthesis.
    """

    def __init__(
        self,
        block_fn: Callable[[], torch.Tensor],
        links_fn: Callable[[], torch.Tensor],
        offset_fn: Callable[[], torch.Tensor],
    ):
        self._block_fn = block_fn
        self._links_fn = links_fn
        self._offset_fn = offset_fn

    @property
    def block(self) -> torch.Tensor:
        return self._block_fn()

    @property
    def emit_links(self) -> torch.Tensor:
        return self._links_fn()

    @property
    def emit_offset(self) -> torch.Tensor:
        return self._offset_fn()
