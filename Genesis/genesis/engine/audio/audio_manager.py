from typing import TYPE_CHECKING

import genesis as gs

from .base_source import AudioSource

if TYPE_CHECKING:
    from genesis.engine.simulator import Simulator


class AudioManager:
    """
    Scene-level owner of audio *sources* (the synthesis stage), peer of ``SensorManager``.

    Sources generate sound from physics and register here; receiver sensors (``SpatialAudio`` / microphones) render
    the registry without knowing concrete source types. Two kinds of entries are held:

    - standalone sources added via ``scene.add_audio_source(...)`` (e.g. ``ActuationSource``), which the manager steps
      every ``scene.step()``;
    - *published* entries that a sensor exposes for its own already-synthesized block (e.g. ``ContactAudio``), which
      are synthesized by their backing sensor and need no stepping here.

    ``sources`` returns both kinds as a uniform list of :class:`AudioSource`.
    """

    # Maps an AudioSource options class to its source class. Populated as source types are registered (Phase B).
    SOURCE_TYPES_MAP: dict = {}

    def __init__(self, sim: "Simulator"):
        self._sim = sim
        self._sources: list[AudioSource] = []
        self._published: list[AudioSource] = []

    def add_source(self, source_options) -> AudioSource:
        source_cls = self.SOURCE_TYPES_MAP.get(type(source_options))
        if source_cls is None:
            gs.raise_exception(f"Unknown audio source options type: {type(source_options).__name__}.")
        source = source_cls(source_options, len(self._sources), self)
        self._sources.append(source)
        return source

    def register_published(self, entry: AudioSource):
        """Register a registry entry whose block is synthesized by some backing object (e.g. a sensor)."""
        self._published.append(entry)

    @property
    def sources(self) -> list[AudioSource]:
        return [*self._sources, *self._published]

    def build(self):
        for source in self._sources:
            source.build()

    def step(self):
        for source in self._sources:
            source.emit()

    def reset(self, envs_idx=None):
        for source in self._sources:
            source.reset(envs_idx)

    def destroy(self):
        self._sources.clear()
        self._published.clear()
