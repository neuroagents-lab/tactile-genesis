from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.options.sensors import SpatialAudio as SpatialAudioOptions
from genesis.utils.misc import concat_with_tensor, make_tensor_field, tensor_to_array

from .base_sensor import SimpleSensor, SimpleSensorMetadata

if TYPE_CHECKING:
    from genesis.engine.audio import AudioManager
    from genesis.engine.entities.rigid_entity.rigid_link import RigidLink
    from genesis.engine.solvers import RigidSolver
    from genesis.ext.pyrender.mesh import Mesh
    from genesis.vis.rasterizer_context import RasterizerContext

    from .sensor_manager import SensorManager


@dataclass
class SpatialAudioSensorMetadata(SimpleSensorMetadata):
    """
    Shared metadata for all airborne ``SpatialAudio`` (microphone) sensors.

    Holds the per-listener geometry/propagation parameters, a reference to the scene's ``AudioManager`` (the source
    registry it renders), and the rolling source-sample history used for the propagation delay line. ``audio_substeps``
    is class-uniform (asserted at build) so the output blocks are rectangular over all listeners.
    """

    solver: "RigidSolver | None" = None
    # The scene's audio source registry. Each step the mic concatenates every registered source's block / emit_links /
    # emit_offset and renders them with distance attenuation + propagation delay. None until build.
    audio_manager: "AudioManager | None" = None
    audio_substeps: int = 0
    hist_len: int = 0  # source-history length in samples; sized from the largest listener `max_delay`

    # Per-listener parameters, shape (n_listeners,) (or (n_listeners, 3) for the offset). One column grown per sensor.
    listener_link: torch.Tensor = make_tensor_field((0,), dtype_factory=lambda: gs.tc_int)  # global link idx, -1=static
    listener_offset: torch.Tensor = make_tensor_field((0, 3))
    speed_of_sound: torch.Tensor = make_tensor_field((0,))
    ref_distance: torch.Tensor = make_tensor_field((0,))
    atten_power: torch.Tensor = make_tensor_field((0,))  # 1.0 (inverse) or 2.0 (inverse_square)
    doppler: torch.Tensor = make_tensor_field((0,))  # 1.0 if Doppler ramp enabled, else 0.0

    # Allocated lazily on the first update once n_listeners / n_src / hist_len are final.
    src_hist: torch.Tensor = make_tensor_field((0, 0, 0))  # (B, n_src, hist_len) rolling source samples
    prev_dist: torch.Tensor = make_tensor_field((0, 0, 0))  # (B, n_listeners, n_src) last-step distances (Doppler)


class SpatialAudioSensor(SimpleSensor[SpatialAudioOptions, None, SpatialAudioSensorMetadata]):
    """
    Airborne mono point-microphone sensor (see :class:`~genesis.options.sensors.SpatialAudio`).

    Renders the airborne sound a listener point hears by summing, over every audio source registered with the scene's
    ``AudioManager`` (contact mics, actuation, ...), that source's synthesized block attenuated by distance and delayed
    by ``distance / speed_of_sound``. A fractional delay line (a rolling per-source sample history) provides the delay;
    ramping it across each block yields Doppler. The mic is decoupled from concrete source types via the registry.
    """

    def __init__(
        self,
        options: SpatialAudioOptions,
        idx: int,
        shared_context,
        shared_metadata,
        manager: "SensorManager",
    ):
        super().__init__(options, idx, shared_context, shared_metadata, manager)
        self._link: "RigidLink | None" = None
        self.debug_object: "Mesh | None" = None

    def build(self):
        super().build()

        sm = self._shared_metadata
        if sm.solver is None:
            sm.solver = self._manager._sim.rigid_solver
        if sm.audio_manager is None:
            sm.audio_manager = self._manager._sim._audio_manager

        # audio_substeps is class-uniform; each source's block must also carry this K (checked at render time).
        if sm.audio_substeps == 0:
            sm.audio_substeps = self._options.audio_substeps
        elif sm.audio_substeps != self._options.audio_substeps:
            gs.raise_exception(
                "All SpatialAudio sensors must share the same audio_substeps. "
                f"Got {self._options.audio_substeps} vs existing {sm.audio_substeps}."
            )

        # Resolve the listener link (static if not attached to an entity).
        if self._options.entity_idx >= 0:
            entity = self._manager._sim.entities[self._options.entity_idx]
            self._link = entity.links[self._options.link_idx_local]
            link_idx = self._options.link_idx_local + entity.link_start
        else:
            link_idx = -1

        off = self._options.pos_offset
        atten_power = 2.0 if self._options.attenuation == "inverse_square" else 1.0
        sm.listener_link = concat_with_tensor(sm.listener_link, link_idx)
        sm.listener_offset = concat_with_tensor(sm.listener_offset, [[off[0], off[1], off[2]]], dim=0)
        sm.speed_of_sound = concat_with_tensor(sm.speed_of_sound, self._options.speed_of_sound)
        sm.ref_distance = concat_with_tensor(sm.ref_distance, self._options.ref_distance)
        sm.atten_power = concat_with_tensor(sm.atten_power, atten_power)
        sm.doppler = concat_with_tensor(sm.doppler, 1.0 if self._options.enable_doppler else 0.0)

        # History must span the longest propagation delay across all listeners plus the current block.
        dt_sub = self._manager._sim.dt / self._options.audio_substeps
        delay_samples = int(self._options.max_delay / dt_sub) + 2
        sm.hist_len = max(sm.hist_len, self._options.audio_substeps + delay_samples)

    def _get_return_format(self) -> tuple[int, ...]:
        return (self._options.audio_substeps,)

    @classmethod
    def _get_cache_dtype(cls) -> torch.dtype:
        return gs.tc_float

    @classmethod
    def reset(cls, shared_metadata: SpatialAudioSensorMetadata, shared_ground_truth_cache: torch.Tensor, envs_idx):
        super().reset(shared_metadata, shared_ground_truth_cache, envs_idx)
        for name in ("src_hist", "prev_dist"):
            t = getattr(shared_metadata, name)
            if t.numel() > 0:
                t[envs_idx] = 0.0

    @classmethod
    def _update_raw_data(
        cls, shared_context: None, shared_metadata: SpatialAudioSensorMetadata, raw_data_T: torch.Tensor
    ):
        sm = shared_metadata
        solver = sm.solver
        assert solver is not None

        K = sm.audio_substeps
        B = raw_data_T.shape[1]
        n_listeners = sm.listener_link.shape[0]

        # Gather every registered source into one combined block. Each entry exposes block (B, n_emit, K),
        # emit_links (n_emit,) and emit_offset (B, n_emit, 3); we concatenate along the emission-point axis.
        blocks, links, offsets = [], [], []
        for source in sm.audio_manager.sources:
            blk = source.block
            if blk.numel() == 0:
                continue
            if blk.shape[-1] != K:
                gs.raise_exception(
                    f"SpatialAudio audio_substeps={K} but an audio source emits blocks of {blk.shape[-1]} samples; "
                    "all sources and microphones must share audio_substeps."
                )
            blocks.append(blk)
            links.append(source.emit_links)
            offsets.append(source.emit_offset)
        # No radiation sources (none registered, or none synthesized yet) -> the mic is silent.
        if not blocks:
            raw_data_T[:] = 0.0
            return
        new_block = torch.cat(blocks, dim=1)  # (B, n_src, K)
        src_link = torch.cat(links, dim=0).long()  # (n_src,)
        src_offset = torch.cat(offsets, dim=1)  # (B, n_src, 3)
        n_src = new_block.shape[1]
        H = sm.hist_len

        # Lazily (re)allocate the delay line + Doppler history once shapes are final.
        if sm.src_hist.shape != (B, n_src, H):
            sm.src_hist = torch.zeros((B, n_src, H), dtype=gs.tc_float, device=gs.device)
        if sm.prev_dist.shape != (B, n_listeners, n_src):
            sm.prev_dist = torch.zeros((B, n_listeners, n_src), dtype=gs.tc_float, device=gs.device)

        # Append this step's source blocks to the rolling history (newest samples at the end).
        sm.src_hist.copy_(torch.cat([sm.src_hist[..., K:], new_block], dim=-1))

        dt_sub = solver._sim.dt / K
        all_pos = solver.get_links_pos()
        if solver.n_envs == 0:
            all_pos = all_pos[None]  # (B, n_links, 3)

        # Source positions: link-attached emission points ride their link; static points (link < 0) sit at the offset.
        src_attached = all_pos[:, src_link.clamp(min=0), :] + src_offset  # (B, n_src, 3)
        src_pos = torch.where((src_link < 0).view(1, -1, 1), src_offset, src_attached)

        # Listener positions: attached links ride their link, static listeners sit at their world offset.
        listener_link = sm.listener_link.long()  # (n_listeners,)
        lp_attached = all_pos[:, listener_link.clamp(min=0), :] + sm.listener_offset[None]  # (B, nL, 3)
        is_static = (listener_link < 0).view(1, -1, 1)
        listener_pos = torch.where(is_static, sm.listener_offset[None].expand(B, n_listeners, 3), lp_attached)

        # Distance, attenuation, and delay (in samples) for every (listener, source) pair.
        r = (listener_pos[:, :, None, :] - src_pos[:, None, :, :]).norm(dim=-1)  # (B, nL, n_src)
        ref = sm.ref_distance.view(1, -1, 1)
        power = sm.atten_power.view(1, -1, 1)
        c = sm.speed_of_sound.view(1, -1, 1)
        max_d = float(H - K - 1)
        atten_cur = (ref / r.clamp(min=ref)) ** power
        d_cur = (r / c / dt_sub).clamp(max=max_d)

        # Doppler: ramp delay/attenuation across the block from last step's value to this step's. On the first step
        # (prev_dist == 0) or with Doppler disabled, hold this step's value so there is no spurious ramp.
        prev_dist = sm.prev_dist
        atten_prev = (ref / prev_dist.clamp(min=ref)) ** power
        d_prev = (prev_dist / c / dt_sub).clamp(max=max_d)
        hold = (prev_dist <= 0) | (sm.doppler.view(1, -1, 1) <= 0.5)
        d_prev = torch.where(hold, d_cur, d_prev)
        atten_prev = torch.where(hold, atten_cur, atten_prev)

        kfrac = (torch.arange(1, K + 1, device=gs.device, dtype=gs.tc_float) / K).view(1, 1, 1, K)
        d_k = d_prev[..., None] * (1.0 - kfrac) + d_cur[..., None] * kfrac  # (B, nL, n_src, K)
        atten_k = atten_prev[..., None] * (1.0 - kfrac) + atten_cur[..., None] * kfrac

        # Fractional read index into the history for block sample k: now - (K-1-k) - delay.
        kpos = torch.arange(K, device=gs.device, dtype=gs.tc_float).view(1, 1, 1, K)
        idx_f = ((H - K) + kpos - d_k).clamp(min=0.0, max=float(H) - 1.001)
        i0 = idx_f.floor().long()
        frac = idx_f - i0.to(gs.tc_float)
        hist = sm.src_hist[:, None, :, :].expand(B, n_listeners, n_src, H)  # broadcast over listeners
        s0 = hist.gather(3, i0)
        s1 = hist.gather(3, (i0 + 1).clamp(max=H - 1))
        sample = s0 * (1.0 - frac) + s1 * frac  # (B, nL, n_src, K)

        out = (atten_k * sample).sum(dim=2)  # (B, nL, K)
        sm.prev_dist.copy_(r)

        # Cache layout is (n_listeners * K, B), matching ContactAudio.
        raw_data_T[:] = out.permute(1, 2, 0).reshape(n_listeners * K, B)

    def _draw_debug(self, context: "RasterizerContext"):
        """
        Draw a sphere at the listener position whose radius grows with the loudness of the most recent block.
        """
        env_idx = context.rendered_envs_idx[0] if self._manager._sim.n_envs > 0 else None
        if self._link is not None:
            pos = tensor_to_array(self._link.get_pos(env_idx).reshape((3,))) + self._options.pos_offset
        else:
            pos = self._options.pos_offset
        amp = float(self.read(env_idx).reshape(-1).abs().max())

        if self.debug_object is not None:
            context.clear_debug_object(self.debug_object)
            self.debug_object = None
        radius = 0.02 + min(amp, 1.0) * 0.05
        self.debug_object = context.draw_debug_sphere(pos=pos, radius=radius, color=(0.2, 0.6, 1.0, 0.6))
