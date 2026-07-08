from typing import TYPE_CHECKING

import torch

import genesis as gs
from genesis.options.audio import ActuationSource as ActuationSourceOptions

from .audio_manager import AudioManager
from .base_source import AudioSource
from .synthesis import bandpass_coeffs, resonator_coeffs

if TYPE_CHECKING:
    from genesis.engine.solvers import RigidSolver


class ActuationAudioSource(AudioSource):
    """
    Motor/joint actuation-noise source (see :class:`~genesis.options.audio.ActuationSource`).

    One emission point per covered DOF, radiating from that joint's child link. Each step the per-DOF actuation effort
    (``get_dofs_control_force``) and speed (``get_dofs_velocity``) drive a velocity-pitched whine partial bank, an idle
    hum, velocity-scaled friction noise, and a reversal click; loudness follows
    ``load_gain*|tau| + power_gain*|tau*omega|``. Fully batched over (B, n_emit).
    """

    def __init__(self, options: ActuationSourceOptions, source_idx: int, manager: AudioManager):
        self._options = options
        self._idx = source_idx
        self._manager = manager
        self._sim = manager._sim
        self._solver: "RigidSolver | None" = None

    def build(self):
        sim = self._sim
        solver = sim.rigid_solver
        self._solver = solver
        B = sim._B
        K = self._options.audio_substeps
        n_partials = self._options.n_partials
        self._K = K
        self._dt_sub = sim.dt / K
        self._nyquist = 0.45 * 0.5 / self._dt_sub
        self._harm_mult = torch.arange(1, n_partials + 1, dtype=gs.tc_float, device=gs.device)  # (n_partials,)

        entity = sim.entities[self._options.entity_idx]
        joints = entity.joints
        if self._options.joints is not None:
            wanted = set(self._options.joints)
            joints = [j for j in joints if j.name in wanted]

        # One emission point per actuated DOF; radiate from the joint's child link.
        dof_idx, emit_links, rows = [], [], []
        for joint in joints:
            if joint.n_dofs == 0:
                continue
            props = self._options.properties.get(joint.name, self._options.default_properties)
            for d in range(joint.dof_start, joint.dof_end):
                dof_idx.append(d)
                emit_links.append(joint.link.idx)
                rows.append(props)
        n_emit = len(dof_idx)

        self._dof_idx = torch.tensor(dof_idx, dtype=gs.tc_int, device=gs.device)
        self._emit_links_t = torch.tensor(emit_links, dtype=gs.tc_int, device=gs.device)
        self._emit_offset_t = torch.zeros((B, n_emit, 3), dtype=gs.tc_float, device=gs.device)

        def col(attr):
            return torch.tensor([getattr(p, attr) for p in rows], dtype=gs.tc_float, device=gs.device)

        self._pitch_slope = col("pitch_slope")
        self._idle_freq = col("idle_freq")
        self._idle_gain = col("idle_gain")
        self._idle_velocity_gain = col("idle_velocity_gain")
        self._friction_gain = col("friction_gain")
        self._friction_freq = col("friction_freq").clamp(max=self._nyquist)
        self._friction_bw = col("friction_bandwidth").clamp(min=gs.EPS)
        self._load_gain = col("load_gain")
        self._power_gain = col("power_gain")
        self._reversal_click_gain = col("reversal_click_gain")
        self._click_freq = col("click_freq").clamp(max=self._nyquist)
        self._click_decay = col("click_decay").clamp(min=gs.EPS)
        self._slew = col("slew_coeff")
        harm = torch.zeros((n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        idle_harm = torch.zeros((n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        for i, p in enumerate(rows):
            hg = p.harmonic_gains[:n_partials]
            harm[i, : len(hg)] = torch.tensor(hg, dtype=gs.tc_float, device=gs.device)
            ihg = p.idle_harmonic_gains[:n_partials]
            idle_harm[i, : len(ihg)] = torch.tensor(ihg, dtype=gs.tc_float, device=gs.device)
        self._harm = harm  # (n_emit, n_partials)
        self._idle_harm = idle_harm  # (n_emit, n_partials), overtone gains of the idle hum

        # Persistent synthesis state.
        self._phase = torch.zeros((B, n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        self._idle_phase = torch.zeros((B, n_emit, n_partials), dtype=gs.tc_float, device=gs.device)
        self._ty1 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ty2 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ay1 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._ay2 = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._prev_omega = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._slew_tau = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._slew_omega = torch.zeros((B, n_emit), dtype=gs.tc_float, device=gs.device)
        self._block = torch.zeros((B, n_emit, K), dtype=gs.tc_float, device=gs.device)

    @property
    def block(self) -> torch.Tensor:
        return self._block

    @property
    def emit_links(self) -> torch.Tensor:
        return self._emit_links_t

    @property
    def emit_offset(self) -> torch.Tensor:
        return self._emit_offset_t

    def reset(self, envs_idx):
        idx = slice(None) if envs_idx is None else envs_idx
        for t in (
            self._phase,
            self._idle_phase,
            self._ty1,
            self._ty2,
            self._ay1,
            self._ay2,
            self._prev_omega,
            self._slew_tau,
            self._slew_omega,
            self._block,
        ):
            t[idx] = 0.0

    def emit(self):
        solver = self._solver
        if self._block.numel() == 0:
            return
        B, n_emit, K = self._block.shape
        dt_sub = self._dt_sub

        tau = solver.get_dofs_control_force(self._dof_idx)
        omega = solver.get_dofs_velocity(self._dof_idx)
        if solver.n_envs == 0:
            tau, omega = tau[None], omega[None]  # (B, n_emit)

        # One-pole slew limit denoises the controller's per-step torque/velocity ripple.
        a = self._slew
        self._slew_tau += a * (tau - self._slew_tau)
        self._slew_omega += a * (omega - self._slew_omega)
        tau_f, omega_f = self._slew_tau, self._slew_omega

        load_amp = self._load_gain * tau_f.abs() + self._power_gain * (tau_f * omega_f).abs()  # (B, n_emit)
        f0 = (self._pitch_slope * omega_f.abs()).clamp(max=self._nyquist)
        # Static (torque-on) idle floor + a motion-driven term so the hum tracks joint speed instead of droning
        # constantly while the joint merely holds.
        idle_amp = self._idle_gain * (tau_f.abs() > 1e-3).to(gs.tc_float) + self._idle_velocity_gain * omega_f.abs()
        friction_amp = self._friction_gain * omega_f.abs()
        # Reversal click: velocity changed sign since last step (backlash takeup).
        sign_flip = (torch.sign(omega_f) * torch.sign(self._prev_omega)) < 0.0
        click_kick = self._reversal_click_gain * omega_f.abs() * sign_flip.to(gs.tc_float)

        ft1, ft2 = bandpass_coeffs(self._friction_freq, self._friction_bw, dt_sub)  # (n_emit,)
        ac1, ac2 = resonator_coeffs(self._click_freq, self._click_decay, dt_sub)

        phase = self._phase.clone()
        idle_phase = self._idle_phase.clone()
        ty1, ty2 = self._ty1.clone(), self._ty2.clone()
        ay1, ay2 = self._ay1.clone(), self._ay2.clone()
        noise = torch.randn((B, n_emit, K), dtype=gs.tc_float, device=gs.device)
        dphase = 2.0 * torch.pi * f0.unsqueeze(-1) * self._harm_mult * dt_sub  # (B, n_emit, n_partials)
        didle = 2.0 * torch.pi * self._idle_freq.unsqueeze(-1) * self._harm_mult * dt_sub  # (n_emit, n_partials)
        out = torch.empty((B, n_emit, K), dtype=gs.tc_float, device=gs.device)
        for k in range(K):
            phase = phase + dphase
            whine = load_amp * (self._harm * torch.sin(phase)).sum(dim=-1)  # (B, n_emit)

            idle_phase = idle_phase + didle  # (B, n_emit, n_partials)
            idle = idle_amp * (self._idle_harm * torch.sin(idle_phase)).sum(dim=-1)  # (B, n_emit)

            ty = ft1 * ty1 - ft2 * ty2 + friction_amp * noise[:, :, k]
            ty2, ty1 = ty1, ty

            ay = ac1 * ay1 - ac2 * ay2
            if k == 0:
                ay = ay + click_kick
            ay2, ay1 = ay1, ay

            out[:, :, k] = whine + idle + ty + ay

        self._phase.copy_(phase % (2.0 * torch.pi))
        self._idle_phase.copy_(idle_phase % (2.0 * torch.pi))
        self._ty1.copy_(ty1)
        self._ty2.copy_(ty2)
        self._ay1.copy_(ay1)
        self._ay2.copy_(ay2)
        self._prev_omega.copy_(omega_f)
        self._block.copy_(out)


AudioManager.SOURCE_TYPES_MAP[ActuationSourceOptions] = ActuationAudioSource
