from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
from pydantic import Field, StrictBool

import genesis as gs
from genesis.typing import PositiveFloat, PositiveInt

from .options import KinematicSensorOptionsMixin, RigidSensorOptionsMixin, SimpleSensorOptions

if TYPE_CHECKING:
    from genesis.engine.sensors.contact_audio import ContactAudioSensor
    from genesis.engine.sensors.spatial_audio import SpatialAudioSensor


class ContactAudioProperties(NamedTuple):
    """
    Vibroacoustic material descriptor for an object that is *being contacted*, used by the ``ContactAudio`` sensor to
    synthesize contact vibration. Modal terms are the object's resonant signature (impact ring-down); roughness terms
    are the sliding-texture source.

    The source-filter split: ``ContactAudio`` excites a bank of damped modal oscillators (the filter) with the
    solver contact-force onset (impact) and a velocity-scaled noise source (sliding texture). Metal-like materials use
    high frequencies with slow decay (high-Q ringing); wood-like materials use lower frequencies with fast decay plus
    a stronger roughness source.

    Parameters
    ----------
    modal_freqs : tuple[float, ...]
        Resonant mode frequencies in Hz. One entry per mode; shorter than the sensor's ``n_modes`` is zero-padded.
    modal_decays : tuple[float, ...]
        Per-mode amplitude decay rate in 1/s (large = fast decay = wood-like; small = long ring = metal-like). Must
        match the length of ``modal_freqs``.
    modal_gains : tuple[float, ...]
        Per-mode output weight. Must match the length of ``modal_freqs``.
    roughness_gain : float
        Amplitude of the sliding-texture noise source. Scales with normal force and slip speed at runtime. ``0``
        disables texture (smooth surface). Default ``0``.
    roughness_spatial_freq : float
        Surface roughness spatial frequency in bumps per meter. The texture's temporal pitch is
        ``roughness_spatial_freq * slip_speed`` (Hz), so faster sliding raises the pitch. Default ``0``.
    roughness_bandwidth : float
        Spectral width (Hz) of the sliding-texture noise source. This is the bandwidth of a broadband band-pass
        centered on the slip-dependent pitch; larger values give a noisier "shhh/scrrr" scrape, smaller values an
        unrealistic tone. Use several hundred Hz for a realistic scrape. Default ``600``.
    impact_gain : float
        Scale applied to the (transient-gated) contact-force onset impulse that excites the modal bank on a tap.
        Default ``1``.
    impact_threshold : float
        Minimum positive force jump (Newtons, per physics step) that counts as a tap and injects a modal impulse.
        Steady-sliding force ripple stays below this, so sliding does not re-ping the modes into a sustained tone;
        only a sharp onset (an actual strike) excites the ring-down. Default ``0.5``.
    contact_damping : float
        Extra modal decay (1/s) added to ``modal_decays`` *while the surface is in contact*. A finger pressing on an
        object mass-loads and damps its modes, so they should not ring freely during a slide; the long free ring-down
        appears only after release (when this term is removed). Larger values give a more deadened in-contact sound.
        This is the force-independent floor; see ``contact_damping_per_force`` for the force-coupled term. Default
        ``80``.
    contact_damping_per_force : float
        Force-coupled in-contact modal decay in ``1/(s·N)``: the extra decay while in contact is
        ``contact_damping + contact_damping_per_force * f_normal``. This is the cheap per-mode form of the
        contact-dependent viscous damping of Zheng & James 2011 (damping proportional to contact force), which
        reproduces the coffee-mug effect (a firmer press deadens the ring more than a light touch). Default ``0``
        (back-compatible: in-contact damping is the constant ``contact_damping``).
    accel_noise_gain : float
        Amplitude of the acceleration-noise "click" injected on a tap (a sharp, fast-decaying broadband burst that
        models the Hertzian contact transient of small hard objects, which the slow modal ring-down misses; cf. the
        acceleration-noise shader of Wang et al. 2018 / Chadwick et al.). Scales the impact impulse into a dedicated
        high-frequency, fast-decay resonator. ``0`` disables the click. Default ``0``.
    accel_noise_freq : float
        Center frequency (Hz) of the acceleration-noise click resonator. Default ``5000``.
    accel_noise_decay : float
        Decay rate (1/s) of the acceleration-noise click resonator. Large = a short snappy click. Default ``800``.
    surface_points : tuple[tuple[float, float, float], ...]
        Surface sample positions in the struck object's *link-local* frame (meters), shape ``(n_surface, 3)``. With
        ``surface_mode_shapes`` these make the timbre depend on *where* the object is struck (a mode is silent at its
        node, loud at its antinode; van den Doel, Zheng & James). Populated by ``from_mesh``; empty disables
        position dependence (flat ``modal_gains`` are used everywhere). Default ``()``.
    surface_mode_shapes : tuple[tuple[float, ...], ...]
        Per-surface-point, per-mode normalized mode-shape amplitude in ``[-1, 1]`` (1 at each mode's antinode), shape
        ``(n_surface, n_modes)`` aligned with ``surface_points`` / ``modal_freqs``. Default ``()``.
    """

    modal_freqs: tuple[float, ...] = (250.0,)
    modal_decays: tuple[float, ...] = (40.0,)
    modal_gains: tuple[float, ...] = (1.0,)
    roughness_gain: float = 0.0
    roughness_spatial_freq: float = 0.0
    roughness_bandwidth: float = 600.0
    impact_gain: float = 1.0
    impact_threshold: float = 0.5
    contact_damping: float = 80.0
    contact_damping_per_force: float = 0.0
    accel_noise_gain: float = 0.0
    accel_noise_freq: float = 5000.0
    accel_noise_decay: float = 800.0
    surface_points: tuple = ()
    surface_mode_shapes: tuple = ()

    @classmethod
    def from_mesh(cls, verts, elems, material, n_modes: int = 8, sample_rate: float | None = None, **overrides):
        """
        Build physically-derived modal properties from a tetrahedral mesh and an isotropic material via linear modal
        analysis (see ``genesis.utils.modal_analysis``), instead of hand-tuning ``modal_freqs/decays/gains``.

        Parameters
        ----------
        verts : array-like, shape (N, 3)
            Tetrahedral mesh vertices in meters (e.g. from ``genesis.utils.modal_analysis.tetrahedralize``).
        elems : array-like, shape (T, 4)
            Tetrahedra (vertex indices).
        material : str | genesis.utils.modal_analysis.Material
            A key of ``MATERIAL_PRESETS`` (e.g. ``"steel"``) or an explicit ``Material``.
        n_modes : int
            Number of modes to extract (rigid-body modes are skipped). Default ``8``.
        sample_rate : float, optional
            If given, modes above the carrier band edge are dropped (anti-aliasing).
        overrides
            Any remaining ``ContactAudioProperties`` fields to set (e.g. ``roughness_gain``, ``impact_gain``).
        """
        from genesis.utils.modal_analysis import MATERIAL_PRESETS, compute_modal_model

        if isinstance(material, str):
            material = MATERIAL_PRESETS[material]
        model = compute_modal_model(verts, elems, material, n_modes, sample_rate)
        # Normalize each mode's raw surface amplitude to a [-1, 1] shape (1 at its antinode) so position weighting
        # modulates the flat modal_gains rather than rescaling overall loudness.
        sg = np.asarray(model.surface_gains, dtype=np.float64)
        shapes = sg / np.maximum(np.abs(sg).max(axis=0, keepdims=True), 1e-12)
        props = cls(
            modal_freqs=tuple(float(f) for f in model.freqs),
            modal_decays=tuple(float(d) for d in model.decays),
            modal_gains=tuple(float(g) for g in model.gains),
            contact_damping_per_force=float(material.contact_damping_per_force),
            surface_points=tuple(tuple(float(c) for c in p) for p in model.surface_points.tolist()),
            surface_mode_shapes=tuple(tuple(float(s) for s in row) for row in shapes.tolist()),
        )
        return props._replace(**overrides) if overrides else props


class ExcitationSignal(NamedTuple):
    """
    Active-acoustic excitation injected by an emitter into the grasped object's modal bank (Lu & Culbertson 2023). The
    receiver ``ContactAudio`` sensor records the modal response, whose spectrum encodes the object's resonances and how
    contact formations damp them.

    Parameters
    ----------
    kind : str
        ``"impulse"`` (a click each period), ``"linear_sweep"`` (frequency rises linearly), or ``"exp_sweep"``
        (frequency rises exponentially, emphasizing low frequencies). Default ``"linear_sweep"``.
    f_lo : float
        Sweep start frequency in Hz. Default ``20``.
    f_hi : float
        Sweep end frequency in Hz. Default ``10000``.
    duration : float
        Sweep duration in seconds (one pass low->high). Default ``0.5``.
    amplitude : float
        Drive amplitude. Default ``1``.
    period : float
        Repeat interval in seconds; the excitation restarts every ``period`` (``<= 0`` uses ``duration``, i.e. it loops
        back-to-back). Default ``0`` (loop).
    """

    kind: str = "linear_sweep"
    f_lo: float = 20.0
    f_hi: float = 10000.0
    duration: float = 0.5
    amplitude: float = 1.0
    period: float = 0.0


class ContactAudio(RigidSensorOptionsMixin["ContactAudioSensor"], SimpleSensorOptions["ContactAudioSensor"]):
    """
    Link-level contact vibration / audio sensor.

    Reads the rigid solver's contact forces on the attached link and the relative velocity at the contact, then
    synthesizes a high-rate vibration waveform via source-filter modal synthesis: the contact-force onset excites a
    bank of damped modal oscillators (impact ring-down) and a velocity-scaled noise source drives a texture resonator
    (sliding roughness). The timbre is keyed by the material of the *struck* link via ``properties_dict``.

    Each ``scene.step()`` emits a block of ``audio_substeps`` samples (the physics step is the slow envelope; the
    block synthesis runs above the physics Nyquist), so the effective sample rate is ``audio_substeps / dt`` Hz. The
    ``read()`` output has shape ``(audio_substeps,)`` per environment; concatenating blocks across steps yields a
    continuous waveform suitable for the Pacinian band or for writing to an audio file.

    Note
    ----
    The synthesized signal is a single (mono) normal-acceleration-like channel per sensor. Vibration is a whole-body
    phenomenon, so it is reported per link rather than per taxel.

    Parameters
    ----------
    properties_dict : dict[int, ContactAudioProperties]
        Maps a *struck* link index (the object in contact, not the sensor's own link) to its vibroacoustic material.
        Key ``-1`` is the default for links not present in the dict; if omitted, contacts with unlisted links
        generate no sound. Shared across all ``ContactAudio`` sensors (dicts are merged).
    audio_substeps : int
        Number of synthesized samples emitted per physics step (the carrier upsampling factor K). The effective
        audio sample rate is ``audio_substeps / dt`` Hz. Default ``20``.
    n_modes : int
        Size of the modal oscillator bank. Materials with fewer modes are zero-padded. Shared across all
        ``ContactAudio`` sensors. Default ``8``.
    excitation : ExcitationSignal, optional
        If set, the sensor runs in *active-acoustic* mode (Lu & Culbertson 2023): while the sensor link is in contact,
        the given excitation is injected into the struck object's modal bank and the synthesized output is the modal
        response (the "received" waveform), whose spectrum reveals the object's resonances and how the contact damps
        them. Combine with ``roughness_gain=0`` to isolate the active response from passive scrape noise. Default
        ``None`` (passive contact-mic mode).
    velocity_gate_ref : float
        If ``> 0``, scale the synthesized output by a soft gate that tracks how fast the *sensor's own link* (the
        attached body) is moving, so the sensor goes quiet when that body is nearly still. The contact-force synthesis
        alone clicks on every tap/regrip even when the body barely moves; this gate suppresses those when the body is
        not actually in motion. The gate gain is ``motion / (motion + velocity_gate_ref)`` (0 at rest, 0.5 at
        ``velocity_gate_ref``, ->1 when fast), where ``motion = |linear_vel| + velocity_gate_ang_weight*|angular_vel|``
        of the sensor link (m/s-equivalent). ``0`` (default) disables the gate entirely (unchanged behavior).
    velocity_gate_ang_weight : float
        Weight converting the sensor link's angular speed (rad/s) to a linear-equivalent (m/s) in the gate's motion
        metric, roughly the body's radius. Only used when ``velocity_gate_ref > 0``. Default ``0``.
    velocity_gate_smooth : float
        One-pole smoothing coefficient in ``(0, 1]`` applied to the gate gain across physics steps (1 = no smoothing,
        smaller = slower/heavier) so the gain cannot step abruptly and create its own clicks. Only used when
        ``velocity_gate_ref > 0``. Default ``1.0``.
    """

    properties_dict: dict[int, ContactAudioProperties] = Field(default_factory=dict)
    audio_substeps: PositiveInt = 20
    n_modes: PositiveInt = 8
    excitation: ExcitationSignal | None = None
    velocity_gate_ref: float = Field(default=0.0, ge=0.0)
    velocity_gate_ang_weight: float = Field(default=0.0, ge=0.0)
    velocity_gate_smooth: float = Field(default=1.0, gt=0.0, le=1.0)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.excitation is not None and self.excitation.kind not in ("impulse", "linear_sweep", "exp_sweep"):
            gs.raise_exception(
                f"ExcitationSignal.kind must be 'impulse', 'linear_sweep', or 'exp_sweep', got "
                f"'{self.excitation.kind}'."
            )
        for link_idx, props in self.properties_dict.items():
            if not (len(props.modal_freqs) == len(props.modal_decays) == len(props.modal_gains)):
                gs.raise_exception(
                    f"ContactAudioProperties for link {link_idx}: modal_freqs, modal_decays, and modal_gains must "
                    f"have equal length. Got {len(props.modal_freqs)}, {len(props.modal_decays)}, "
                    f"{len(props.modal_gains)}."
                )
            if len(props.modal_freqs) > self.n_modes:
                gs.raise_exception(
                    f"ContactAudioProperties for link {link_idx} has {len(props.modal_freqs)} modes, exceeding "
                    f"n_modes={self.n_modes}."
                )


class SpatialAudio(KinematicSensorOptionsMixin["SpatialAudioSensor"], SimpleSensorOptions["SpatialAudioSensor"]):
    """
    Airborne point-microphone sensor: a mono listener in world space that renders the airborne sound radiated by the
    scene's ``ContactAudio`` (contact-mic) sensors, with geometric propagation -- distance attenuation plus a
    speed-of-sound delay (and the Doppler shift a changing delay implies).

    The listener is either *static* (``entity_idx < 0``: fixed at ``pos_offset`` in world frame) or *attached* to a
    link (``entity_idx >= 0``: riding at ``link_pos + pos_offset``, e.g. a head). Every ``ContactAudio`` sensor in the
    scene is treated as a point radiation source located at its sensor link, and its synthesized structure-borne block
    is reused as the radiated signal. This is a deliberate batched approximation -- radiated pressure is taken
    proportional to surface normal acceleration (the Neumann boundary intuition of Wang et al. 2018), not a true
    radiation/directivity model. The source block is consumed one physics step late, so the mic is independent of
    sensor step order.

    Each ``scene.step()`` emits a block of ``audio_substeps`` samples (which must equal the ``ContactAudio`` sources'
    ``audio_substeps``); ``read()`` returns shape ``(audio_substeps,)`` per environment, concatenable into a continuous
    waveform exactly like ``ContactAudio``.

    Parameters
    ----------
    audio_substeps : int
        Samples emitted per physics step; must equal the ``ContactAudio`` sources' ``audio_substeps``. Default ``20``.
    speed_of_sound : float
        Propagation speed (m/s) for the source->listener delay. Default ``343``.
    ref_distance : float
        Distance (m) at which attenuation is unity, and the near-field rolloff floor (gain is clamped for
        ``r < ref_distance`` to avoid the ``1/r`` singularity). Default ``0.1``.
    attenuation : str
        Distance rolloff law: ``"inverse"`` (1/r) or ``"inverse_square"`` (1/r^2). Default ``"inverse"``.
    enable_doppler : bool
        If True, ramp the propagation delay across each block from last step's value to this step's, so a moving
        source or listener produces a Doppler pitch shift (and block-boundary discontinuities are avoided). Default
        ``True``.
    max_delay : float
        Maximum modeled propagation delay (s); sizes the internal source-history buffer and clamps larger delays. Set
        to at least ``max_source_distance / speed_of_sound``. Default ``0.03`` (~10 m at 343 m/s).
    enable_occlusion : bool
        Reserved for raycast-based occlusion; not yet implemented (raises if set True). Default ``False``.
    """

    audio_substeps: PositiveInt = 20
    speed_of_sound: PositiveFloat = 343.0
    ref_distance: PositiveFloat = 0.1
    attenuation: str = "inverse"
    enable_doppler: StrictBool = True
    max_delay: PositiveFloat = 0.03
    enable_occlusion: StrictBool = False

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.attenuation not in ("inverse", "inverse_square"):
            gs.raise_exception(
                f"SpatialAudio attenuation must be 'inverse' or 'inverse_square', got '{self.attenuation}'."
            )
        if self.enable_occlusion:
            gs.raise_exception("SpatialAudio enable_occlusion is not yet implemented.")
