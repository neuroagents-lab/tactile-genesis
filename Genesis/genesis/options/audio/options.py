from typing import Any, NamedTuple

from pydantic import Field, StrictInt

import genesis as gs
from genesis.typing import PositiveFloat, PositiveInt

from ..options import Options


class AudioSourceOptions(Options):
    """
    Base options for audio *sources* -- generators added with ``scene.add_audio_source(...)`` (the synthesis stage).
    Sources are not sensors: a receiver sensor (``SpatialAudio`` / microphone) renders them.

    Parameters
    ----------
    entity_idx : int
        Global entity index this source is attached to.
    audio_substeps : int
        Number of synthesized samples emitted per physics step (the carrier upsampling factor K). Must match the
        microphone(s) rendering it. Default ``20``.
    """

    entity_idx: StrictInt = Field(default=-1, ge=-1)
    audio_substeps: PositiveInt = 20


class ActuationSourceProperties(NamedTuple):
    """
    Per-joint actuation-noise timbre for the ``ActuationSource``. Models the sound a motor/gearbox at a joint radiates
    as a function of its torque/effort and speed: a velocity-pitched whine (gear mesh / commutation), an idle
    electrical hum, velocity-scaled bearing/friction hiss, and a transient click on direction reversal.

    Parameters
    ----------
    pitch_slope : float
        Whine fundamental frequency in Hz per rad/s of joint speed (bundles gear ratio x mesh teeth). The whine pitch
        is ``pitch_slope * |omega|``. Default ``120``.
    harmonic_gains : tuple[float, ...]
        Relative gains of the whine partials (fundamental, 2x, 3x, ...). Truncated / zero-padded to ``n_partials``.
        Default ``(1.0, 0.5, 0.25)``.
    idle_freq : float
        Frequency (Hz) of the idle electrical hum (PWM / magnetostriction). Default ``120``.
    idle_gain : float
        Amplitude of the idle hum while the motor is energized (``|tau| > 0``). ``0`` disables it. Default ``0``.
        This is the *static* component: with ``|tau|`` roughly constant while the joint holds, it produces a
        constant tone. Pair with ``idle_velocity_gain`` (and a small/zero ``idle_gain``) to drive the hum by motion
        instead, so a held joint is quiet rather than droning.
    idle_velocity_gain : float
        Adds a joint-speed-proportional term ``idle_velocity_gain * |omega|`` to the idle hum amplitude, so the hum's
        fixed-pitch lines are driven by *motion* rather than just holding torque. Keeps the hum quiet while the joint
        is still (no constant ambient drone) and makes faster motion louder. Default ``0``.
    idle_harmonic_gains : tuple[float, ...]
        Relative gains of the idle hum's partials (fundamental, 2x, 3x, ...) at ``idle_freq``. Truncated /
        zero-padded to ``n_partials`` (shares the whine's partial count). A real servo hum is harmonic-rich, so
        ``(1.0, 0.6, 0.4)`` etc. reproduces the buzzy overtone stack rather than a pure sine. Default ``(1.0,)``
        (a single sine, matching the previous behavior).
    friction_gain : float
        Amplitude of the velocity-scaled bearing/friction hiss (broadband). ``0`` disables it. Default ``0``.
    friction_freq : float
        Center frequency (Hz) of the friction noise band. Default ``1200``.
    friction_bandwidth : float
        Spectral width (Hz) of the friction noise band. Default ``800``.
    load_gain : float
        Loudness per unit |torque| (1/N·m): the holding/stall component (a motor straining in place still hums).
        Default ``0.2``.
    power_gain : float
        Loudness per unit mechanical power |tau*omega| (1/W): the working-whine component. Default ``0.1``.
    reversal_click_gain : float
        Amplitude of the transient click emitted when joint velocity changes sign (backlash takeup). Scaled by
        ``|omega|``. ``0`` disables it. Default ``0``.
    click_freq : float
        Center frequency (Hz) of the reversal click resonator. Default ``3000``.
    click_decay : float
        Decay rate (1/s) of the reversal click resonator. Default ``900``.
    slew_coeff : float
        One-pole smoothing coefficient in ``(0, 1]`` applied to torque/velocity before synthesis, to denoise the
        controller's per-step ripple (1 = no smoothing, smaller = heavier). Default ``0.3``.
    """

    pitch_slope: float = 120.0
    harmonic_gains: tuple[float, ...] = (1.0, 0.5, 0.25)
    idle_freq: float = 120.0
    idle_gain: float = 0.0
    idle_velocity_gain: float = 0.0
    idle_harmonic_gains: tuple[float, ...] = (1.0,)
    friction_gain: float = 0.0
    friction_freq: float = 1200.0
    friction_bandwidth: float = 800.0
    load_gain: float = 0.2
    power_gain: float = 0.1
    reversal_click_gain: float = 0.0
    click_freq: float = 3000.0
    click_decay: float = 900.0
    slew_coeff: float = 0.3


class ActuationSource(AudioSourceOptions):
    """
    Motor/joint actuation-noise source: synthesizes the sound a robot's actuators radiate from the joints' torque and
    velocity. One emission point per covered DOF, radiating from that joint's child link, so a microphone hears the
    arm's motors located correctly in space.

    Parameters
    ----------
    entity_idx : int
        Global entity index of the actuated robot.
    joints : tuple[str, ...] | None
        Names of the joints to voice. ``None`` (default) covers every non-fixed joint of the entity.
    default_properties : ActuationSourceProperties
        Timbre applied to every covered joint unless overridden in ``properties``.
    properties : dict[str, ActuationSourceProperties]
        Per-joint-name timbre overrides.
    n_partials : int
        Number of whine partials in the oscillator bank. Default ``3``.
    audio_substeps : int
        Samples per physics step; must match the rendering microphone. Default ``20``.
    """

    joints: tuple[str, ...] | None = None
    default_properties: ActuationSourceProperties = ActuationSourceProperties()
    properties: dict[str, ActuationSourceProperties] = Field(default_factory=dict)
    n_partials: PositiveInt = 3

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.entity_idx < 0:
            gs.raise_exception("ActuationSource requires entity_idx >= 0 (the actuated entity).")
