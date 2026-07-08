"""
Tests for the audio sensors: ContactAudio realism upgrades (acceleration-noise click, force-coupled damping),
physically-derived modal analysis, active-acoustic excitation, and the airborne SpatialAudio microphone.
"""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import tensor_to_array

DT = 0.005
K = 80  # audio substeps -> 16 kHz at dt=0.005


# ------------------------------------------------------------------------------------------
# ------------------------------------ Modal analysis --------------------------------------
# ------------------------------------------------------------------------------------------


@pytest.mark.required
def test_modal_analysis_material_ordering():
    """Physically-derived modes order correctly across materials: stiffer -> higher pitch, metal rings vs wood dead."""
    import genesis.utils.element as eu
    from genesis.utils.modal_analysis import MATERIAL_PRESETS, compute_modal_model

    verts, elems = eu.box_to_elements(size=(0.2, 0.03, 0.03))
    verts, elems = np.asarray(verts), np.asarray(elems)

    steel = compute_modal_model(verts, elems, MATERIAL_PRESETS["steel"], n_modes=4)
    abs_ = compute_modal_model(verts, elems, MATERIAL_PRESETS["abs"], n_modes=4)
    wood = compute_modal_model(verts, elems, MATERIAL_PRESETS["wood"], n_modes=4)

    # Frequencies are positive and ascending.
    assert np.all(steel.freqs > 0.0)
    assert np.all(np.diff(steel.freqs) >= -1e-3)
    # Stiff steel resonates higher than soft ABS (same mesh) -- driven by sqrt(E/rho).
    assert steel.freqs[0] > abs_.freqs[0]
    # Rayleigh damping: steel rings long (small decay), wood is heavily damped (large decay).
    assert steel.decays[0] < wood.decays[0]
    # from_mesh produces a usable ContactAudioProperties carrying the material's contact damping.
    props = gs.sensors.ContactAudioProperties.from_mesh(verts, elems, "steel", n_modes=3)
    assert len(props.modal_freqs) == len(props.modal_decays) == len(props.modal_gains) == 3
    assert props.contact_damping_per_force == pytest.approx(MATERIAL_PRESETS["steel"].contact_damping_per_force)


# ------------------------------------------------------------------------------------------
# ----------------------------------- SpatialAudio mic -------------------------------------
# ------------------------------------------------------------------------------------------


def _onset(x, thr=0.05):
    a = np.abs(x)
    m = a.max()
    return int(np.argmax(a > thr * m)) if m > 0 else -1


@pytest.mark.required
def test_spatial_audio_propagation(show_viewer):
    """A near and a far airborne mic: energy falls ~1/r^2 and the far onset lags by ~distance/speed_of_sound."""
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    ball = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.5)))

    props = {
        -1: gs.sensors.ContactAudioProperties(
            modal_freqs=(400.0, 1500.0),
            modal_decays=(8.0, 12.0),
            modal_gains=(1.0, 0.6),
            impact_gain=2.0,
            impact_threshold=0.2,
            accel_noise_gain=1.0,
        )
    }
    src = scene.add_sensor(
        gs.sensors.ContactAudio(
            entity_idx=ball.idx,
            link_idx_local=0,
            properties_dict=props,
            audio_substeps=K,
            n_modes=2,
        )
    )
    mic_near = scene.add_sensor(gs.sensors.SpatialAudio(pos_offset=(0.0, 0.0, 0.6), audio_substeps=K))
    mic_far = scene.add_sensor(gs.sensors.SpatialAudio(pos_offset=(3.0, 0.0, 0.6), audio_substeps=K))
    scene.build()

    near, far, srcb = [], [], []
    for _ in range(120):
        scene.step()
        srcb.append(tensor_to_array(src.read()).reshape(-1))
        near.append(tensor_to_array(mic_near.read()).reshape(-1))
        far.append(tensor_to_array(mic_far.read()).reshape(-1))
    near, far, srcb = np.concatenate(near), np.concatenate(far), np.concatenate(srcb)

    en, ef = float(np.sum(near**2)), float(np.sum(far**2))
    assert en > 0.0 and ef > 0.0
    # Inverse (1/r) amplitude -> energy ratio ~ (r_far / r_near)^2. r_near ~ 0.55, r_far ~ 3.05 -> ~30.
    assert 15.0 < en / ef < 60.0
    # Far onset lags the source by ~ r_far / c samples (~142 at 16 kHz); near lags less.
    fs = K / DT
    assert _onset(far) - _onset(srcb) > _onset(near) - _onset(srcb)
    assert abs((_onset(far) - _onset(srcb)) - (3.05 / 343.0) * fs) < 20.0


# ------------------------------------------------------------------------------------------
# --------------------------------- Active acoustic sensing --------------------------------
# ------------------------------------------------------------------------------------------


_SPINNER_URDF = """<?xml version="1.0"?>
<robot name="spinner">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <origin xyz="0.1 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="spin" type="continuous">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


def _spin_actuation(tmp_path, show_viewer, omega_target):
    """Spin a gravity-loaded 1-DOF arm at omega_target; return (actuation waveform, near-mic, far-mic, mean |omega|)."""
    urdf = tmp_path / "spinner.urdf"
    urdf.write_text(_SPINNER_URDF)

    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=show_viewer)
    robot = scene.add_entity(gs.morphs.URDF(file=str(urdf), fixed=True, pos=(0.0, 0.0, 0.0)))
    src = scene.add_audio_source(
        gs.audio.ActuationSource(
            entity_idx=robot.idx,
            audio_substeps=K,
            default_properties=gs.audio.ActuationSourceProperties(
                pitch_slope=50.0,
                harmonic_gains=(1.0,),
                idle_gain=0.0,
                friction_gain=0.0,
                load_gain=1.0,
                power_gain=0.0,
                reversal_click_gain=0.0,
            ),
        )
    )
    mic_near = scene.add_sensor(gs.sensors.SpatialAudio(pos_offset=(0.3, 0.0, 0.5), audio_substeps=K))
    mic_far = scene.add_sensor(gs.sensors.SpatialAudio(pos_offset=(3.0, 0.0, 0.5), audio_substeps=K))
    scene.build()

    robot.set_dofs_kv([20.0], dofs_idx_local=[0])
    act, near, far, omegas = [], [], [], []
    for i in range(160):
        robot.control_dofs_velocity([omega_target], dofs_idx_local=[0])
        scene.step()
        act.append(tensor_to_array(src.block).reshape(-1))  # n_emit == 1 -> mono
        near.append(tensor_to_array(mic_near.read()).reshape(-1))
        far.append(tensor_to_array(mic_far.read()).reshape(-1))
        if i >= 60:
            omegas.append(abs(float(tensor_to_array(robot.get_dofs_velocity([0])).reshape(-1)[0])))
    settle = 60 * K
    return (
        np.concatenate(act)[settle:],
        np.concatenate(near)[settle:],
        np.concatenate(far)[settle:],
        float(np.mean(omegas)),
    )


@pytest.mark.required
def test_actuation_source(tmp_path, show_viewer):
    """ActuationSource whine pitch tracks joint speed, and the airborne mic hears it with distance attenuation."""
    fs = K / DT

    def whine_peak_hz(wave):
        spec = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
        return float(np.fft.rfftfreq(len(wave), 1.0 / fs)[np.argmax(spec)])

    act_lo, near_lo, far_lo, w_lo = _spin_actuation(tmp_path, show_viewer, omega_target=8.0)
    act_hi, near_hi, far_hi, w_hi = _spin_actuation(tmp_path, show_viewer, omega_target=24.0)

    # Source produces sound and the mic picks it up, louder near than far.
    assert np.sum(act_hi**2) > 0.0
    assert np.sum(near_hi**2) > np.sum(far_hi**2)

    # Whine fundamental tracks |omega|: each peak sits near pitch_slope*|omega|, and the faster spin is higher-pitched.
    assert whine_peak_hz(act_lo) == pytest.approx(50.0 * w_lo, rel=0.2)
    assert whine_peak_hz(act_hi) == pytest.approx(50.0 * w_hi, rel=0.2)
    assert whine_peak_hz(act_hi) > 1.5 * whine_peak_hz(act_lo)


def _tap_spectrum(show_viewer, strike_x, use_surface):
    """Tap a fixed aluminium bar at world x=strike_x and return the normalized magnitude spectrum of the contact audio."""
    import genesis.utils.element as eu

    fs = K / DT
    bar_size = (0.4, 0.05, 0.05)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    bar = scene.add_entity(gs.morphs.Box(size=bar_size, pos=(0.0, 0.0, 0.1), fixed=True))
    finger = scene.add_entity(gs.morphs.Box(size=(0.03, 0.03, 0.03), pos=(strike_x, 0.0, 0.2)))

    verts, elems = eu.box_to_elements(size=bar_size)
    mat = gs.sensors.ContactAudioProperties.from_mesh(
        np.asarray(verts), np.asarray(elems), "aluminium", n_modes=6, sample_rate=fs
    )
    mat = mat._replace(roughness_gain=0.0, impact_gain=2.0, impact_threshold=0.1)
    if not use_surface:  # control: drop the surface tables -> flat gains, no position dependence
        mat = mat._replace(surface_points=(), surface_mode_shapes=())
    sensor = scene.add_sensor(
        gs.sensors.ContactAudio(
            entity_idx=finger.idx,
            link_idx_local=0,
            properties_dict={bar.base_link_idx: mat},
            audio_substeps=K,
            n_modes=6,
        )
    )
    scene.build()

    finger.set_dofs_kp(np.full(3, 600.0), dofs_idx_local=slice(0, 3))
    finger.set_dofs_kv(np.full(3, 60.0), dofs_idx_local=slice(0, 3))
    blocks = []
    for i in range(90):
        finger.control_dofs_position(np.array([strike_x, 0.0, 0.132]), dofs_idx_local=slice(0, 3))
        scene.step()
        blocks.append(tensor_to_array(sensor.read()).reshape(-1))
    audio = np.concatenate(blocks)
    spec = np.abs(np.fft.rfft(audio))
    return spec / max(np.linalg.norm(spec), 1e-12)


@pytest.mark.required
def test_position_dependent_timbre(show_viewer):
    """Striking the same bar at different locations changes the timbre only when surface mode shapes are present."""

    def divergence(a, b):  # 0 when spectra identical, grows as their mode mix differs
        return 1.0 - float(np.dot(a, b))

    # With surface data: strike near the end vs. the center -> different mode mix -> spectra diverge.
    d_surface = divergence(
        _tap_spectrum(show_viewer, strike_x=-0.17, use_surface=True),
        _tap_spectrum(show_viewer, strike_x=0.0, use_surface=True),
    )
    # Control: same two strikes with flat gains (no surface data) -> position has no effect -> spectra ~identical.
    d_flat = divergence(
        _tap_spectrum(show_viewer, strike_x=-0.17, use_surface=False),
        _tap_spectrum(show_viewer, strike_x=0.0, use_surface=False),
    )
    # The flat control isolates dynamics-only variation (different contact point/timing) and is small; with surface
    # mode shapes the strike location strongly reshapes the mode mix (observed ~70x the flat baseline).
    assert d_flat < 0.05
    assert d_surface > 0.2
    assert d_surface > 8.0 * max(d_flat, 1e-6)


@pytest.mark.required
def test_active_acoustic_resonance(show_viewer):
    """A swept emitter excitation makes the received signal resonate at the struck object's modal frequencies."""
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=DT), show_viewer=show_viewer)
    scene.add_entity(gs.morphs.Plane())
    tile = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.1), pos=(0.0, 0.0, 0.05), fixed=True),
        material=gs.materials.Rigid(friction=0.8),
    )
    finger = scene.add_entity(
        gs.morphs.Box(size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.128)),
        material=gs.materials.Rigid(friction=0.8),
    )

    modes = (1500.0, 3000.0)
    props = {
        -1: gs.sensors.ContactAudioProperties(modal_freqs=(150.0,), modal_decays=(120.0,), modal_gains=(0.2,)),
        tile.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=modes,
            modal_decays=(6.0, 8.0),
            modal_gains=(1.0, 0.8),
            roughness_gain=0.0,
            impact_gain=0.0,
            contact_damping=40.0,
        ),
    }
    exc = gs.sensors.ExcitationSignal(kind="linear_sweep", f_lo=50.0, f_hi=7000.0, duration=0.5)
    sensor = scene.add_sensor(
        gs.sensors.ContactAudio(
            entity_idx=finger.idx,
            link_idx_local=0,
            properties_dict=props,
            audio_substeps=K,
            n_modes=2,
            excitation=exc,
        )
    )
    scene.build()

    blocks = []
    for _ in range(140):
        scene.step()
        blocks.append(tensor_to_array(sensor.read()).reshape(-1))
    audio = np.concatenate(blocks)

    fs = K / DT
    seg = audio[40 * K :]  # skip the settling transient
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)
    # Each modal frequency should be a clear spectral peak relative to a nearby off-resonance shoulder.
    for mf in modes:
        band = (freqs > mf * 0.85) & (freqs < mf * 1.15)
        shoulder = (freqs > mf * 1.3) & (freqs < mf * 1.6)
        assert spec[band].max() > 3.0 * max(spec[shoulder].mean(), 1e-9)
