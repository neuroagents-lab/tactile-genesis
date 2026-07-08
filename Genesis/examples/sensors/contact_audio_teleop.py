"""
ContactAudio sensor demo (free-rolling fingertip or keyboard teleop) with WAV / MP4 export.

A spherical "fingertip" carries a ContactAudio sensor. By default it is dropped onto the left tile with an initial
forward velocity and bounces/rolls freely across three material tiles (wood, metal, glass) under gravity -- each tile
has different vibroacoustic properties, so the bounce impacts (and the ring-down between them) sound distinct per
material: wood thuds, metal rings, glass pings. Rigid-rigid contact is inelastic in the solver, so the bounce is added
by hand (reflecting the landing velocity by ``BOUNCE_RESTITUTION``). The synthesized contact vibration is accumulated
every step. Pass ``--teleop`` (with ``--vis``) to instead drive the fingertip with the keyboard.

Output is chosen by the ``--out`` file extension:
  - ``.wav``  the synthesized contact audio only.
  - ``.mp4``  a rendered video of the simulation with the contact audio muxed in as its soundtrack (requires a
              camera render each step; the audio track is muxed with the bundled ffmpeg).
A ``<out>_spectrogram.png`` (waveform + log-magnitude spectrogram) is always written alongside the output.

A static airborne ``SpatialAudio`` microphone off to the side also records the contact sound radiated through the air
(distance attenuation + speed-of-sound delay), written to ``<out>_airborne.wav``. With ``--active`` the contact mic
runs in active-acoustic mode (a swept emitter excitation injected into the contacted object), written to
``<out>_active.wav`` -- its spectrogram shows the object's resonances rather than the passive scrape.

Teleop controls (--teleop --vis):
  [up/down/left/right]  move the finger in XY
  [j / k]               lower / raise the finger
  [space]               tap down (quick impact)
  [\\]                   reset finger position
  [esc]                 quit and write the output
"""

import argparse
import os
import subprocess
import tempfile
import wave

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array
from genesis.vis.keybindings import Key, KeyAction, Keybind

DT = 0.005
AUDIO_SUBSTEPS = 160  # 160 samples / 0.005 s = 32 kHz audio (high enough that metal/glass's bright modes survive the
# anti-alias guard, which silences any mode above 0.45 * Nyquist = 7.2 kHz; at 16 kHz it killed everything above 3.6 kHz)
N_MODES = 4

KEY_DPOS = 0.04
KEY_DPOS_Z = 0.01
FORCE_SCALE = 4.0

FINGER_SIZE = 0.05
TILE_SIZE = 0.3
TILE_HEIGHT = 0.1
FINGER_Z0 = TILE_HEIGHT + FINGER_SIZE / 2 + 0.02
ROLL_SPEED = 0.6  # initial +x speed (m/s) of the free fingertip traversing the tiles
DROP_HEIGHT = 0.12  # height (m) above the tile the fingertip is dropped from, so it bounces on landing
# Rigid-rigid contact in the solver is inelastic (no restitution term -- coup_restitution is only for cross-solver
# coupling), so the ball would just thud and stop. We add restitution by hand: on each landing the downward velocity
# is reflected upward scaled by this coefficient, making the ball bounce material-to-material with an airborne ring-
# down between hits. 0 = no bounce (pure roll), <1 = bounces decay, ~1 = nearly elastic.
BOUNCE_RESTITUTION = 0.85


def write_wav(path: str, samples: np.ndarray, sample_rate: int):
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(samples)))
    norm = samples / peak if peak > 1e-9 else samples
    pcm = np.clip(norm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    gs.logger.info(f"Wrote {len(pcm16)} samples ({len(pcm16) / sample_rate:.2f}s @ {sample_rate} Hz) to {path}")


def write_spectrogram(path: str, samples: np.ndarray, sample_rate: int, title: str | None = None):
    """
    Save a log-magnitude spectrogram of synthesized audio (a simple STFT computed with numpy, so no scipy
    dependency). ``title`` labels the plot; defaults to a generic label. Skips gracefully if matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        gs.logger.warning("matplotlib not available, skipping spectrogram.")
        return

    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    nfft, hop = 1024, 256
    window = np.hanning(nfft)
    n_frames = max(1, 1 + (len(samples) - nfft) // hop) if len(samples) >= nfft else 1
    frames = np.zeros((n_frames, nfft))
    for i in range(n_frames):
        seg = samples[i * hop : i * hop + nfft]
        frames[i, : len(seg)] = seg
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    spec_db = 20.0 * np.log10(spec.T + 1e-6)  # (freq, time)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    times = np.arange(n_frames) * hop / sample_rate

    fig, (ax_w, ax_s) = plt.subplots(2, 1, figsize=(10, 4), height_ratios=(1, 3), sharex=True)
    ax_w.plot(np.arange(len(samples)) / sample_rate, samples, lw=0.4, color="0.2")
    ax_w.set_ylabel("amplitude")
    ax_w.set_title(title or "Audio: waveform + spectrogram")
    vmax = spec_db.max()
    im = ax_s.pcolormesh(times, freqs, spec_db, vmin=vmax - 80.0, vmax=vmax, shading="auto", cmap="magma")
    ax_s.set_ylabel("frequency (Hz)")
    ax_s.set_xlabel("time (s)")
    ax_s.set_ylim(0, min(8000.0, sample_rate / 2))
    fig.colorbar(im, ax=ax_s, label="dB")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    gs.logger.info(f"Wrote spectrogram to {path}")


def mux_audio_video(video_path: str, wav_path: str, out_path: str):
    """
    Mux a WAV audio track onto a (silent) video file using the ffmpeg bundled with imageio_ffmpeg.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-i",
        wav_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    gs.logger.info(f"Muxed audio + video to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Interactive ContactAudio sensor demo")
    parser.add_argument("-v", "--vis", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Keyboard teleop: position-control the fingertip with the arrow keys (requires --vis). Without this flag "
        "the fingertip is given an initial velocity and rolls/slides freely across the three tiles under gravity.",
    )
    parser.add_argument("-t", "--seconds", type=float, default=4.0, help="Seconds to simulate")
    parser.add_argument(
        "-o", "--out", type=str, default="contact_audio.wav", help="Output path; .wav (audio) or .mp4 (video+audio)"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate for .mp4 output")
    parser.add_argument(
        "--active",
        action="store_true",
        help="Active-acoustic mode: inject a swept excitation into the contacted object and record the modal "
        "response (Lu & Culbertson). Writes <out>_active.wav.",
    )
    args = parser.parse_args()

    out_ext = os.path.splitext(args.out)[1].lower()
    if out_ext not in (".wav", ".mp4"):
        raise SystemExit(f"--out must end in .wav or .mp4, got '{args.out}'")
    if args.teleop and not args.vis:
        raise SystemExit("--teleop needs the GUI; pass --vis as well (or drop --teleop for the free-roll motion).")
    write_video = out_ext == ".mp4"

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        # Soften the contact (default constraint_timeconst sits at the stiff 2*dt floor). A maximally-stiff contact
        # launches the position-controlled fingertip off the surface every step while sliding, so contact is lost and
        # re-struck ~100x/s -- a real physical chatter that the audio sensor faithfully renders as a buzz. A softer
        # constraint keeps the slide in continuous contact (measured: contact-force std drops from 1.4 N to ~0).
        rigid_options=gs.options.RigidOptions(constraint_timeconst=2 * DT),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -1.2, 1.0),
            camera_lookat=(0.0, 0.0, TILE_HEIGHT),
            camera_fov=40,
            max_FPS=60,
        ),
        # vis_options=gs.options.VisOptions(background_color=(1.0, 1.0, 1.0)),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=args.vis,
    )

    # Three material tiles side by side.
    tile_colors = [(0.6, 0.4, 0.2, 1.0), (0.7, 0.7, 0.75, 1.0), (0.6, 0.8, 0.9, 1.0)]
    tiles = []
    for i, color in enumerate(tile_colors):
        tile = scene.add_entity(
            gs.morphs.Box(
                size=(TILE_SIZE, TILE_SIZE, TILE_HEIGHT),
                pos=((i - 1) * (TILE_SIZE + 0.01), 0.0, TILE_HEIGHT / 2),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=color),
            material=gs.materials.Rigid(friction=0.6),
        )
        tiles.append(tile)
    wood, metal, glass = tiles

    # A rounded (spherical) fingertip, not a box: a sphere makes single-point contact that rolls/slides smoothly,
    # whereas a box's flat face contacts at its corners and the dominant contact point jumps corner-to-corner each
    # step, adding its own contact-force chatter. (radius = FINGER_SIZE/2 so the contact height matches a box bottom.)
    # Teleop starts hovering above the wood tile; free mode starts above wood's left edge to bounce/roll rightward.
    if args.teleop:
        finger_pos_init = np.array([-(TILE_SIZE + 0.01), 0.0, FINGER_Z0], dtype=np.float32)
    else:
        finger_pos_init = np.array(
            [-(TILE_SIZE + 0.01) - 0.4 * TILE_SIZE, 0.0, TILE_HEIGHT + FINGER_SIZE / 2 + DROP_HEIGHT],
            dtype=np.float32,
        )
    finger = scene.add_entity(
        gs.morphs.Sphere(radius=FINGER_SIZE / 2, pos=finger_pos_init),
        surface=gs.surfaces.Default(color=(0.2, 0.2, 0.2, 1.0)),
        material=gs.materials.Rigid(friction=0.6),
    )

    # Vibroacoustic materials keyed by the *struck* tile link. The contact (impact + roll/slide) excites a bank of
    # modal resonators that carry the material identity. contact_damping is kept modest (the modes act as resonant
    # formants while in contact and ring on freely after the contact lifts) -- when it was cranked far higher the modes
    # were over-damped, so every impact collapsed to the same dull click and the slide was undifferentiated noise.
    # wood = low resonances, fast decay (dead knock); metal = high quasi-harmonic resonances, long ring; glass = very
    # high resonances, clean ping. Key -1 is a quiet default (e.g. the ground plane).
    properties_dict = {
        -1: gs.sensors.ContactAudioProperties(
            modal_freqs=(180.0,), modal_decays=(120.0,), modal_gains=(0.3,), roughness_gain=0.0, impact_gain=0.3
        ),
        wood.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(180.0, 420.0, 900.0, 1600.0),
            modal_decays=(40.0, 60.0, 90.0, 130.0),  # fast decay -> dull, dead knock
            modal_gains=(1.0, 0.6, 0.3, 0.15),
            roughness_gain=0.7,  # loud, grainy scrape
            roughness_spatial_freq=1100.0,
            roughness_bandwidth=700.0,
            impact_gain=1.0,
            impact_threshold=0.4,
            contact_damping=120.0,
            accel_noise_gain=0.4,  # low, soft Hertzian knock
            accel_noise_freq=2600.0,
            accel_noise_decay=1400.0,
        ),
        metal.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(1300.0, 2600.0, 4400.0, 6500.0),  # high, quasi-harmonic
            modal_decays=(2.0, 3.0, 4.5, 6.0),  # very slow decay -> long ring after release
            modal_gains=(1.0, 0.75, 0.5, 0.3),
            roughness_gain=1.0,
            roughness_spatial_freq=800.0,
            roughness_bandwidth=1200.0,  # bright, smooth scrape
            impact_gain=1.4,
            impact_threshold=0.4,
            contact_damping=350.0,
            accel_noise_gain=0.8,  # bright metallic "ting" attack
            accel_noise_freq=7000.0,
            accel_noise_decay=500.0,
        ),
        glass.base_link_idx: gs.sensors.ContactAudioProperties(
            modal_freqs=(2500.0, 4300.0, 5800.0, 7000.0),  # very high, clean ping
            modal_decays=(3.0, 4.5, 6.0, 8.0),  # rings, but shorter/cleaner than metal
            modal_gains=(1.0, 0.6, 0.4, 0.25),
            roughness_gain=0.5,  # fine, quiet scrape
            roughness_spatial_freq=600.0,
            roughness_bandwidth=1400.0,
            impact_gain=2.0,
            impact_threshold=0.4,
            contact_damping=500.0,
            accel_noise_gain=1.0,  # crisp, glassy tick
            accel_noise_freq=7200.0,
            accel_noise_decay=450.0,
        ),
    }

    # Active-acoustic mode: a swept emitter excitation injected into the contacted object's modal bank; the received
    # waveform's spectrum then reveals the object's resonances (and how the contact damps them).
    excitation = (
        gs.sensors.ExcitationSignal(kind="linear_sweep", f_lo=80.0, f_hi=7000.0, duration=0.5) if args.active else None
    )
    audio_sensor = scene.add_sensor(
        gs.sensors.ContactAudio(
            entity_idx=finger.idx,
            link_idx_local=0,
            properties_dict=properties_dict,
            audio_substeps=AUDIO_SUBSTEPS,
            n_modes=N_MODES,
            excitation=excitation,
            draw_debug=args.vis,
        )
    )
    # Airborne microphone: a static listener off to the side that hears the finger's contact sound radiated through the
    # air (distance attenuation + speed-of-sound delay). Recorded alongside the contact mic for comparison.
    mic_sensor = scene.add_sensor(
        gs.sensors.SpatialAudio(
            pos_offset=(0.8, -0.8, 0.6),
            audio_substeps=AUDIO_SUBSTEPS,
            draw_debug=args.vis,
        )
    )
    sample_rate = int(round(AUDIO_SUBSTEPS / DT))

    camera = None
    if write_video:
        camera = scene.add_camera(
            res=(960, 720),
            pos=(0.0, -1.2, 1.0),
            lookat=(0.0, 0.0, TILE_HEIGHT),
            fov=40,
            GUI=False,
        )

    scene.build()

    if camera is not None:
        camera.start_recording()

    is_running = True
    target_pos = np.concatenate([finger_pos_init, np.zeros(3, dtype=np.float32)])

    if args.teleop:
        # Teleop: position-control all 6 DOFs (XYZ + the 3 rotational DOFs held at 0). Pinning rotation keeps the
        # fingertip level AND forces it to *slide* rather than roll, so the dragged contact produces slip (scrape
        # texture). Uniform gains over all 6 DOFs keep the slide in steady, chatter-free contact.
        finger.set_dofs_kp(np.full(6, FORCE_SCALE / KEY_DPOS))
        finger.set_dofs_kv(np.full(6, 0.2 * FORCE_SCALE / KEY_DPOS))
        finger.control_dofs_position(target_pos)  # [3:6] stay 0; the teleop callbacks only touch indices [0:3]

        def stop():
            nonlocal is_running
            is_running = False

        def reset_pose():
            target_pos[:3] = finger_pos_init
            target_pos[3:] = 0.0
            finger.set_dofs_position(target_pos)

        def translate(index: int, is_negative: bool):
            target_pos[index] += (-1 if is_negative else 1) * (KEY_DPOS if index < 2 else KEY_DPOS_Z)

        def tap():
            target_pos[2] = TILE_HEIGHT + FINGER_SIZE / 2 - 0.005

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=translate, args=(1, False)),
            Keybind("move_backward", Key.DOWN, KeyAction.HOLD, callback=translate, args=(1, True)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=translate, args=(0, False)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=translate, args=(0, True)),
            Keybind("lower", Key.J, KeyAction.HOLD, callback=translate, args=(2, True)),
            Keybind("raise", Key.K, KeyAction.HOLD, callback=translate, args=(2, False)),
            Keybind("tap", Key.SPACE, KeyAction.PRESS, callback=tap),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
    else:
        # Free mode: drop the fingertip from a height onto the left (wood) tile with an initial +x velocity and let it
        # bounce / roll across all three tiles under gravity. No position control -> no controlled-contact chatter; the
        # manual restitution in the loop turns each landing into a clean material impact with an airborne ring-down.
        finger.set_dofs_velocity(np.array([ROLL_SPEED, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        tiles_right_edge = (TILE_SIZE + 0.01) + TILE_SIZE / 2  # x past which the ball has run off the last (glass) tile
        bounce_in_contact = False
        prev_vz = 0.0
        finger_x = finger_pos_init[0]

    print("\n=== ContactAudio demo ===")
    print(f"Audio: {AUDIO_SUBSTEPS} samples/step @ dt={DT}s  ->  {sample_rate} Hz")
    print("Tiles (left to right): wood | metal | glass")
    if args.teleop:
        print("Teleop: [arrows] move XY  [j/k] lower/raise  [space] tap  [\\] reset  [esc] quit")
    else:
        print(
            f"Free bounce: fingertip is dropped on wood and bounces left->right across the tiles (<= {args.seconds}s)"
        )
    print()

    audio_blocks: list[np.ndarray] = []
    mic_blocks: list[np.ndarray] = []
    n_steps = int(args.seconds / DT)

    # Render a frame every `render_every` steps so the video plays back in real time (and stays in sync with the
    # audio). effective_fps is the real-time frame rate handed to the encoder.
    render_every = max(1, round((1.0 / DT) / args.fps)) if write_video else 0
    effective_fps = (1.0 / DT) / render_every if write_video else args.fps

    try:
        step = 0
        while is_running:
            if args.teleop:
                finger.control_dofs_position(target_pos)
            scene.step()

            if args.teleop:
                cur = tensor_to_array(finger.get_pos())
                target_pos[:2] = np.clip(target_pos[:2] - cur[:2], -KEY_DPOS, KEY_DPOS) + cur[:2]
            else:
                # Manual restitution (the rigid solver is inelastic): on the step the descending ball *first* makes
                # contact, reflect its vertical velocity upward by BOUNCE_RESTITUTION so it bounces on to the next
                # material rather than sticking. Detecting the fresh contact-force onset (not a penetration depth) fires
                # at the true impact instant, before the soft contact has bled off the incoming speed.
                finger_x = float(tensor_to_array(finger.get_pos()).reshape(-1)[0])
                vz = float(tensor_to_array(finger.get_vel()).reshape(-1)[2])
                contact_force = scene.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)["force"]
                touching = contact_force.numel() > 0 and float(contact_force.norm(dim=-1).sum()) > 1e-4
                if touching and not bounce_in_contact and prev_vz < -0.1:
                    finger.set_dofs_velocity(
                        np.array([BOUNCE_RESTITUTION * -prev_vz], dtype=np.float32), dofs_idx_local=[2]
                    )
                bounce_in_contact = touching
                prev_vz = vz

            audio_blocks.append(tensor_to_array(audio_sensor.read()).reshape(-1))
            mic_blocks.append(tensor_to_array(mic_sensor.read()).reshape(-1))
            if camera is not None and step % render_every == 0:
                camera.render()

            step += 1
            if "PYTEST_VERSION" in os.environ and step >= 5:
                break
            # Teleop runs until [esc]; the free bounce/roll ends at the time limit or once the ball leaves the tiles
            # (stopping there avoids recording the long silence -- and the hard drop -- after it rolls off the edge).
            if not args.teleop and (step >= n_steps or finger_x > tiles_right_edge):
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        if audio_blocks:
            audio = np.concatenate(audio_blocks)
            if write_video:
                with tempfile.TemporaryDirectory() as tmp:
                    wav_tmp = os.path.join(tmp, "audio.wav")
                    video_tmp = os.path.join(tmp, "video.mp4")
                    write_wav(wav_tmp, audio, sample_rate)
                    camera.stop_recording(save_to_filename=video_tmp, fps=effective_fps)
                    mux_audio_video(video_tmp, wav_tmp, args.out)
            else:
                write_wav(args.out, audio, sample_rate)
            write_spectrogram(
                os.path.splitext(args.out)[0] + "_spectrogram.png",
                audio,
                sample_rate,
                title="ContactAudio: waveform + spectrogram (wood | metal | glass)",
            )

            base = os.path.splitext(args.out)[0]
            if args.active:
                # The contact-mic recording is the active-acoustic response; save it under a clear name too.
                write_wav(base + "_active.wav", audio, sample_rate)
            if mic_blocks:
                mic_audio = np.concatenate(mic_blocks)
                write_wav(base + "_airborne.wav", mic_audio, sample_rate)
                write_spectrogram(
                    base + "_airborne_spectrogram.png",
                    mic_audio,
                    sample_rate,
                    title="SpatialAudio (airborne mic): waveform + spectrogram",
                )
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
