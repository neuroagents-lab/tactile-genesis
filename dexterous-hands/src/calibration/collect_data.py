"""Collect real-XHand1 sysid trajectories via the deployment interface.

This script is XHand1-specific: it talks to ``RoboTeraXHandDeployment``,
the only hand deployment implemented today. The sim-twin / identification
side (``sysid_config.py``, ``identify.py``, ``verify.py``,
``scripts/manual_calibration_gui.py``) is robot-agnostic — once another
hand grows a deployment class, this script generalises with it.

Usage
-----

Identification trace (chirp on all 12 joints, 30 s):

    python src/calibration/collect_data.py --side right \
        --excitation chirp --duration 30 --output data/xhand_chirp.npz

Verification trace (held-out PRBS, 20 s):

    python src/calibration/collect_data.py --side right \
        --excitation prbs --duration 20 --output data/xhand_prbs.npz

Amplitude sweep (good for backlash / small-signal behaviour): sinusoid per
DOF whose envelope grows from 1 % to 100 % of the peak excursion implied by
``--dofs-range-ratio``. Use ``--use-current-pos`` so all excitations are applied
around the hand's pose after ``init_sequence()`` (Eden default) instead of the
URDF range midpoint; amplitude vs joint limits is then checked from that pose.

    python src/calibration/collect_data.py --side right \
        --excitation amp_sweep --duration 40 --use-current-pos \
        --output data/xhand_amp_sweep.npz
"""

from __future__ import annotations

import pathlib
import sys
from argparse import BooleanOptionalAction
from typing import Literal, Sequence

import eden as en
import numpy as np
from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY
from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
from eden.extensions.sysid import (
    ChirpExcitation,
    DeploymentRecorder,
    PRBSExcitation,
)
from eden.extensions.sysid.excitation import Excitation
from eden.options.extensions.deployment import DeploymentOptions

from calibration.sysid_config import make_argparser, make_sim_twin_config

DEFAULT_CHIRP_DOFS_RANGE_RATIO = 0.40
DEFAULT_PRBS_DOFS_RANGE_RATIO = 0.20
DEFAULT_AMP_SWEEP_DOFS_RANGE_RATIO = 0.40


class AmpSweepExcitation(Excitation):
    """Per-DOF sine with time-growing envelope (offset relative to recorder center).

    ``g(t) * A[d] * sin(2π f t + φ[d])`` with ``g`` ramping from ``amp_frac_lo``
    to ``amp_frac_hi``. The recorder adds this to either the URDF midpoint or
    the live pose (see ``--use-current-pos``).
    """

    def __init__(
        self,
        num_dofs: int,
        dof_indices: Sequence[int],
        duration: float,
        amplitude: float | Sequence[float] | np.ndarray,
        amp_frac_lo: float,
        amp_frac_hi: float,
        f_hz: float,
        rng: np.random.Generator,
        stagger_phase: float = np.pi,
    ) -> None:
        self.num_dofs = int(num_dofs)
        self.dof_indices = np.asarray(list(dof_indices), dtype=np.int64)
        self._duration = float(duration)
        self.amp_frac_lo = float(amp_frac_lo)
        self.amp_frac_hi = float(amp_frac_hi)
        self.f_hz = float(f_hz)

        amp = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
        if amp.size == 1:
            amp = np.full(self.dof_indices.size, float(amp[0]))
        if amp.size != self.dof_indices.size:
            raise ValueError("amplitude size must be 1 or len(dof_indices).")
        self.amplitude = amp

        n = self.dof_indices.size
        shifts = stagger_phase * (np.arange(n) / max(n, 1))
        phase_extra = rng.uniform(0.0, 2.0 * np.pi, size=n)
        self._phase0 = shifts + phase_extra

    @property
    def duration(self) -> float:
        return self._duration

    def __call__(self, t: float) -> np.ndarray:
        t = max(0.0, min(t, self._duration))
        if self._duration > 0.0:
            g = self.amp_frac_lo + (self.amp_frac_hi - self.amp_frac_lo) * (t / self._duration)
        else:
            g = self.amp_frac_lo
        ang = 2.0 * np.pi * self.f_hz * t + self._phase0
        active = g * self.amplitude * np.sin(ang)
        offsets = np.zeros(self.num_dofs, dtype=np.float64)
        offsets[self.dof_indices] = active
        return offsets


def _build_excitation(
    kind: Literal["chirp", "prbs", "amp_sweep"],
    num_dofs: int,
    duration: float,
    amplitude: np.ndarray,
    seed: int,
    *,
    amp_sweep_min_frac: float = 0.01,
    amp_sweep_max_frac: float = 1.0,
    amp_sweep_f_hz: float = 0.8,
) -> Excitation:
    dof_indices = list(range(num_dofs))
    if kind == "chirp":
        return ChirpExcitation(
            num_dofs=num_dofs,
            dof_indices=dof_indices,
            f_start=0.2,
            f_end=3.0,
            duration=duration,
            amplitude=amplitude,
            stagger_phase=np.pi,  # phase-shift joints so input isn't rank-1
        )
    if kind == "prbs":
        return PRBSExcitation(
            num_dofs=num_dofs,
            dof_indices=dof_indices,
            period=1.0,
            duration=duration,
            amplitude=amplitude,
            seed=seed,
        )
    if kind == "amp_sweep":
        if not 0.0 <= amp_sweep_min_frac <= amp_sweep_max_frac:
            raise ValueError("--amp-sweep-min-frac and --amp-sweep-max-frac must satisfy 0 <= min <= max.")
        if amp_sweep_f_hz <= 0.0:
            raise ValueError("--amp-sweep-f-hz must be positive.")
        rng = np.random.default_rng(seed)
        return AmpSweepExcitation(
            num_dofs=num_dofs,
            dof_indices=dof_indices,
            duration=duration,
            amplitude=amplitude,
            amp_frac_lo=amp_sweep_min_frac,
            amp_frac_hi=amp_sweep_max_frac,
            f_hz=amp_sweep_f_hz,
            rng=rng,
        )
    raise ValueError(f"Unknown excitation: {kind!r}")


def _get_dof_limit_arrays(
    env, entity_name: str, dofs_name: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entity = env.entities[entity_name]
    _, dofs_idx_local = entity.find_named_dofs_idx_local(dofs_name, name_scope=entity.dofs_name, preserve_order=True)
    lower, upper = entity.get_dofs_limit(dofs_idx_local=dofs_idx_local)
    if lower.ndim == 2:
        lower = lower[0]
    if upper.ndim == 2:
        upper = upper[0]
    lower_np = lower.detach().cpu().numpy().astype(np.float64)
    upper_np = upper.detach().cpu().numpy().astype(np.float64)
    raw_range = upper_np - lower_np
    dof_range = np.where(np.isfinite(raw_range), raw_range, 2.0 * np.pi).clip(min=1e-6)
    center = np.where(np.isfinite(lower_np) & np.isfinite(upper_np), 0.5 * (lower_np + upper_np), 0.0)
    return lower_np, upper_np, center, dof_range


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument(
        "--excitation",
        choices=["chirp", "prbs", "amp_sweep"],
        default="chirp",
        help="Excitation signal type. amp_sweep: growing sine envelope (see --amp-sweep-*).",
    )
    parser.add_argument(
        "--amp-sweep-min-frac",
        type=float,
        default=0.01,
        help="For amp_sweep: starting envelope as a fraction of the peak excursion (default 1%%).",
    )
    parser.add_argument(
        "--amp-sweep-max-frac",
        type=float,
        default=1.0,
        help="For amp_sweep: ending envelope fraction (default 100%% of peak).",
    )
    parser.add_argument(
        "--amp-sweep-f-hz",
        type=float,
        default=0.8,
        help="For amp_sweep: sine frequency in Hz on each active DOF.",
    )
    parser.add_argument(
        "--use-current-pos",
        action=BooleanOptionalAction,
        default=False,
        help="Add excitation offsets on top of the hand's joint positions after init (read at recording start) "
        "instead of the URDF range midpoint. Amplitude vs limits is checked from that pose. Chirp, PRBS, and "
        "amp_sweep all honor this.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument(
        "--dofs-range-ratio",
        dest="dofs_range_ratio",
        type=float,
        default=None,
        help=(
            "Total excitation span as a fraction of each joint's URDF position range. "
            "1.0 sweeps from lower to upper around the midpoint. Defaults depend on excitation."
        ),
    )
    parser.add_argument(
        "--control-freq",
        type=float,
        default=None,
        help="Control-loop frequency in Hz.",
    )
    parser.add_argument(
        "--finger-mode",
        choices=[0, 3, 5],
        default=3,
        type=int,
        help="Control mode: 0: powerless, 3: position, 5: powerful.",
    )
    parser.add_argument("--protocol", choices=["RS485", "EtherCAT"], default="RS485")
    parser.add_argument("--serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--hand-id", type=int, default=0)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output npz path for the recorded Trajectory.",
    )
    args = parser.parse_args()

    dofs_range_ratio = args.dofs_range_ratio
    if dofs_range_ratio is None:
        if args.excitation == "chirp":
            dofs_range_ratio = DEFAULT_CHIRP_DOFS_RANGE_RATIO
        elif args.excitation == "prbs":
            dofs_range_ratio = DEFAULT_PRBS_DOFS_RANGE_RATIO
        else:
            dofs_range_ratio = DEFAULT_AMP_SWEEP_DOFS_RANGE_RATIO
    if not 0.0 <= dofs_range_ratio <= 1.0:
        raise SystemExit(f"dofs_range_ratio {dofs_range_ratio} must be in [0.0, 1.0]. Aborting.")

    # A lightweight sim twin is needed because DeploymentBase.__init__ pulls
    # dofs_name / default_dofs_{pos,kp,kd} off the configured entity in the scene.
    en.init(performance_mode=False)
    config = make_sim_twin_config(args.robot, num_envs=1, sim_dt=args.sim_dt, decimation=args.decimation)
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()
    dofs_name = list(config.scene_options.robot.dofs_name)
    dof_lower, dof_upper, dof_center, dof_ranges = _get_dof_limit_arrays(env, "robot", dofs_name)
    amplitude = 0.5 * dofs_range_ratio * dof_ranges
    finite_limits = np.isfinite(dof_lower) & np.isfinite(dof_upper)
    limit_headroom = np.minimum(dof_center - dof_lower, dof_upper - dof_center)
    if not args.use_current_pos:
        too_large_for_limits = finite_limits & (amplitude > limit_headroom + 1e-9)
        if np.any(too_large_for_limits):
            worst = int(np.argmax(np.where(too_large_for_limits, amplitude - limit_headroom, -np.inf)))
            raise SystemExit(
                f"dofs_range_ratio {dofs_range_ratio} exceeds centered joint-limit headroom on {dofs_name[worst]}: "
                f"requested +/-{amplitude[worst]:.3f} rad, available +/-{limit_headroom[worst]:.3f} rad. Aborting."
            )

    # Confirm the backend is registered (also checks SDK availability).
    if "robo_tera_x_hand_deployment" not in DEPLOYMENT_REGISTRY:
        raise SystemExit("RoboTeraXHandDeployment is not registered.")

    deployer = RoboTeraXHandDeployment(
        env=env,
        options=DeploymentOptions(
            entity_name="robot",
            control_freq=args.control_freq,
            finger_mode=args.finger_mode,
        ),
    )
    # Per-hand fields live on the deployer subclass, not on DeploymentOptions.
    deployer.protocol = args.protocol
    deployer.serial_port = args.serial_port
    deployer.hand_id = args.hand_id

    en.logger.info(f"Connecting to XHand1 via {args.protocol} ({args.serial_port}) …")
    deployer.connect()
    try:
        en.logger.info("Running init sequence — robot will hold its default pose.")
        deployer.init_sequence()

        if args.use_current_pos:
            q_pose = np.asarray(deployer.read_state().dofs_pos, dtype=np.float64)
            head_pose = np.minimum(q_pose - dof_lower, dof_upper - q_pose)
            bad_pose = finite_limits & (amplitude > head_pose + 1e-9)
            if np.any(bad_pose):
                worst = int(np.argmax(np.where(bad_pose, amplitude - head_pose, -np.inf)))
                raise SystemExit(
                    f"With --use-current-pos, dofs_range_ratio {dofs_range_ratio} exceeds headroom from the "
                    f"current pose on {dofs_name[worst]}: requested +/-{amplitude[worst]:.3f} rad, "
                    f"available +/-{head_pose[worst]:.3f} rad toward limits. Lower --dofs-range-ratio or reposition."
                )
            center_for_recorder: np.ndarray | None = None
        else:
            center_for_recorder = dof_center

        try:
            excitation = _build_excitation(
                args.excitation,
                num_dofs=deployer.num_dofs,
                duration=args.duration,
                amplitude=amplitude,
                seed=args.seed,
                amp_sweep_min_frac=args.amp_sweep_min_frac,
                amp_sweep_max_frac=args.amp_sweep_max_frac,
                amp_sweep_f_hz=args.amp_sweep_f_hz,
            )
        except ValueError as e:
            raise SystemExit(str(e)) from e
        recorder = DeploymentRecorder(
            deployer=deployer,
            excitation=excitation,
            kp_scale=args.kp_scale,
            kd_scale=args.kd_scale,
            center_dofs_pos=center_for_recorder,
            dofs_limit=(dof_lower, dof_upper),
        )

        pos_note = "center=current_pose" if args.use_current_pos else "center=URDF_midpoint"
        if args.excitation == "amp_sweep":
            en.logger.info(
                f"Recording {args.duration:.1f} s of amp_sweep data "
                f"(dofs_range_ratio={dofs_range_ratio:.3f}, peak_exc={amplitude.min():.3f}–{amplitude.max():.3f} rad, "
                f"envelope {args.amp_sweep_min_frac:.0%}→{args.amp_sweep_max_frac:.0%}, f={args.amp_sweep_f_hz:g} Hz, "
                f"{pos_note}, seed={args.seed})."
            )
        else:
            en.logger.info(
                f"Recording {args.duration:.1f} s of {args.excitation} data "
                f"(dofs_range_ratio={dofs_range_ratio:.3f}, amp={amplitude.min():.3f}–{amplitude.max():.3f} rad, "
                f"{pos_note})."
            )
        trajectory = recorder.run()
    finally:
        deployer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(args.output)
    en.logger.info(f"Saved {len(trajectory)} samples to {args.output} (nominal dt={trajectory.dt * 1000:.2f} ms).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
