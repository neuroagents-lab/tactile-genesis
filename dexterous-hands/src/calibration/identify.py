"""Identify dexterous-hand DOF parameters from recorded real-hand trajectories.

Robot-agnostic: ``--robot`` selects which registered hand the sim twin is
built for (default ``xhand1``). Trajectories must have been recorded from
that hand's real hardware.

Usage
-----

Basic run on one chirp trace:

    python src/calibration/identify.py \
        --robot xhand1 \
        --trajectory data/xhand_chirp.npz \
        --output results/xhand_sysid/

Multiple traces (jointly fit):

    python src/calibration/identify.py \
        --trajectory data/xhand_chirp.npz data/xhand_prbs_train.npz \
        --output results/xhand_sysid/

What is identified
------------------
By default, per-joint ``damping``, ``armature``, ``frictionloss``, ``kp``,
``kd``, and the PD ``HAND_CONTROLLER`` modifiers: ``deadband_epsilon``,
GearBacklash fields (``gear_backlash``, …), ``motor_strength``,
ConstantTorqueKick fields (``torque_kick``, ``activation_epsilon``),
T-N limits (``driving_torque_limit``, …), and
``FrictionModel`` channels (``friction_static``, …). ``stiffness`` is left
out unless added with ``--include stiffness``. Narrow ``--include`` to skip
any default group.

The residual uses ``dofs_pos`` only. Both ``dofs_vel`` and ``dofs_torque``
are **excluded** because the XHand1 SDK does not report them (see
``RoboTeraXHandDeployment.read_state`` — both fields are zero-filled).
Including either would contaminate the normalised residual with spurious
zeros.

Parallelism
-----------
- ``--optimizer scipy`` (default): serial Trust-Region Reflective, one
  rollout per FD column per iteration. ``--num-envs 1`` is enough.
- ``--optimizer scipy_parallel_fd``: compute all FD columns in one batched
  replay. Requires ``--num-envs (n_params + 1)`` (e.g. 181 when the default
  ``--include`` fits 15 groups × 12 joints = 180 scalars).
- ``--optimizer cmaes``: CMA-ES with batched candidate evaluation.
  ``--num-envs`` sets the population size — tune to your GPU budget
  (16–4096 are all reasonable, larger pops trade compute for fewer
  generations).
"""

from __future__ import annotations

import glob
import pathlib
import sys
import time
from argparse import BooleanOptionalAction
from typing import Any, Sequence, cast

import eden as en
import genesis as gs
import numpy as np
from eden.extensions.sysid import Parameter, Trajectory
from eden.extensions.sysid.base import FitResult
from eden.extensions.sysid.report import signal_plots, write_summary
from eden.extensions.sysid.residual import multi_trajectory_residual

from calibration.action_mod_sysid import PROPERTIES, install_action_mod_sysid_patch
from calibration.sysid_config import ParameterSet, make_argparser, make_sim_twin_config
from calibration.sysid_rollout import batched_candidate_rollout, single_candidate_rollout


def _build_parameters(
    env,
    dof_names: Sequence[str],
    properties: Sequence[str],
    entity_name: str = "robot",
    per_dof: bool = True,
) -> ParameterSet:
    """Build a per-joint Parameter per requested property with nominal = URDF value.

    For each property the parameter is constructed with the current entity
    value as ``nominal`` and absolute bounds from ``PROPERTIES`` — not
    scale-relative, since ``frictionloss`` often defaults to 0 which
    degenerates multiplicative bounds. Nominal is clipped into ``[lo, hi]``
    so tightening bounds is always effective; solver fields and modifier
    properties go through the same single loop.
    """
    params: list[Parameter] = []
    for prop in properties:
        tp = PROPERTIES[prop]
        lo, hi = tp.bounds
        nominal_full = tp.read_row(env, list(dof_names))
        if per_dof:
            nominal = nominal_full.astype(np.float64, copy=False)
        else:
            nominal = np.array([float(np.mean(nominal_full))], dtype=np.float64)
        min_v = np.full(nominal.shape, lo, dtype=np.float64)
        max_v = np.full(nominal.shape, hi, dtype=np.float64)
        nominal = np.clip(nominal, min_v, max_v)
        p = Parameter(
            name=prop,
            property=cast(Any, prop),
            dof_names=tuple(dof_names),
            entity_name=entity_name,
            per_dof=per_dof,
            nominal=nominal,
            min_value=min_v,
            max_value=max_v,
        )
        p.value = p.nominal.copy()
        params.append(p)
    return ParameterSet(params)


def _residual(
    env,
    params: ParameterSet,
    trajectories: Sequence[Trajectory],
    entity_name: str,
    signals: Sequence[str],
    signal_weights: dict[str, float],
    normalize: bool,
) -> np.ndarray:
    preds = [single_candidate_rollout(env, params, traj, entity_name, signals) for traj in trajectories]
    return multi_trajectory_residual(
        preds,
        trajectories,
        signals=signals,
        weights=signal_weights,
        normalize=normalize,
    )


def _fit_scipy(
    env,
    params: ParameterSet,
    trajectories: Sequence[Trajectory],
    *,
    entity_name: str,
    signals: Sequence[str],
    signal_weights: dict[str, float],
    normalize: bool,
    max_iters: int,
    diff_step: float,
    parallel_fd: bool,
    verbose: bool,
) -> tuple[ParameterSet, FitResult]:
    """Trust-Region Reflective least-squares in the unit hypercube."""
    from scipy.optimize import least_squares

    opt_params = params.copy()
    n = opt_params.size
    if n == 0:
        return opt_params, FitResult(cost=0.0, nfev=0, x=np.zeros(0), message="no free parameters")

    lo, hi = opt_params.get_bounds()
    span = np.where(hi > lo, hi - lo, 1.0)
    x0_unit = np.clip((opt_params.as_vector() - lo) / span, 0.0, 1.0)

    def _to_physical(x_unit: np.ndarray) -> np.ndarray:
        return lo + np.clip(x_unit, 0.0, 1.0) * span

    def residual_fn(x_unit: np.ndarray) -> np.ndarray:
        opt_params.update_from_vector(_to_physical(x_unit))
        return _residual(env, opt_params, trajectories, entity_name, signals, signal_weights, normalize)

    jac_arg: Any = "2-point"
    if parallel_fd:
        K = n + 1
        if env.num_envs < K:
            raise ValueError(
                f"--optimizer scipy_parallel_fd requires env.num_envs >= {K} "
                f"(got {env.num_envs}). Build the sysid env with num_envs = n_params + 1."
            )
        h = float(diff_step)

        def jac_fn(x_unit: np.ndarray) -> np.ndarray:
            unit_candidates = np.tile(x_unit, (K, 1))
            for i in range(n):
                xp = x_unit[i] + h
                if xp > 1.0:
                    unit_candidates[i + 1, i] = x_unit[i] - h
                else:
                    unit_candidates[i + 1, i] = xp
            candidates = lo + np.clip(unit_candidates, 0.0, 1.0) * span

            per_env_res: list[np.ndarray | None] = [None] * K
            for traj in trajectories:
                preds = batched_candidate_rollout(env, opt_params, candidates, traj, entity_name, signals)
                for k in range(K):
                    pred_k = {name: preds[name][k] for name in preds}
                    r = multi_trajectory_residual(
                        [pred_k],
                        [traj],
                        signals=signals,
                        weights=signal_weights,
                        normalize=normalize,
                    )
                    per_env_res[k] = r if per_env_res[k] is None else np.concatenate([per_env_res[k], r])

            r_center = per_env_res[0]
            assert r_center is not None
            J = np.empty((r_center.size, n), dtype=np.float64)
            for i in range(n):
                step_sign = 1.0 if (x_unit[i] + h) <= 1.0 else -1.0
                col = per_env_res[i + 1]
                assert col is not None
                J[:, i] = (col - r_center) / (step_sign * h)
            return J

        jac_arg = jac_fn

    result = least_squares(
        residual_fn,
        x0_unit,
        bounds=(np.zeros(n), np.ones(n)),
        max_nfev=max_iters,
        verbose=2 if verbose else 0,
        x_scale=1.0,
        diff_step=diff_step,
        jac=jac_arg,
    )

    x_star = _to_physical(result.x)
    opt_params.update_from_vector(x_star)
    return opt_params, FitResult(
        cost=float(result.cost),
        nfev=int(result.nfev),
        x=x_star,
        jacobian=getattr(result, "jac", None),
        message=str(result.message),
        extras={
            "optimality": float(result.optimality),
            "status": int(result.status),
            "x_unit": np.asarray(result.x),
        },
    )


def _fit_cmaes(
    env,
    params: ParameterSet,
    trajectories: Sequence[Trajectory],
    *,
    entity_name: str,
    signals: Sequence[str],
    signal_weights: dict[str, float],
    normalize: bool,
    max_iters: int,
    sigma0: float,
    population_size: int | None,
    batched: bool,
    verbose: bool,
) -> tuple[ParameterSet, FitResult]:
    """CMA-ES on the unit hypercube, serial or one-rollout-per-generation."""
    try:
        import cma
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CMA-ES backend requires the 'cma' package.") from exc

    opt_params = params.copy()
    n = opt_params.size
    if n == 0:
        return opt_params, FitResult(cost=0.0, nfev=0, x=np.zeros(0), message="no free parameters")

    lo, hi = opt_params.get_bounds()
    span = np.where(hi > lo, hi - lo, 1.0)

    def _to_unit(x: np.ndarray) -> np.ndarray:
        return np.clip((x - lo) / span, 0.0, 1.0)

    def _from_unit(x_unit: np.ndarray) -> np.ndarray:
        return lo + x_unit * span

    pop = population_size or (4 + int(3 * np.log(max(n, 1))))
    if batched and pop != env.num_envs:
        raise ValueError(
            f"--optimizer cmaes (batched) requires env.num_envs == population_size "
            f"(got num_envs={env.num_envs}, pop={pop})."
        )

    es = cma.CMAEvolutionStrategy(
        _to_unit(opt_params.as_vector()),
        sigma0,
        {
            "bounds": [[0.0] * n, [1.0] * n],
            "popsize": pop,
            "maxiter": max_iters,
            "verbose": -9,
        },
    )

    def _score_serial(candidates: np.ndarray) -> np.ndarray:
        costs = np.zeros(candidates.shape[0], dtype=np.float64)
        for k, candidate in enumerate(candidates):
            opt_params.update_from_vector(candidate)
            for traj in trajectories:
                pred = single_candidate_rollout(env, opt_params, traj, entity_name, signals)
                r = multi_trajectory_residual(
                    [pred],
                    [traj],
                    signals=signals,
                    weights=signal_weights,
                    normalize=normalize,
                )
                costs[k] += 0.5 * float(np.dot(r, r))
        return costs

    def _score_batched(candidates: np.ndarray) -> np.ndarray:
        K = candidates.shape[0]
        costs = np.zeros(K, dtype=np.float64)
        for traj in trajectories:
            preds = batched_candidate_rollout(env, opt_params, candidates, traj, entity_name, signals)
            for k in range(K):
                per_env_pred = {name: preds[name][k] for name in preds}
                r = multi_trajectory_residual(
                    [per_env_pred],
                    [traj],
                    signals=signals,
                    weights=signal_weights,
                    normalize=normalize,
                )
                costs[k] += 0.5 * float(np.dot(r, r))
        return costs

    score_fn = _score_batched if batched else _score_serial
    history: list[float] = []
    best_ever = float("inf")
    t_start = time.monotonic()
    if verbose:
        mode = "batched" if batched else "serial"
        en.logger.info(
            f"[CMAES] n_params={n}, popsize={pop}, mode={mode}, num_envs={env.num_envs}, max_iters={max_iters}"
        )
        en.logger.info(f"{'gen':>4}  {'nfev':>7}  {'best':>12}  {'mean':>12}  {'sigma':>10}  {'elapsed':>8}")
    gen = 0
    while not es.stop():
        t_gen = time.monotonic()
        unit_candidates = np.asarray(es.ask())
        candidates = np.stack([_from_unit(c) for c in unit_candidates])
        if verbose:
            en.logger.info(
                f"[gen {gen + 1:>3d}] scoring {candidates.shape[0]} candidates over {len(trajectories)} trajectories …"
            )
        costs = score_fn(candidates)
        es.tell(list(unit_candidates), list(costs.tolist()))
        gen_best = float(np.min(costs))
        history.append(gen_best)
        best_ever = min(best_ever, gen_best)
        gen += 1
        if verbose:
            gen_dt = time.monotonic() - t_gen
            elapsed = time.monotonic() - t_start
            en.logger.info(
                f"{gen:>4d}  {int(es.result.evaluations):>7d}  "
                f"{best_ever:>12.4g}  {float(np.mean(costs)):>12.4g}  "
                f"{float(es.sigma):>10.4g}  {elapsed:>6.1f}s  (+{gen_dt:.1f}s)"
            )

    best = _from_unit(np.asarray(es.result.xbest))
    opt_params.update_from_vector(best)
    return opt_params, FitResult(
        cost=float(es.result.fbest),
        nfev=int(es.result.evaluations),
        x=best,
        message="cma completed",
        history=np.asarray(history) if history else None,
    )


def _warm_start_from_yaml(params: ParameterSet, yaml_path: pathlib.Path) -> None:
    """Copy ``value`` from a saved ParameterSet YAML into an existing set.

    Bounds and nominal in ``params`` are preserved — only values are overwritten.
    Missing params, shape mismatches, and out-of-bounds values are each logged
    once and fall back to the current nominal (or a clipped value).
    """
    if not yaml_path.exists():
        raise SystemExit(f"--resume path {yaml_path} does not exist.")
    prev = ParameterSet.load_yaml(yaml_path)
    en.logger.info(f"Resuming from {yaml_path} ({len(prev)} params in file)")

    copied, skipped = 0, 0
    for p in params:
        if p.name not in prev:
            en.logger.warning(f"--resume: '{p.name}' not in {yaml_path.name}; using nominal.")
            skipped += 1
            continue
        loaded = prev[p.name]
        if loaded.value.shape != p.value.shape:
            en.logger.warning(
                f"--resume: '{p.name}' shape {loaded.value.shape} in file "
                f"doesn't match current {p.value.shape}; using nominal."
            )
            skipped += 1
            continue
        if tuple(loaded.dof_names) != tuple(p.dof_names):
            # Shape can match while joint names diverge (rename, reorder, or
            # left/right swap). Silently slotting per-DOF values into the wrong
            # joints corrupts the warm start; fail loud instead.
            en.logger.warning(
                f"--resume: '{p.name}' dof_names mismatch — file has "
                f"{list(loaded.dof_names)} but current is {list(p.dof_names)}; using nominal."
            )
            skipped += 1
            continue
        clipped = np.clip(loaded.value, p.min_value, p.max_value)
        if not np.allclose(clipped, loaded.value):
            en.logger.warning(
                f"--resume: '{p.name}' value clipped to current bounds [{p.min_value.min():g}, {p.max_value.max():g}]."
            )
        p.value = clipped.copy()
        copied += 1
    en.logger.info(f"--resume: warm-started {copied} params, skipped {skipped}.")


def main() -> int:
    parser = make_argparser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="One or more .npz files written by collect_data.py (or SysIDRecorder).",
    )
    parser.add_argument(
        "--include",
        choices=list(PROPERTIES.keys()),
        nargs="+",
        default=list(PROPERTIES.keys()),
        help="Which parameters to identify (entity fields + HAND_CONTROLLER modifiers by default).",
    )
    parser.add_argument(
        "--per-joint",
        action=BooleanOptionalAction,
        default=True,
        help="If set, identify a separate scalar per joint (recommended).",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["dofs_pos"],
        choices=["dofs_pos", "dofs_vel", "dofs_torque"],
        help="Signals used in the residual. XHand1 SDK reports only dofs_pos.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["scipy", "scipy_parallel_fd", "cmaes"],
        default="scipy",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Sim-twin env count. For scipy_parallel_fd use n_params + 1; for cmaes use population size.",
    )
    parser.add_argument(
        "--diff-step",
        type=float,
        default=1e-2,
        help="Relative FD step in unit-hypercube parameter space (scipy backends). "
        "Default 1e-2 = 1%% of each bound range.",
    )
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument(
        "--sigma0",
        type=float,
        default=None,
        help="CMA-ES initial step size in unit-hypercube parameter space. "
        "Defaults: 0.25 from scratch, 0.05 when --resume is set (exploit a warm-start). "
        "Higher = more exploration, lower = tighter local search.",
    )
    parser.add_argument(
        "--resume",
        type=pathlib.Path,
        default=None,
        help="Path to a previously saved params_identified.yaml to warm-start from. "
        "Bounds are taken from the current PROPERTIES table; only fitted values are "
        "copied. Mismatched shapes, missing params, or dof_name mismatches fall back "
        "to the fresh nominal.",
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    en.init(backend=gs.gpu, performance_mode=False)
    config = make_sim_twin_config(
        args.robot, num_envs=args.num_envs, sim_dt=args.sim_dt, decimation=args.decimation
    )
    from eden.envs.base import RLEnvBase

    env = RLEnvBase.from_config(config)
    env.build()
    install_action_mod_sysid_patch()

    entity = env.entities["robot"]
    dof_names = list(entity.dofs_name)

    file_paths = []
    for path in args.trajectory:
        file_paths.extend(glob.glob(str(path)))
    trajectories = [Trajectory.load(p) for p in file_paths]
    en.logger.info(
        f"Loaded {len(trajectories)} trajectories "
        f"(total {sum(len(t) for t in trajectories)} samples across "
        f"{sum(len(t) * t.dt for t in trajectories):.1f} s)."
    )

    # The rollout steps at env.dt per trajectory sample. If the real loop ran
    # at a different cadence than the sim twin, the replayed chirp will be
    # frequency-shifted and the residual will carry a systematic bias no
    # parameter fit can remove.
    sim_step_dt = config.env_options.sim_dt * config.env_options.decimation
    for i, t in enumerate(trajectories):
        rel_err = abs(t.dt - sim_step_dt) / sim_step_dt if sim_step_dt > 0 else 0.0
        if rel_err > 0.02:
            en.logger.warning(
                f"trajectory {i} dt={t.dt * 1000:.2f} ms but sim step "
                f"dt={sim_step_dt * 1000:.2f} ms ({rel_err * 100:.1f} % mismatch). "
                f"Either re-record at --control-freq {1 / sim_step_dt:.0f} or "
                f"retune the sim's sim_dt * decimation to match the recorded dt."
            )

    params = _build_parameters(env, dof_names, args.include, per_dof=args.per_joint)
    if args.resume is not None:
        _warm_start_from_yaml(params, args.resume)
    en.logger.info(
        f"Optimising {params.size} free scalars across {len(args.include)} parameter group(s) "
        f"({len(dof_names)} DOFs, per_joint={args.per_joint})."
    )

    initial = params.copy()
    signals = tuple(args.signals)
    if args.optimizer == "cmaes":
        sigma0 = args.sigma0
        if sigma0 is None:
            sigma0 = 0.05 if args.resume is not None else 0.25
        en.logger.info(f"CMA-ES sigma0={sigma0:g} ({'warm-start' if args.resume is not None else 'cold-start'}).")
        identified, result = _fit_cmaes(
            env,
            params,
            trajectories,
            entity_name="robot",
            signals=signals,
            signal_weights={},
            normalize=True,
            max_iters=args.max_iters,
            sigma0=sigma0,
            population_size=args.num_envs if args.num_envs > 1 else None,
            batched=args.num_envs > 1,
            verbose=True,
        )
    else:
        parallel_fd = args.optimizer == "scipy_parallel_fd"
        if parallel_fd and args.num_envs < params.size + 1:
            raise SystemExit(
                f"scipy_parallel_fd needs --num-envs >= n_params + 1 (got {args.num_envs}, need {params.size + 1})."
            )
        identified, result = _fit_scipy(
            env,
            params,
            trajectories,
            entity_name="robot",
            signals=signals,
            signal_weights={},
            normalize=True,
            max_iters=args.max_iters,
            diff_step=args.diff_step,
            parallel_fd=parallel_fd,
            verbose=True,
        )

    en.logger.info(f"Optimisation done. cost={result.cost:.6g} nfev={result.nfev} msg={result.message!r}")

    args.output.mkdir(parents=True, exist_ok=True)
    write_summary(args.output, initial, identified, result)

    # Per-signal plots on the first trajectory with both the pre-fit (URDF
    # defaults) and post-fit predictions overlaid on the measurement so the
    # improvement from identification is visible at a glance.
    traj0 = trajectories[0]
    initial_pred = single_candidate_rollout(env, initial, traj0, "robot", signals)
    identified_pred = single_candidate_rollout(env, identified, traj0, "robot", signals)
    signal_plots(
        predicted={"initial": initial_pred, "identified": identified_pred},
        measured=traj0,
        signals=signals,
        save_dir=args.output / "plots",
        dof_names=dof_names,
    )
    en.logger.info(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
