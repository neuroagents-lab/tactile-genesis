"""CMA-ES optimizer for system identification."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

import eden as en
from eden.extensions.sysid.base import SYSID_REGISTRY, FitResult, SystemIdentificationBase
from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.residual import multi_trajectory_residual
from eden.extensions.sysid.rollout import (
    batched_candidate_rollout,
    single_candidate_rollout,
)
from eden.extensions.sysid.trajectory import Trajectory


@SYSID_REGISTRY.register()
class CMAES(SystemIdentificationBase):
    """CMA-ES identifier for noisy or contact-heavy residuals (e.g. dex hand).

    By default evaluates candidates **sequentially** on the current env —
    each candidate ``ask()`` writes its own parameter vector across all
    envs and runs one replay. This works with any ``env.num_envs`` and
    matches the serial assumption used by ``SciPyLeastSquares``.

    To enable **batched** candidate evaluation set ``batched=True`` and
    build the sysid env with ``env_options.num_envs == population_size``;
    a single replay then scores the whole population, reducing the
    per-generation wall-clock by a factor close to the population size.

    Extra options
    -------------
    sigma0: float
        Initial step size in the unit-hypercube reparameterisation. 0.25
        explores ~50% of each bound interval on the first generation.
    population_size: int | None
        CMA population. Defaults to the library's recommendation
        ``4 + floor(3*log(n))``.
    batched: bool
        If True, score the whole population in one replay per trajectory;
        requires ``env.num_envs == population_size``.
    """

    sigma0: float = 0.25
    population_size: int | None = None
    batched: bool = False

    def fit(
        self,
        params: ParameterSet,
        trajectories: Sequence[Trajectory],
    ) -> tuple[ParameterSet, FitResult]:
        try:
            import cma
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("CMAES backend requires the 'cma' package.") from exc

        opt_params = params.copy()
        n = opt_params.size
        if n == 0:
            return opt_params, FitResult(cost=0.0, nfev=0, x=np.zeros(0), message="no free parameters")

        lo, hi = opt_params.get_bounds()
        x0 = opt_params.as_vector()
        x0_scaled = _to_unit(x0, lo, hi)

        pop = self.population_size or (4 + int(3 * np.log(max(n, 1))))
        if self.batched and pop != self._env.num_envs:
            raise ValueError(
                f"CMAES(batched=True) requires env.num_envs == population_size "
                f"(got num_envs={self._env.num_envs}, pop={pop})."
            )

        es = cma.CMAEvolutionStrategy(
            x0_scaled,
            self.sigma0,
            {
                "bounds": [[0.0] * n, [1.0] * n],
                "popsize": pop,
                "maxiter": self.max_iters,
                # Quiet cma's internal chatter — we print our own per-generation line below.
                "verbose": -9,
            },
        )

        history: list[float] = []
        score_fn = self._score_batched if self.batched else self._score_serial
        best_ever = float("inf")
        t_start = time.monotonic()
        if self.verbose:
            mode = "batched" if self.batched else "serial"
            en.logger.info(
                f"[CMAES] n_params={n}, popsize={pop}, mode={mode}, "
                f"num_envs={self._env.num_envs}, max_iters={self.max_iters}"
            )
            en.logger.info(f"{'gen':>4}  {'nfev':>7}  {'best':>12}  {'mean':>12}  {'sigma':>10}  {'elapsed':>8}")
        gen = 0
        while not es.stop():
            t_gen = time.monotonic()
            unit_candidates = np.asarray(es.ask())
            candidates = np.stack([_from_unit(c, lo, hi) for c in unit_candidates])
            if self.verbose:
                en.logger.info(
                    f"[gen {gen + 1:>3d}] scoring {candidates.shape[0]} candidates "
                    f"over {len(trajectories)} trajectories …"
                )
            costs = score_fn(opt_params, candidates, trajectories)
            es.tell(list(unit_candidates), list(costs.tolist()))
            gen_best = float(np.min(costs))
            history.append(gen_best)
            best_ever = min(best_ever, gen_best)
            gen += 1
            if self.verbose:
                gen_dt = time.monotonic() - t_gen
                elapsed = time.monotonic() - t_start
                en.logger.info(
                    f"{gen:>4d}  {int(es.result.evaluations):>7d}  "
                    f"{best_ever:>12.4g}  {float(np.mean(costs)):>12.4g}  "
                    f"{float(es.sigma):>10.4g}  {elapsed:>6.1f}s  (+{gen_dt:.1f}s)"
                )

        best_unit = np.asarray(es.result.xbest)
        best = _from_unit(best_unit, lo, hi)
        opt_params.update_from_vector(best)

        return opt_params, FitResult(
            cost=float(es.result.fbest),
            nfev=int(es.result.evaluations),
            x=best,
            message="cma completed",
            history=np.asarray(history) if history else None,
        )

    def _score_serial(
        self,
        params: ParameterSet,
        candidates: np.ndarray,
        trajectories: Sequence[Trajectory],
    ) -> np.ndarray:
        costs = np.zeros(candidates.shape[0], dtype=np.float64)
        for k, candidate in enumerate(candidates):
            params.update_from_vector(candidate)
            for traj in trajectories:
                pred = single_candidate_rollout(
                    self._env,
                    params,
                    traj,
                    entity_name=self.entity_name,
                    signals=self.signals,
                )
                res = multi_trajectory_residual(
                    [pred],
                    [traj],
                    signals=self.signals,
                    weights=self.signal_weights,
                    normalize=self.normalize,
                )
                costs[k] += 0.5 * float(np.dot(res, res))
        return costs

    def _score_batched(
        self,
        params: ParameterSet,
        candidates: np.ndarray,
        trajectories: Sequence[Trajectory],
    ) -> np.ndarray:
        K = candidates.shape[0]
        costs = np.zeros(K, dtype=np.float64)
        for traj in trajectories:
            preds = batched_candidate_rollout(
                self._env,
                params,
                candidates,
                traj,
                entity_name=self.entity_name,
                signals=self.signals,
            )
            for k in range(K):
                per_env_pred = {name: preds[name][k] for name in preds}
                res = multi_trajectory_residual(
                    [per_env_pred],
                    [traj],
                    signals=self.signals,
                    weights=self.signal_weights,
                    normalize=self.normalize,
                )
                costs[k] += 0.5 * float(np.dot(res, res))
        return costs


def _to_unit(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.clip((x - lo) / span, 0.0, 1.0)


def _from_unit(x_unit: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo + x_unit * (hi - lo)
