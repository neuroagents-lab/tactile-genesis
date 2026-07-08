"""SciPy least-squares optimizer for system identification."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from eden.extensions.sysid.base import SYSID_REGISTRY, FitResult, SystemIdentificationBase
from eden.extensions.sysid.parameter import ParameterSet
from eden.extensions.sysid.residual import multi_trajectory_residual
from eden.extensions.sysid.rollout import batched_candidate_rollout
from eden.extensions.sysid.trajectory import Trajectory


@SYSID_REGISTRY.register()
class SciPyLeastSquares(SystemIdentificationBase):
    """Trust-Region Reflective least-squares with box bounds.

    Operates on a unit-hypercube reparameterisation of the free parameters
    — decision variables live in ``[0, 1]`` regardless of their physical
    scale — so the finite-difference Jacobian uses a step of ``diff_step``
    fraction of each parameter's bound range (not ``sqrt(eps)`` of its
    absolute value, which is below the sim's numerical floor for most
    solver fields). This is the same reparameterisation the CMA-ES backend
    uses.

    Extra options
    -------------
    diff_step: float
        Relative FD step in unit space. ``1e-2`` = 1 % of each bound
        range. Too small → FD underflows to numerical noise and the
        optimiser terminates at ``xtol`` with no progress; too large →
        the Jacobian becomes a secant rather than a derivative.
    parallel_fd: bool
        If True, write ``n_params + 1`` FD columns into as many envs and
        score them all in one replay per trajectory via
        :func:`batched_candidate_rollout`. Requires
        ``env.num_envs >= n_params + 1``. Falls back to scipy's default
        serial 2-point FD when False.
    """

    diff_step: float = 1e-2
    parallel_fd: bool = False

    def fit(
        self,
        params: ParameterSet,
        trajectories: Sequence[Trajectory],
    ) -> tuple[ParameterSet, FitResult]:
        opt_params = params.copy()
        n = opt_params.size
        if n == 0:
            return opt_params, FitResult(cost=0.0, nfev=0, x=np.zeros(0), message="no free parameters")

        lo, hi = opt_params.get_bounds()
        span = np.where(hi > lo, hi - lo, 1.0)
        x0 = opt_params.as_vector()
        x0_unit = np.clip((x0 - lo) / span, 0.0, 1.0)

        def _to_physical(x_unit: np.ndarray) -> np.ndarray:
            return lo + np.clip(x_unit, 0.0, 1.0) * span

        def residual_fn(x_unit: np.ndarray) -> np.ndarray:
            opt_params.update_from_vector(_to_physical(x_unit))
            return self.evaluate(opt_params, trajectories)

        jac_arg = "2-point"
        K = n + 1  # center + one perturbation per parameter
        if self.parallel_fd:
            if self._env.num_envs < K:
                raise ValueError(
                    f"SciPyLeastSquares(parallel_fd=True) requires env.num_envs >= {K} "
                    f"(got {self._env.num_envs}). Build the sysid env with num_envs = n_params + 1."
                )
            jac_arg = self._build_parallel_jacobian(opt_params, trajectories, lo, span, K)

        result = least_squares(
            residual_fn,
            x0_unit,
            bounds=(np.zeros(n), np.ones(n)),
            max_nfev=self.max_iters,
            verbose=2 if self.verbose else 0,
            x_scale=1.0,
            diff_step=self.diff_step,
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

    def _build_parallel_jacobian(
        self,
        opt_params: ParameterSet,
        trajectories: Sequence[Trajectory],
        lo: np.ndarray,
        span: np.ndarray,
        K: int,
    ):
        """Forward-difference Jacobian computed with one batched rollout per trajectory.

        Writes K = n_params + 1 candidate vectors into K envs (row 0 is the
        centre, rows 1..n are ``x + diff_step * e_i`` in unit space), then
        a single replay per trajectory produces K parallel predictions.
        """
        n = opt_params.size
        h = float(self.diff_step)

        def jac_fn(x_unit: np.ndarray) -> np.ndarray:
            # Build K candidate vectors in physical space. Row 0 = centre;
            # rows 1..n = x + h*e_i (clipped to the unit box).
            unit_candidates = np.tile(x_unit, (K, 1))
            for i in range(n):
                xp = x_unit[i] + h
                if xp > 1.0:
                    # Near the upper bound: backward step instead.
                    unit_candidates[i + 1, i] = x_unit[i] - h
                else:
                    unit_candidates[i + 1, i] = xp
            candidates = lo + np.clip(unit_candidates, 0.0, 1.0) * span

            # One batched rollout per trajectory; aggregate residuals per env.
            per_env_res: list[np.ndarray] = [None] * K  # type: ignore[list-item]
            for traj in trajectories:
                preds = batched_candidate_rollout(
                    self._env,
                    opt_params,
                    candidates,
                    traj,
                    entity_name=self.entity_name,
                    signals=self.signals,
                )
                for k in range(K):
                    pred_k = {name: preds[name][k] for name in preds}
                    r = multi_trajectory_residual(
                        [pred_k],
                        [traj],
                        signals=self.signals,
                        weights=self.signal_weights,
                        normalize=self.normalize,
                    )
                    per_env_res[k] = r if per_env_res[k] is None else np.concatenate([per_env_res[k], r])

            r_center = per_env_res[0]
            J = np.empty((r_center.size, n), dtype=np.float64)
            for i in range(n):
                step_sign = 1.0 if (x_unit[i] + h) <= 1.0 else -1.0
                # Derivative in unit space: dr / d(x_unit_i); least_squares is
                # optimising x_unit so no further chain rule is needed.
                J[:, i] = (per_env_res[i + 1] - r_center) / (step_sign * h)
            # scipy caches the "current" residual separately; return J only.
            # We rely on the serial residual_fn being called once per iteration
            # for the centre, which is cheap (one rollout per trajectory).
            return J

        return jac_fn
