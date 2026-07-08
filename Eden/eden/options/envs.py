"""Top-level environment configuration options (sim, solvers, batching)."""

from typing import Literal

import genesis as gs
from genesis.typing import Vec2FType, Vec3FType

from eden.options.options import ConfigurableOptions


class EnvOptions(ConfigurableOptions):
    """Top-level environment configuration (sim, solvers, batching).

    Parameters
    ----------
    num_envs: int
        Number of environments to run in parallel.
    num_eval_envs: int
        Number of environments to run in evaluation mode.
    sim_dt: float
        Simulation time step.
    sim_substeps: int
        Number of simulation substeps.
    decimation: int
        Decimation factor for the simulation. The actual time step is `sim_dt * decimation`.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    episode_length_s: float
        Length of the episode in seconds.
    use_gjk_collision: bool
        Whether to use GJK collision detection. Defaults to True. If False, MPR+GJK will be used.
    solver: Literal["newton", "cg"]
        Solver to use for collision detection.
    constraint_timeconst: float | None
        Time constant for the constraint solver. If None, it will be computed as max(0.01, 2 * sim_dt / sim_substeps).
    max_collision_pairs: int
        Maximum number of collision pairs.
    iterations: int
    tolerance: float
    ls_iterations: int
    ls_tolerance: float
    enable_self_collision: bool
        Whether to enable self collision detection. Defaults to True.
    batch_links_info: bool
        Whether to batch links information. Defaults to False.
    batch_joints_info: bool
        Whether to batch joints information. Defaults to False.
    batch_dofs_info: bool
        Whether to batch DOFs information. Defaults to True.
    enable_multi_contact: bool
        Whether to enable multi-contact detection. Defaults to False.
    noslip_iterations: int
        Number of iterations for the noslip solver. Defaults to 0 (disabled).
        noslip is a post-processing step after the main solver to suppress slip/drift.
        Recommended to set this value to 5 for manipulation tasks or when slip/drift is a big problem.

    env_spacing: array-like[float, float]
        Spacing between environments. Defaults to (0.0, 0.0).
    center_envs_at_origin: bool
        Whether to center environments at the origin. Defaults to False.
    show_link_frame: bool
        Whether to show link frames. Defaults to False.
    show_world_frame: bool
        Whether to show world frames. Defaults to False.
    show_camera_frustums: bool
        Whether to show camera frustums. Defaults to False.
    segmentation_level: Literal["entity", "link", "geom"]
        Level of segmentation. Defaults to "entity".
    show_FPS: bool
        Whether to show FPS for profiling. Defaults to False.
    enable_default_keybinds: bool
        Whether to disable keyboard shortcuts in the default viewer. Defaults to False.
    background_color: array-like[float, float, float]
        RGB background color (each component in [0, 1]) for the viewer/renderer. Defaults to (0.04, 0.08, 0.12).
    update_sensors_every_substep: bool
        If False (default), Genesis's ``SensorManager.step()`` runs only on the
        last sub-step of the env decimation loop. Intermediate sub-step sensor
        readings would be discarded by the next call before anything reads
        them (``observation_manager.compute()`` reads from the sensor cache
        once at the end of ``step()``), so updating them is pure waste — for
        ray-casting depth cameras at perceptive_mimic scale this is a ~2.3×
        end-to-end ``env.step()`` speedup at decimation=4.

        The skip auto-disables on a per-step basis when any sensor declares
        Genesis-level ``SensorOptions.history_length`` or ``delay`` (i.e. the
        SensorManager allocates timeline / return rings) — those rings need
        sim-rate fresh writes to stay correct, so leaving the optimisation on
        by default is safe even for sensor-history users.

        Set True to force per-sub-step updates unconditionally (mostly useful
        for direct A/B comparison against the legacy per-sub-step behavior).

    record_final_observations: bool
        Snapshot pre-reset observations for done envs into ``extras['final_observations']``.

        Off by default. When on, ``RLEnvBase.step()`` runs one extra
        ``observation_manager.compute(update_history=False)`` every step. The
        returned ``extras`` dict gains a ``'final_observations'`` entry shaped
        ``(num_envs, *obs_shape)`` per group; rows for non-done envs are zero,
        and on a no-termination step every row is zero. For groups with
        ``history_length > 0`` the row is history-shaped with the freshly-
        computed post-physics frame in the most-recent slot (composed via
        ``CircularBuffer.peek_buffer``; the canonical once-per-step advance
        still happens in the post-reset compute).

        **No per-step GPU→CPU sync.** Both the gating logic and the snapshot
        copy stay on-device. Cost: one extra
        ``observation_manager.compute(update_history=False)`` per step plus
        one ``torch.where`` per observation group.

        **rsl_rl PPO does not consume this**, by design — across rsl_rl 5.0.1
        (installed) → 5.2.0 (latest reference) the only ``extras`` keys read
        are ``"time_outs"``, ``"log"``, and ``"episode"``. rsl_rl bootstraps
        time-out V-targets from ``V(s_t)`` via ``extras['time_outs']`` (a
        deliberate approximation; not something this flag changes). Leaving
        this flag off in pure rsl_rl runs costs nothing.

        Turn it on for off-policy algorithms whose replay buffers need the
        actual ``s'`` for time-out transitions (FastSAC), eval-recording
        callbacks that segment trajectories on done and need the real terminal
        frame for visualization / per-trajectory metrics, or a future
        Holosoma-style PPO runner that bootstraps from ``V(s_{t+1}^{phys})``
        instead of rsl_rl's ``V(s_t)`` proxy.

    coupler_options : gs.options.BaseCouplerOptions | None = None
        The options configuring the coupler. Defaults to None (LegacyCouplerOptions).

    mpm_options: gs.options.MPMOptions | None = None
        The options configuring the mpm_solver. Defaults to None.
    sph_options: gs.options.SPHOptions | None = None
        The options configuring the sph_solver. Defaults to None.
    pbd_options: gs.options.PBDOptions | None = None
        The options configuring the pbd_solver (Position-Based Dynamics: liquids,
        cloth, soft bodies, free particles). Defaults to None.
    """

    num_envs: int = 4096
    num_eval_envs: int = 16
    sim_dt: float = 0.005
    sim_substeps: int = 1
    decimation: int = 4
    requires_grad: bool = False
    episode_length_s: float = 20.0
    use_gjk_collision: bool | None = True
    solver: Literal["newton", "cg"] = "newton"
    constraint_timeconst: float | None = None

    max_collision_pairs: int = 50
    multiplier_collision_broad_phase: int = 8
    iterations: int = 50
    tolerance: float = 1e-5
    ls_iterations: int = 50
    ls_tolerance: float = 1e-2
    enable_self_collision: bool = True
    batch_links_info: bool = False
    batch_joints_info: bool = False
    batch_dofs_info: bool = True
    enable_multi_contact: bool = True
    noslip_iterations: int = 0
    use_hibernation: bool = False

    teleport_robots: bool = False  # TODO
    env_separate_rigid: bool = False
    env_spacing: Vec2FType = (0.0, 0.0)
    center_envs_at_origin: bool = False
    show_link_frame: bool = False
    show_world_frame: bool = False
    show_camera_frustums: bool = False
    segmentation_level: Literal["entity", "link", "geom"] = "entity"

    show_FPS: bool = False
    enable_default_keybinds: bool = True
    background_color: Vec3FType = (0.04, 0.08, 0.12)

    record_final_observations: bool = False
    update_sensors_every_substep: bool = False

    coupler_options: gs.options.BaseCouplerOptions | None = None
    fem_options: gs.options.FEMOptions | None = None
    mpm_options: gs.options.MPMOptions | None = None
    sph_options: gs.options.SPHOptions | None = None
    pbd_options: gs.options.PBDOptions | None = None

    def model_post_init(self, context):
        super().model_post_init(context)

        assert self.num_envs > 0, "`num_envs` should be greater than 0"
        assert self.num_eval_envs > 0, "`num_eval_envs` should be greater than 0"

        if isinstance(self.coupler_options, gs.options.IPCCouplerOptions):
            if self.batch_dofs_info:
                raise ValueError("`batch_dofs_info` must be False when using IPCCouplerOptions.")
            if self.num_envs > 1:
                raise ValueError(
                    "IPC coupler does not support per-env reset (num_envs > 1). "
                    "Use num_envs=1 until per-env reset via libuipc StateAccessor is implemented."
                )
