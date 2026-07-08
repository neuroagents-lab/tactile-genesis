"""Rotate screwdriver task."""

from __future__ import annotations

from typing import Annotated, Any

import eden as en
import genesis as gs
from eden import terminations
from eden.constants import EventMode
from eden.managers.modifiers.observations import GaussianNoise
from eden.options import ObservationGroupOptions, PlaneOptions, SensorsOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.learning.rsl_rl import (
    RslRlBaseRunnerOptions,
    RslRlDistillationAlgorithmOptions,
    RslRlGaussianDistributionOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import (
    ActionManagerOptions,
    CommandManagerOptions,
    EdenRLConfig,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    TerminationManagerOptions,
)

from entities.screwdrivers import FatScrewdrivers
from registry import HYPERPARAMS_REGISTRY, TASK_REGISTRY
from shared_terms import (
    CONTROLLER_RANDOMIZATIONS,
    DECIMATION,
    PARTIAL_HAND_CONTROLLER,
    PI,
    SIM_DT,
    LoadGraspPose,
    RotationAxisCommand,
    TactileSensorRead,
    base_rot6d,
    filter_hand_dof_names,
    obj_pos_drift_from_grasp,
    obj_tilted_past_threshold,
    rotation_reward,
    surface_distance_reward,
    termination_penalty,
    work_penalty,
)
from task_mods import DexHandRslRlRunnerMod, RobotHandWithPrivSensorsMod, RobotLiteral, TactileSensorsMod

from .custom_terms import (
    ScrewAxisRotationProgressObs,
    ScrewNutStagnation,
    ScrewObjTiltObs,
    ScrewPoseDiffPenalty,
    ScrewVerticalAlignmentPenalty,
    screw_object_dist_penalty,
    screw_total_rotation_metric,
)

RUNNER_CFG: dict[str, Any] = {
    "num_steps_per_env": 12,
    "max_iterations": 5000,
    "save_interval": 500,
    "empirical_normalization": True,
    "obs_groups": {
        # actor will be set to student/teacher based on stage
        "critic": ["priv_proprio", "priv_obj_state"],
        "student": ["proprio", "tactile_sensors"],
        "teacher": ["priv_proprio", "priv_obj_state"],
        "rnd_state": ["priv_proprio", "priv_obj_state"],
    },
}
PPO_OPTIONS = RslRlPpoAlgorithmOptions(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.0,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
DISTILL_OPTIONS = RslRlDistillationAlgorithmOptions(
    num_learning_epochs=5,
    learning_rate=1.0e-5,
    clip_param=0.1,
    gradient_length=4,
    max_grad_norm=0.5,
    optimizer="adam",
    # Huber is robust to the rare large normalized errors produced by
    # normalize_action_targets on near-constant action dims (idle/frozen DOFs),
    # which otherwise spike the behavior loss at synchronized reset waves.
    loss_type="huber",
    # Standardize teacher action targets per-dim so the behavior loss becomes an
    # inverse-variance-weighted MSE -- keeps the small, task-critical screwdriver
    # corrections from being drowned out by larger idle-pose actions.
    normalize_action_targets=True,
    auxiliary_losses=[
        # Predict the screwdriver's off-axis tilt (how much it is falling over)
        # from the student latent. Tilt is ~0..pi/2 rad; scale up toward unit range.
        {
            "class_name": "rsl_rl.extensions.auxiliary:StudentLatentPredictionLoss",
            "target_obs": "aux_obj_tilt",
            "name": "obj_tilt",
            "weight": 0.5,
            "target_scale": 1.0,
            "hidden_dims": [512, 128],
            "loss_type": "mse",
        },
        # Predict the per-step target-axis rotation progress from the student
        # latent. Per-step deltas are small (~0.01..0.1 rad); scale up to unit range.
        {
            "class_name": "rsl_rl.extensions.auxiliary:StudentLatentPredictionLoss",
            "target_obs": "aux_rotation_progress",
            "name": "rotation_progress",
            "weight": 0.5,
            "target_scale": 1.0,
            "hidden_dims": [512, 128],
            "loss_type": "mse",
        },
    ],
)

MLP_CFG: dict[str, Any] = {
    "hidden_dims": [512, 256, 128],
    "activation": "elu",
}


class ScrewdriverRobotMod(RobotHandWithPrivSensorsMod):
    """Load the hand and apply screwdriver-specific robot/action settings."""

    def __init__(
        self,
        robot: Annotated[RobotLiteral, "The robot to use for the task."] = "xhand1",
        include_ext_force: Annotated[
            bool,
            "Enable the random ApplyExternalForce event on the screwdriver. Defaults on for "
            "teacher RL robustness; turn off for distillation since the student lacks the "
            "obj-velocity privileged signal needed to react to kicks.",
        ] = True,
        *,
        entity_name: str = "robot",
        action_term_name: str = "dofs_pos_controller",
        priv_sensor_cfg_name: str = "fingertips",
        track_link_idx: str | tuple[int, ...] | tuple[str, ...] = "obj",
        name: str = "",
    ) -> None:
        super().__init__(
            robot=robot,
            entity_name=entity_name,
            action_term_name=action_term_name,
            priv_sensor_cfg_name=priv_sensor_cfg_name,
            track_link_idx=track_link_idx,
            name=name,
        )
        self.include_ext_force = include_ext_force

    def _configure_robot_and_actions(self, config: EdenRLConfig, robot: Any) -> Any:
        robot.is_fixed_base = True
        robot.batch_fixed_verts = True
        active_dofs = filter_hand_dof_names(robot.dofs_name, excluded_fingers=("ring", "pinky"))
        frozen_dofs = tuple(name for name in robot.dofs_name if name not in active_dofs)
        config.action_options.dofs_pos_controller = config.action_options.dofs_pos_controller.model_copy(
            update={
                "frozen_dofs": frozen_dofs,
                "dofs_name": list(active_dofs),
            }
        )
        config.scene_options.obj.surface = gs.surfaces.Default(
            diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.1, 0.0, 1.0))
        )
        return robot

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        if not self.include_ext_force and hasattr(config.event_options, "ext_force"):
            del config.event_options.ext_force
        return config


HYPERPARAMS_REGISTRY.register(
    # Stage-2 (distillation) sweep ranges. Centered on the screwdriver defaults
    # in DISTILL_OPTIONS above. Ranges with max/min >= 100 are sampled
    # log-uniform automatically (see optimization.sample_hyperparams); tuples
    # of ints become integer ranges; lists become categorical.
    {
        # Algorithm params -- routed to runner.algorithm by apply_params_to_config.
        "learning_rate": (5e-7, 5e-4),  # default 1e-5  (log)
        "num_learning_epochs": [2, 4, 8, 12],  # default 5
        "clip_param": (0.05, 0.3),  # default 0.1
        "max_grad_norm": (0.25, 2.0),  # default 0.5
        "gradient_length": [2, 4, 6, 8],  # default 4   (categorical)
        # Auxiliary-loss weights. Suffix matches `name` in
        # DISTILL_OPTIONS.auxiliary_losses; routed by apply_params_to_config.
        "aux_obj_tilt": (0.0, 2.0),  # default 0.5
        "aux_rotation_progress": (0.0, 2.0),  # default 0.5
    },
    # Stage-1 (PPO) reward weights -- left commented for reference. Uncomment
    # if reusing this registry to sweep teacher rewards.
    # {
    #     "rew_rotation_reward": (1.0, 20.0),
    #     "rew_surface_distance_reward": (0.1, 5.0),
    #     "rew_obj_dist_penalty": (-1000.0, -10.0),
    #     "rew_vertical_alignment_penalty": (-1000.0, -10.0),
    #     "rew_pose_diff_penalty": (-1.0, -0.01),
    #     "rew_termination_penalty": (-100.0, -1.0),
    #     "rew_action_rate_penalty": (-1e-2, -1e-4),
    #     "rew_joint_limits_penalty": (-50.0, -1.0),
    #     "rew_work_penalty": (-1e-2, -1e-4),
    # },
    name="screwdriver",
)


@TASK_REGISTRY.register(
    name="screwdriver",
    modifiers=[
        ScrewdriverRobotMod(
            robot="xhand1",
            priv_sensor_cfg_name="fingertips",
            track_link_idx="obj",
        ),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        DexHandRslRlRunnerMod(
            runner_cfg=RUNNER_CFG,
            ppo_options=PPO_OPTIONS,
            distill_options=DISTILL_OPTIONS,
            mlp_cfg=MLP_CFG,
            actor_distribution=RslRlGaussianDistributionOptions(init_std=1.0),
        ),
    ],
)
class ScrewdriverConfig(EdenRLConfig):
    """Free-moving heterogeneous screwdriver rotation task."""

    env_options: EnvOptions = EnvOptions(
        num_envs=8192,
        num_eval_envs=16,
        env_spacing=(0.5, 0.5),
        episode_length_s=20.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        use_gjk_collision=True,
        enable_multi_contact=True,
    )

    cameras_options: CamerasOptions = CamerasOptions(
        rec=CameraOptions(
            cam_pos=(-0.780822, -0.083966, 0.697496),
            cam_lookat=(1.918845, 1.435871, 0.0),
            cam_up=(0.0, 0.0, 1.0),
            cam_fov=60.0,
        ),
    )

    scene_options: SceneOptions = SceneOptions(
        plane=PlaneOptions(),
        obj=FatScrewdrivers(),
    )

    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors is set by TactileSensorsMod
        # priv_surface_distance_* is set by InHandRotateRobotMod
        # priv_contact_* is set by InHandRotateRobotMod
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=PARTIAL_HAND_CONTROLLER,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        tactile_sensors=ObservationGroupOptions(
            tactile_sensors=TactileSensorRead.configure(
                sensor_names=["tactile_*"],
            ),
        ),
        proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos,
                params={"entity_name": "robot", "offset_from_default": False},
                noise=GaussianNoise.configure(std=0.001),
            ),
            dofs_vel=ObsTerm.configure(
                func=en.observations.dofs_vel,
                params={"entity_name": "robot"},
                scale=0.2,
                noise=GaussianNoise.configure(std=0.001),
            ),
            last_action=ObsTerm.configure(
                func=en.observations.last_action,
            ),
        ),
        priv_proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos,
                params={"entity_name": "robot", "offset_from_default": False},
            ),
            dofs_vel=ObsTerm.configure(
                func=en.observations.dofs_vel,
                params={"entity_name": "robot"},
                scale=0.2,
            ),
            last_action=ObsTerm.configure(
                func=en.observations.last_action,
            ),
            contact=en.observations.SensorRead.configure(
                sensor_names=["priv_contact_*"],
            ),
        ),
        priv_obj_state=ObservationGroupOptions(
            obj_pos=ObsTerm.configure(
                func=en.observations.base_pos,
                params={"entity_name": "obj"},
            ),
            obj_rot6d=ObsTerm.configure(
                func=base_rot6d,
                params={"entity_name": "obj"},
            ),
            obj_lin_vel=ObsTerm.configure(
                func=en.observations.base_lin_vel,
                params={"entity_name": "obj"},
            ),
            obj_ang_vel=ObsTerm.configure(
                func=en.observations.base_ang_vel,
                params={"entity_name": "obj"},
                scale=0.2,
            ),
            surface_distance=en.observations.SensorRead.configure(
                sensor_names=["priv_surface_distance_*"],
            ),
        ),
        # Auxiliary distillation targets: privileged scalars the student is
        # trained to predict from its latent (see DISTILL_OPTIONS.auxiliary_losses).
        # Not part of any actor/critic obs group -- targets only.
        aux_obj_tilt=ObservationGroupOptions(
            enable_corruption=False,
            obj_tilt=ScrewObjTiltObs.configure(
                obj_name="obj",
                local_up_axis=(0.0, 1.0, 0.0),
            ),
        ),
        aux_rotation_progress=ObservationGroupOptions(
            enable_corruption=False,
            rotation_progress=ScrewAxisRotationProgressObs.configure(
                obj_name="obj",
                local_axis=(0.0, -1.0, 0.0),
            ),
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        rotation_axis=RotationAxisCommand.configure(
            resampling_time_range=(999.0, 999.0),
            axis_mode="z_only",
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        rotation_reward=RewardTerm.configure(
            func=rotation_reward,
            weight=10.0,
            params={
                "local_axis": (0.0, -1.0, 0.0),
                "angvel_clip_min": -0.2,
                "angvel_clip_max": 0.2,
            },
        ),
        surface_distance_reward=RewardTerm.configure(
            func=surface_distance_reward,
            weight=1.0,
            params={
                "obs_name": "surface_distance",
                "nearest_k": 2,
                "sigma": 0.01,
            },
        ),
        obj_dist_penalty=RewardTerm.configure(
            func=screw_object_dist_penalty,
            weight=-300.0,
            params={
                "entity_name": "obj",
                "margin": 0.01,
            },
        ),
        vertical_alignment_penalty=ScrewVerticalAlignmentPenalty.configure(
            weight=-400.0,
            obj_name="obj",
            tilt_margin=0.0,
        ),
        pose_diff_penalty=ScrewPoseDiffPenalty.configure(
            weight=-1.0,
            entity_name="robot",
        ),
        termination_penalty=RewardTerm.configure(
            func=termination_penalty,
            weight=-100.0,
            params={"termination_names": ("handle_dropped", "nut_stagnation", "obj_drift")},
        ),
        action_rate_penalty=RewardTerm.configure(
            func=en.rewards.action_rate_l2,
            weight=-0.01,
        ),
        joint_limits_penalty=RewardTerm.configure(
            func=en.rewards.dofs_pos_limits,
            weight=-40.0,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-1e-3,
            params={"entity_name": "robot"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(
            func=terminations.time_out,
            time_out=True,
        ),
        handle_dropped=DoneTerm.configure(
            func=obj_tilted_past_threshold,
            time_out=False,
            params={
                "entity_name": "obj",
                "local_up_axis": (0.0, 1.0, 0.0),
                "max_tilt_deg": 60.0,
            },
        ),
        nut_stagnation=ScrewNutStagnation.configure(
            time_out=False,
            entity_name="obj",
            history_len=60,
            stagnation_eps=0.003,
        ),
        obj_drift=DoneTerm.configure(
            func=obj_pos_drift_from_grasp,
            time_out=False,
            params={"entity_name": "obj", "max_distance": 0.05},
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        # objective=MetricTerm.configure(
        #     func=episode_reward_metric,
        #     params={
        #         "reward_names": [
        #             "rotation_reward",
        #             "pose_diff_penalty",
        #             "termination_penalty",
        #             "action_rate_penalty",
        #         ],
        #         "weights": [0.1, -0.1, -1000.0, -0.0001],
        #     },
        # ),
        objective=MetricTerm.configure(
            func=screw_total_rotation_metric,
            params={
                "obj_name": "obj",
                "local_axis": (0.0, -1.0, 0.0),
            },
            # Episode-cumulative quantity: evaluate success once, at episode end.
            metric_mode="reset",
            direction="hib",
            success_threshold=2.0 * PI,  # one full revolution
        ),
        total_rotation=MetricTerm.configure(
            func=screw_total_rotation_metric,
            params={
                "obj_name": "obj",
                "local_axis": (0.0, -1.0, 0.0),
            },
            # Episode-cumulative quantity: evaluate success once, at episode end.
            metric_mode="reset",
            direction="hib",
            success_threshold=2.0 * PI,  # one full revolution
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        load_grasp=LoadGraspPose.configure(
            mode=EventMode.RESET,
            entity_name="robot",
            obj_name="obj",
            grasps_paths={
                "xhand1": "src/assets/grasps/screwdrivers_xhand1_grasps_64.pt",
                "sharpa": "src/assets/grasps/screwdrivers_sharpa_grasps_64.pt",
            },
        ),
        obj_friction=en.events.RandomizeFrictionRatio.configure(
            mode=EventMode.RESET,
            entity_name="obj",
            links_name="*",
            friction_range=(0.2, 1.2),
        ),
        handle_mass=en.events.RandomizeMassShift.configure(
            mode=EventMode.RESET,
            entity_name="obj",
            links_name="*",
            mass_shift_range=(0.0, 0.2),
        ),
        ext_force=en.events.ApplyExternalForce.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(2.0, 4.0),
            entity_name="obj",
            links_name="*",
            force_x_range=(-5.0, 5.0),
            force_y_range=(-5.0, 5.0),
            force_z_range=(-5.0, 0.0),
        ),
        **CONTROLLER_RANDOMIZATIONS,
    )

    runner_options: RslRlBaseRunnerOptions = RslRlBaseRunnerOptions()
