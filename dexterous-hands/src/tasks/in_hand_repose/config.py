"""Configuration for in-hand repose task.

Based loosely on IsaacLab's Isaac-Repose-Cube-Allegro-v0 environment.
Reference: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/inhand/config/allegro_hand/allegro_env_cfg.py
"""

import eden as en
from eden.managers.modifiers.observations import GaussianNoise
from eden.options import (
    ActionManagerOptions,
    CommandManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorOptions,
    SensorsOptions,
    TerminationManagerOptions,
)
from eden.options.learning.rsl_rl import (
    RslRlBaseRunnerOptions,
    RslRlDistillationAlgorithmOptions,
    RslRlPpoAlgorithmOptions,
)
from eden.terms import DoneTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import EdenRLConfig
from genesis import gs
from genesis.utils.geom import euler_to_quat

from entities.objects import OBJECTS_4CM_16
from optimization import DEFAULT_HYPERPARAM_RANGES
from registry import HYPERPARAMS_REGISTRY, TASK_REGISTRY
from shared_terms import (
    CONTROLLER_RANDOMIZATIONS,
    DECIMATION,
    HAND_CONTROLLER,
    PI,
    SIM_DT,
    CachedObs,
    ForceMagnitudePenalty,
    OrientationErrorObs,
    RandomizeFrictionRatioWithObs,
    RandomizeMassShiftWithObs,
    SetSampledBottomAlignedPos,
    TactileSensorRead,
    base_rot6d,
    episode_reward_metric,
    goal_rot6d_diff,
    obj_below_height,
    orientation_success_bonus,
    surface_distance_reward,
    termination_penalty,
    track_orientation_gaussian,
    track_orientation_inv_l2,
    work_penalty,
)
from task_mods import (
    DexHandRslRlRunnerMod,
    ManipulationObjectMod,
    RobotHandWithPrivSensorsMod,
    TactileSensorsMod,
)

from .custom_terms import (
    OrientationProgressReward,
    TimeoutTrackingTargetRotationCommand,
    target_timeout_reset_penalty,
)

OBJECT_POS = (0.0, 0.0, 0.5)

RUNNER_CFG = {
    "num_steps_per_env": 24,
    "max_iterations": 20000,
    "save_interval": 1000,
    "obs_groups": {
        # actor will be set to student/teacher based on stage
        "critic": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
        "student": ["proprio", "goal", "tactile_sensors"],
        "teacher": ["goal", "priv_proprio", "priv_obj_state"],
        "rnd_state": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
    },
}
PPO_OPTIONS = RslRlPpoAlgorithmOptions(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.002,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=0.001,
    schedule="adaptive",
    gamma=0.998,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
DISTILL_OPTIONS = RslRlDistillationAlgorithmOptions(
    num_learning_epochs=5,
    normalize_action_targets=True,
    learning_rate=1.0e-4,
    gradient_length=2,
    max_grad_norm=None,
    optimizer="adam",
    loss_type="mse",
)
RND_CFG = {
    "learning_rate": 1e-3,
    "num_outputs": 8,
    "state_normalization": True,
    "reward_normalization": True,
}
# Per-sensor tactile encoders are selected at the CLI via
# `--tactile_encoder=tactile_cnn|tactile_convrnn` paired with
# `--model=tac_mlp` (see src/model_config.py:TACTILE_ENCODER_CFGS).
HAND_OFFSETS = {
    "xhand1": (0.07, 0.0, -0.025),
    "shadow": (0.06, 0.0, -0.02),
    "sharpa": (0.07, -0.015, -0.02),
    "default": (0.06, 0.0, -0.02),
}


class InHandReposeRobotMod(RobotHandWithPrivSensorsMod):
    """Task-specific robot material, fixed base, and fingertip proximity sensors."""

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        config.scene_options.vis_obj.default_root_pos = (OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2] + 0.15)
        obj_surface = config.scene_options.obj.surface
        if obj_surface is None:
            config.scene_options.vis_obj.surface = gs.surfaces.Plastic(opacity=0.5)
        else:
            config.scene_options.vis_obj.surface = obj_surface.model_copy(update={"opacity": 0.5})
        config.scene_options.obj.scale = 0.04

        offset = HAND_OFFSETS.get(self.robot, HAND_OFFSETS["default"])
        config.scene_options.robot.default_root_pos = tuple(OBJECT_POS[i] + offset[i] for i in range(3))
        config.scene_options.robot.default_root_quat = euler_to_quat((0.0, -90.0, 0.0))
        config.scene_options.robot.is_fixed_base = True
        return config


HYPERPARAMS_REGISTRY.register(
    DEFAULT_HYPERPARAM_RANGES
    | {
        "rew_track_orientation": (0.1, 5.0),
        "rew_obj_force_penalty": (-10.0, -0.1),
        "rew_surface_distance_reward": (0.05, 0.5),
        "rew_action_rate_penalty": (-1e-1, -1e-3),
        "rew_work_penalty": (-1e-3, -1e-5),
        "rew_torque_penalty": (-1e-4, -1e-6),
    },
    name="in_hand_repose",
)


@TASK_REGISTRY.register(
    name="in_hand_repose",
    modifiers=[
        ManipulationObjectMod(obj="cube", entity_name="obj", add_vis_entity=True, primitives=OBJECTS_4CM_16),
        InHandReposeRobotMod(robot="xhand1", priv_sensor_cfg_name="fingertips", track_link_idx="obj"),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        DexHandRslRlRunnerMod(
            runner_cfg=RUNNER_CFG,
            ppo_options=PPO_OPTIONS,
            distill_options=DISTILL_OPTIONS,
            rnd_cfg=RND_CFG,
        ),
    ],
)
class InHandReposeConfig(EdenRLConfig):
    """
    Reorient an object in hand to match a target orientation.
    """

    env_options: EnvOptions = EnvOptions(
        num_envs=8192,
        num_eval_envs=16,
        env_spacing=(0.30, 0.25),
        episode_length_s=20.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        use_gjk_collision=True,
        max_collision_pairs=30,
        multiplier_collision_broad_phase=12,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = SceneOptions(
        # obj is set by ManipulationObjectMod
        # vis_obj is set by ManipulationObjectMod
        # robot is set by InHandReposeRobotMod
    )
    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors is set by TactileSensorsMod
        # priv_surface_distance_* is set by InHandReposeRobotMod
        # priv_contact_* is set by InHandReposeRobotMod
        obj_force=SensorOptions(
            sensor=gs.sensors.ContactForce(
                draw_debug=True,
            ),
            attach_entity_name="obj",
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=HAND_CONTROLLER,
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        tactile_sensors=ObservationGroupOptions(
            tactile_sensors=TactileSensorRead.configure(
                sensor_names=["tactile_*"],
            ),
        ),
        goal=ObservationGroupOptions(
            goal_pose=ObsTerm.configure(
                func=en.observations.generated_commands,
                params={"command_name": "goal_rot"},
            ),
            goal_rot6d_diff=ObsTerm.configure(
                func=goal_rot6d_diff,
                params={
                    "command_name": "goal_rot",
                    "entity_name": "obj",
                },
            ),
            goal_angular_dist=OrientationErrorObs.configure(
                command_name="goal_rot",
            ),
        ),
        proprio=ObservationGroupOptions(
            dofs_pos=ObsTerm.configure(
                func=en.observations.dofs_pos,
                params={"entity_name": "robot", "offset_from_default": False},
                noise=GaussianNoise.configure(std=0.005),
            ),
            dofs_vel=ObsTerm.configure(
                func=en.observations.dofs_vel,
                params={"entity_name": "robot"},
                scale=0.2,
                noise=GaussianNoise.configure(std=0.01),
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
        ),
        priv_obj_props=ObservationGroupOptions(
            obj_friction=CachedObs.configure(
                default_value=1.0,
            ),
            obj_mass=CachedObs.configure(
                default_value=0.1,
            ),
            surface_distance=en.observations.SensorRead.configure(
                sensor_names=["priv_surface_distance_*"],
            ),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        success_bonus=RewardTerm.configure(
            func=orientation_success_bonus,
            weight=500.0,
            params={
                "command_name": "goal_rot",
            },
        ),
        track_orientation_gaussian=RewardTerm.configure(
            func=track_orientation_gaussian,
            weight=2.0,
            params={
                "command_name": "goal_rot",
                "obs_term_name": "goal_angular_dist",
                "sigma": 0.4,
            },
        ),
        track_orientation_inv_l2=RewardTerm.configure(
            func=track_orientation_inv_l2,
            weight=2.0,
            params={
                "command_name": "goal_rot",
                "rot_eps": 1e-3,
            },
        ),
        rotation_progress=OrientationProgressReward.configure(
            weight=0.2,
            command_name="goal_rot",
            clip=(-15.0, 5.0),
            negative_scale=-2.0,
        ),
        obj_force_penalty=ForceMagnitudePenalty.configure(
            weight=-5.0,
            sensor_name="obj_force",
            threshold=5.0,
            clip=20.0,
        ),
        is_dropped=RewardTerm.configure(
            func=termination_penalty,
            weight=-100.0,
            params={
                "termination_names": ("obj_dropped",),
            },
        ),
        goal_timeout_penalty=RewardTerm.configure(
            func=target_timeout_reset_penalty,
            weight=-200.0,
            params={"command_name": "goal_rot"},
        ),
        surface_distance_reward=RewardTerm.configure(
            func=surface_distance_reward,
            weight=0.5,
            params={
                "obs_name": "surface_distance",
                "nearest_k": 2,
                "sigma": 0.05,
            },
        ),
        action_rate_penalty=RewardTerm.configure(
            func=en.rewards.action_rate_l2,
            weight=-1e-5,
        ),
        joint_limits_penalty=RewardTerm.configure(
            func=en.rewards.dofs_pos_limits,
            weight=-20.0,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-1e-3,
            params={"entity_name": "robot"},
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        objective=MetricTerm.configure(
            func=episode_reward_metric,
            params={
                "reward_names": ["success_bonus", "is_dropped"],
                "weights": [1.0, -10.0],
            },
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(func=en.terminations.time_out, time_out=True),
        obj_dropped=DoneTerm.configure(
            func=obj_below_height,
            params={
                "entity_name": "obj",
                "threshold": OBJECT_POS[2] - 0.04,
            },
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        goal_rot=TimeoutTrackingTargetRotationCommand.configure(
            entity_name="obj",
            goal_entity_name="vis_obj",
            x_range=(-PI / 4, PI / 4),
            y_range=(-PI / 4, PI / 4),
            sample_relative=True,
            update_goal_on_success=True,
            orientation_success_threshold=0.1,
            resampling_time_range=(4.0, 6.0),
        )
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_obj=SetSampledBottomAlignedPos.configure(
            mode=en.EventMode.RESET,
            entity_name="obj",
            pos_z=OBJECT_POS[2],
            range_x=(-0.02, 0.02),
            range_y=(-0.02, 0.02),
            range_yaw=(-PI, PI),
        ),
        obj_friction=RandomizeFrictionRatioWithObs.configure(
            mode=en.EventMode.STARTUP,
            obs_name="obj_friction",
            entity_name="obj",
            links_name="*",
            friction_range=(0.3, 2.0),
        ),
        obj_mass=RandomizeMassShiftWithObs.configure(
            mode=en.EventMode.STARTUP,
            obs_name="obj_mass",
            entity_name="obj",
            links_name="*",
            mass_shift_range=(0.0, 0.1),
        ),
        ext_force=en.events.ApplyExternalForce.configure(
            mode=en.EventMode.INTERVAL,
            interval_range_s=(2.0, 4.0),
            entity_name="obj",
            links_name="*",
            force_x_range=(-1.0, 1.0),
            force_y_range=(-1.0, 1.0),
            force_z_range=(-1.0, 1.0),
        ),
        **CONTROLLER_RANDOMIZATIONS,
    )

    runner_options: RslRlBaseRunnerOptions = RslRlBaseRunnerOptions(
        # runner is set by DexHandRslRlRunnerMod
    )
