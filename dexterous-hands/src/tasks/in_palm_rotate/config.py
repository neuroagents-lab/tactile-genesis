"""Configuration for in-palm rotate task.

A partial-hand in-palm rotation task where only the thumb and middle finger
are active; the index, ring, and pinky fingers are frozen. Uses the
``PARTIAL_HAND_CONTROLLER`` (with ``scale_ratio=0.5``) so the policy controls
only the active fingers while the frozen DOFs hold their reset pose.

Based loosely on dexterous manipulation paper from ByteDance: https://arxiv.org/pdf/2601.02778
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
from registry import HYPERPARAMS_REGISTRY, TASK_REGISTRY
from shared_terms import (
    CONTROLLER_RANDOMIZATIONS,
    DECIMATION,
    PARTIAL_HAND_CONTROLLER,
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
    filter_hand_dof_names,
    obj_below_height,
    object_dist_penalty,
    orientation_success_bonus,
    surface_distance_reward,
    termination_penalty,
    work_penalty,
)
from task_mods import (
    DexHandRslRlRunnerMod,
    ManipulationObjectMod,
    RewardWeightCurriculumMod,
    RobotHandWithPrivSensorsMod,
    TactileSensorsMod,
)

from .custom_terms import (
    GatedAxisRotationProgressReward,
    ObjectSizeObs,
    SteppingRotationCommand,
    off_axis_orientation_penalty,
    reached_max_consecutive_successes,
)

OBJECT_POS = (0.0, 0.0, 0.5)
OBJECT_QUAT = euler_to_quat((0.0, 0.0, 0.0))

RUNNER_CFG = {
    "num_steps_per_env": 24,
    "max_iterations": 6000,
    "save_interval": 500,
    "obs_groups": {
        # actor will be set to student/teacher based on stage
        "critic": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
        "student": ["proprio", "tactile_sensors"],
        "teacher": ["goal", "priv_proprio", "priv_obj_state"],
        "rnd_state": ["goal", "priv_proprio", "priv_obj_state", "priv_obj_props"],
    },
}
CURRICULUM_CFG = {
    "curriculum": {
        "rotation_progress": 1.0,
        "is_dropped": -30.0,
    },
    "curriculum_step_start": 1000,
    "curriculum_step_end": 5000,
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
    max_grad_norm=2.0,
    optimizer="adam",
    loss_type="mse",
    auxiliary_losses=[
        {
            "class_name": "rsl_rl.extensions.auxiliary:StudentLatentPredictionLoss",
            "target_obs": "aux_obj_size",
            "name": "obj_size",
            "weight": 1.0,
            "target_scale": 20.0,
            "hidden_dims": [128, 64],
            "loss_type": "mse",
        },
        {
            "class_name": "rsl_rl.extensions.auxiliary:StudentLatentPredictionLoss",
            "target_obs": "aux_goal_dist",
            "name": "goal_dist",
            "weight": 1.0,
            "hidden_dims": [128, 64],
            "loss_type": "mse",
        },
    ],
)
RND_CFG = {
    "learning_rate": 1e-3,
    "num_outputs": 8,
    "state_normalization": True,
    "reward_normalization": True,
}
HAND_OFFSETS = {
    "xhand1": (0.07, 0.0, -0.025),
    "shadow": (0.06, 0.0, -0.02),
    "sharpa": (0.08, -0.015, -0.02),
    "default": (0.06, 0.0, -0.02),
}
FROZEN_FINGERS = {
    "xhand1": ("index", "ring", "pinky"),
    "sharpa": ("ring", "middle"),
    "default": (),
}
OBJ_SIZE = 0.04


class InPalmRotateRobotMod(RobotHandWithPrivSensorsMod):
    def _configure_robot_and_actions(self, config: EdenRLConfig, robot_instance):
        """Freeze the index/ring/pinky DOFs so only the thumb and middle finger act."""
        frozen_fingers = FROZEN_FINGERS.get(self.robot, FROZEN_FINGERS["default"])
        active_dofs = filter_hand_dof_names(robot_instance.dofs_name, excluded_fingers=frozen_fingers)
        frozen_dofs = tuple(name for name in robot_instance.dofs_name if name not in active_dofs)
        config.action_options.dofs_pos_controller = config.action_options.dofs_pos_controller.model_copy(
            update={
                "frozen_dofs": frozen_dofs,
                "dofs_name": list(active_dofs),
            }
        )
        # Restrict the reset dof-pos randomizer to the active DOFs so the frozen
        # fingers keep their canonical default pose.
        if hasattr(config.event_options, "set_dofs_pos"):
            config.event_options.set_dofs_pos = config.event_options.set_dofs_pos.model_copy(
                update={"dofs_name": tuple(active_dofs)}
            )
        return robot_instance

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)
        if hasattr(config.scene_options, "vis_obj"):
            config.scene_options.vis_obj.default_root_pos = (OBJECT_POS[0], OBJECT_POS[1], OBJECT_POS[2] + 0.15)
            obj_surface = config.scene_options.obj.surface
            if obj_surface is None:
                config.scene_options.vis_obj.surface = gs.surfaces.Plastic(opacity=0.5)
            else:
                config.scene_options.vis_obj.surface = obj_surface.model_copy(update={"opacity": 0.5})
            config.scene_options.vis_obj.scale = OBJ_SIZE

        offset = HAND_OFFSETS.get(self.robot, HAND_OFFSETS["default"])
        config.scene_options.robot.default_root_pos = tuple(OBJECT_POS[i] + offset[i] for i in range(3))
        config.scene_options.robot.default_root_quat = euler_to_quat((0.0, -90.0, 0.0))
        config.scene_options.robot.is_fixed_base = True

        config.scene_options.obj.scale = OBJ_SIZE
        return config


HYPERPARAMS_REGISTRY.register(
    {
        "rew_success_bonus": (10.0, 200.0),
        "rew_rotation_progress": (0.5, 10.0),
        "rew_is_dropped": (-200.0, -1.0),
        "rew_off_axis_orientation_penalty": (-2.0, -0.1),
        "rew_obj_dist_penalty": (-500.0, -1.0),
    },
    name="in_palm_rotate",
)


@TASK_REGISTRY.register(
    name="in_palm_rotate",
    modifiers=[
        ManipulationObjectMod(obj="cube", add_vis_entity=True, primitives=OBJECTS_4CM_16),
        InPalmRotateRobotMod(robot="xhand1", priv_sensor_cfg_name="fingertips", track_link_idx="obj"),
        RewardWeightCurriculumMod(curriculum_cfg=CURRICULUM_CFG),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        DexHandRslRlRunnerMod(
            runner_cfg=RUNNER_CFG,
            ppo_options=PPO_OPTIONS,
            distill_options=DISTILL_OPTIONS,
            rnd_cfg=RND_CFG,
        ),
    ],
)
class InPalmRotateConfig(EdenRLConfig):
    """
    Rotate an object resting on the palm of the robot hand along the target axis,
    using only the thumb and middle finger (index/ring/pinky are frozen).

    The orientation command is a shifting curriculum: each goal is a fixed body-axis
    step (default 90°) from the object's current pose, and advances again on success.
    """

    env_options: EnvOptions = EnvOptions(
        num_envs=8192,
        num_eval_envs=16,
        env_spacing=(0.30, 0.25),
        episode_length_s=10.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        # noslip_iterations=4,
        use_gjk_collision=True,
        max_collision_pairs=48,
        multiplier_collision_broad_phase=16,
        enable_multi_contact=True,
    )

    scene_options: SceneOptions = SceneOptions(
        # obj is set by ManipulationObjectMod
        # robot is set by InPalmRotateRobotMod
    )
    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors is set by TactileSensorsMod
        # priv_surface_distance_* is set by InPalmRotateRobotMod
        # priv_contact_* is set by InPalmRotateRobotMod
        obj_force=SensorOptions(
            sensor=gs.sensors.ContactForce(
                draw_debug=True,
            ),
            attach_entity_name="obj",
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        # PARTIAL_HAND_CONTROLLER with scale_ratio=0.5; frozen_dofs/dofs_name set by InPalmRotateRobotMod
        dofs_pos_controller=PARTIAL_HAND_CONTROLLER.model_copy(update={"scale_ratio": 0.5}),
    )

    observation_options: ObservationManagerOptions = ObservationManagerOptions(
        tactile_sensors=ObservationGroupOptions(
            tactile_sensors=TactileSensorRead.configure(
                sensor_names=["tactile_*"],
            ),
        ),
        goal=ObservationGroupOptions(
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
                noise=GaussianNoise.configure(std=0.005),
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
        aux_obj_size=ObservationGroupOptions(
            enable_corruption=False,
            obj_size=ObjectSizeObs.configure(
                entity_name="obj",
            ),
        ),
        aux_goal_dist=ObservationGroupOptions(
            enable_corruption=False,
            goal_angular_dist=OrientationErrorObs.configure(
                command_name="goal_rot",
            ),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        success_bonus=RewardTerm.configure(
            func=orientation_success_bonus,
            weight=150.0,
            params={
                "command_name": "goal_rot",
            },
        ),
        rotation_progress=GatedAxisRotationProgressReward.configure(
            weight=2.0,
            command_name="goal_rot",
            penalty_scale=5.0,
            clip=6.0,
        ),
        surface_distance_reward=RewardTerm.configure(
            func=surface_distance_reward,
            weight=0.1,
            params={
                "obs_name": "surface_distance",
                "nearest_k": 1,
                "sigma": 0.04,
            },
        ),
        is_dropped=RewardTerm.configure(
            func=termination_penalty,
            weight=-100.0,
            params={
                "termination_names": ("obj_dropped",),
            },
        ),
        off_axis_orientation_penalty=RewardTerm.configure(
            func=off_axis_orientation_penalty,
            weight=-1.0,
            params={
                "command_name": "goal_rot",
            },
        ),
        obj_dist_penalty=RewardTerm.configure(
            func=object_dist_penalty,
            weight=-100.0,
            params={
                "entity_name": "obj",
                "target_pos": OBJECT_POS,
                "margin": 0.02,
            },
        ),
        obj_force_penalty=ForceMagnitudePenalty.configure(
            weight=-5.0,
            sensor_name="obj_force",
            threshold=1.0,
            clip=20.0,
        ),
        action_rate_penalty=RewardTerm.configure(
            func=en.rewards.action_rate_l2,
            weight=-5e-4,
        ),
        joint_limits_penalty=RewardTerm.configure(
            func=en.rewards.dofs_pos_limits,
            weight=-40.0,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-5e-3,
            params={"entity_name": "robot"},
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        objective=MetricTerm.configure(
            func=episode_reward_metric,
            params={
                "reward_names": [
                    "success_bonus",
                    "rotation_progress",
                    "action_rate_penalty",
                    "work_penalty",
                    "is_dropped",
                ],
                "weights": [100.0, 0.1, -1e-5, -1e-5, -1000.0],
            },
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        time_out=DoneTerm.configure(
            func=en.terminations.time_out,
            time_out=True,
        ),
        obj_dropped=DoneTerm.configure(
            func=obj_below_height,
            params={
                "entity_name": "obj",
                "threshold": OBJECT_POS[2] - 0.05,
            },
        ),
        reached_max_consecutive_successes=DoneTerm.configure(
            func=reached_max_consecutive_successes,
            params={
                "command_name": "goal_rot",
                "max_consecutive_successes": 15.0,
            },
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        goal_rot=SteppingRotationCommand.configure(
            entity_name="obj",
            goal_entity_name="vis_obj",
            update_goal_on_success=True,
            orientation_success_threshold=0.1,
            allowed_off_axis_error=PI / 6,
            step_rad=PI / 2,
            rotation_axis_world=(-1.0, 0.0, 0.0),
            resampling_time_range=(1e9, 1e9),  # Goals advance on success only
        )
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_obj=SetSampledBottomAlignedPos.configure(
            mode=en.EventMode.RESET,
            entity_name="obj",
            pos_z=OBJECT_POS[2],
            range_x=(-0.02, 0.02),
            range_y=(-0.02, 0.02),
            range_yaw=(-PI / 12, PI / 12),
        ),
        # set_dofs_pos=SetRandomActiveDofsPos.configure(
        #     mode=en.EventMode.RESET,
        #     entity_name="robot",
        #     dofs_pos_range=(0.0, 0.4),
        #     apply_as_ratio=True,
        #     # dofs_name (active DOFs only) is set by InPalmRotateRobotMod
        # ),
        obj_friction=RandomizeFrictionRatioWithObs.configure(
            mode=en.EventMode.RESET,
            obs_name="obj_friction",
            entity_name="obj",
            links_name="*",
            friction_range=(0.5, 1.5),
        ),
        obj_mass=RandomizeMassShiftWithObs.configure(
            mode=en.EventMode.RESET,
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
        # runner is set by RslRlRunnerMod
    )
