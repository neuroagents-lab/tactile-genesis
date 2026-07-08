"""Configuration for rummaging a bin of objects to find the target hot object."""

from __future__ import annotations

import eden as en
import genesis as gs
import genesis.utils.geom as gu
from eden.constants import EventMode
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
from eden.options.materials import RigidMaterialOptions
from eden.terms import DoneTerm, EventTerm, MetricTerm, ObsTerm, RewardTerm
from eden.utils.configs import EdenRLConfig

from entities.objects import BallyCube, Bin
from optimization import DEFAULT_HYPERPARAM_RANGES
from registry import HYPERPARAMS_REGISTRY, TASK_REGISTRY
from shared_terms import (
    DECIMATION,
    HAND_CONTROLLER,
    SIM_DT,
    RootPoseController,
    TactileSensorRead,
    ee_work_penalty,
    episode_reward_metric,
    obj_below_height,
    surface_distance_reward,
    work_penalty,
)
from task_mods import DexHandRslRlRunnerMod, RobotHandWithPrivSensorsMod, TactileSensorsMod

from .custom_terms import (
    BallGraspSequenceCommand,
    ContactDurationObs,
    CurrentTargetPositionObs,
    RandomlyPlaceInGrid,
    TargetLiftReward,
    TemperatureDiffReading,
    command_hot_success,
    command_hot_success_reward,
    max_temperature_metric,
    target_contact_decay_reward,
)

# --- Scene geometry (meters) ---
BIN_SIZE = 0.5
BIN_HEIGHT = BIN_SIZE / 2.0
BIN_QUAT = gu.euler_to_quat((0.0, 15.0, 0.0))

BALL_SIZE = 0.08
BALL_HALF_EXTENT = BALL_SIZE / 2.0
BALL_GRID_SPACING = BALL_SIZE
TEMP_GRID_SIZE = (1, 1, 1)
NUM_BALLS = 8

# Entity names of every ball in the bin: the hot target ("obj") plus the decoys.
BALL_ENTITY_NAMES = ("obj", *(f"ball_{i}" for i in range(NUM_BALLS - 1)))

BIN_HALF_INTERIOR = BIN_SIZE / 2.0 - BALL_HALF_EXTENT


AMBIENT_TEMPERATURE = 22.0  # Celsius
TARGET_TEMPERATURE = 45.0  # Celsius

TEMPERATURE_HISTORY_LENGTH = 4

# Time the current target ball must stay lifted to advance (or to succeed,
# when the current target is the hot ball).
HOLD_SECONDS = 1.0

# How far above its episode-start z the current target ball must rise to count
# as "lifted" (meters).
LIFT_DELTA_M = 0.01

# Surface distance under which a fingertip counts as touching a ball.
CONTACT_THRESHOLD_M = 0.001


RUNNER_CFG = {
    "num_steps_per_env": 24,
    "max_iterations": 10000,
    "save_interval": 500,
    "clip_actions": 10.0,
    "obs_groups": {
        # Student senses heat (temperature) + tactile; teacher gets the
        # grasp-target command (which ball to pick up next) plus proximity and
        # per-fingertip contact duration so it can execute the policy.
        "student": ["proprio", "tactile_sensors", "temp_sensors"],
        "teacher": ["priv_proprio", "proximity", "grasp_target", "contact_duration"],
        "critic": ["priv_proprio", "priv_obj_state", "grasp_target", "contact_duration"],
        "rnd_state": ["priv_proprio", "priv_obj_state"],
    },
}
PPO_OPTIONS = RslRlPpoAlgorithmOptions(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=5.0e-4,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
)
DISTILL_OPTIONS = RslRlDistillationAlgorithmOptions(
    num_learning_epochs=5,
    learning_rate=1.0e-4,
    gradient_length=2,
    max_grad_norm=None,
    optimizer="adam",
    loss_type="mse",
)


class RummageHotTaskMod(RobotHandWithPrivSensorsMod):
    """Robot after static scene (predictable link indices); hand pose, sensors, and reset events."""

    def apply(self, config: EdenRLConfig) -> EdenRLConfig:
        config = super().apply(config)

        config.scene_options.robot.is_fixed_base = False
        config.scene_options.robot.default_root_pos = (-0.1, 0.0, BIN_HEIGHT + 0.1)
        config.scene_options.robot.default_root_quat = gu.euler_to_quat((0.0, 110.0, 0.0))

        temperature_props: dict[int, gs.sensors.TemperatureProperties] = {
            # hand links
            -1: gs.sensors.TemperatureProperties(
                base_temperature=AMBIENT_TEMPERATURE,
                conductivity=150.0,
                density=2000.0,
                specific_heat=500.0,  # lower to make more responsive to temperature changes
                emissivity=0.85,
            ),
            # bin
            0: gs.sensors.TemperatureProperties(
                base_temperature=AMBIENT_TEMPERATURE,
                conductivity=100.0,
                density=2000.0,
                specific_heat=1000.0,
                emissivity=0.4,
            ),
            # target hot obj
            1: gs.sensors.TemperatureProperties(
                base_temperature=TARGET_TEMPERATURE,
                conductivity=200.0,
                density=3000.0,
                specific_heat=1000.0,
                emissivity=0.4,
            ),
        }

        for link_name in config.scene_options.robot.metadata.finger_links:
            setattr(
                config.sensors_options,
                "temp_" + link_name,
                SensorOptions(
                    sensor=gs.sensors.TemperatureGrid(
                        grid_size=TEMP_GRID_SIZE,
                        properties_dict=temperature_props,
                        ambient_temperature=AMBIENT_TEMPERATURE,
                        convection_coefficient=0.0,
                        simulate_all_link_temperatures=False,
                        debug_temperature_range=(AMBIENT_TEMPERATURE - 5.0, TARGET_TEMPERATURE),
                        draw_debug=True,
                    ),
                    attach_entity_name="robot",
                    attach_link_name=link_name,
                ),
            )

        config.scene_options.obj.surface = gs.surfaces.Default(
            diffuse_texture=gs.textures.ColorTexture(color=(0.9, 0.1, 0.2, 1.0))
        )
        config.scene_options.bin.surface = gs.surfaces.Default(
            diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.5, 0.5, 1.0))
        )
        for i in range(NUM_BALLS - 1):
            setattr(
                config.scene_options,
                f"ball_{i}",
                BallyCube(
                    scale=BALL_SIZE,
                    surface=gs.surfaces.Default(diffuse_texture=gs.textures.ColorTexture(color=(0.3, 0.5, 0.9))),
                ),
            )

        # Per-(fingertip, ball) surface-distance probes: one probe per pair so
        # the state machines can recover which ball each fingertip is near
        # (the aggregated `priv_surface_distance_<link>` probes collapse to
        # min-across-all-balls and lose ball identity).
        fingertip_cfg = config.scene_options.robot.metadata.priv_sensor_cfgs["fingertips"]
        for link_name, offset in fingertip_cfg.items():
            for ball_name in BALL_ENTITY_NAMES:
                setattr(
                    config.sensors_options,
                    f"priv_ftball_dist__{link_name}__{ball_name}",
                    SensorOptions(
                        sensor=gs.sensors.SurfaceDistanceProbe(
                            probe_local_pos=(offset,),
                            probe_radius=0.5,
                            track_link_idx=(0,),
                            draw_debug=False,
                        ),
                        attach_entity_name="robot",
                        attach_link_name=link_name,
                        track_link_names=[ball_name],
                    ),
                )
        return config


HYPERPARAMS_REGISTRY.register(
    DEFAULT_HYPERPARAM_RANGES
    | {
        "rew_target_advance": (1.0, 20.0),
        "rew_success_bonus": (50.0, 500.0),
        "rew_lift": (2.0, 50.0),
        "rew_surface_distance_reward": (0.005, 0.3),
        "rew_action_rate_penalty": (-0.02, -0.001),
        "rew_joint_pos_limits_penalty": (-50.0, -1.0),
    },
    name="rummage_hot",
)


@TASK_REGISTRY.register(
    name="rummage_hot",
    modifiers=[
        RummageHotTaskMod(robot="xhand1", track_link_idx=BALL_ENTITY_NAMES, priv_sensor_cfg_name="tips+palm"),
        TactileSensorsMod(sensors="none", track_link_idx="obj"),
        DexHandRslRlRunnerMod(
            runner_cfg=RUNNER_CFG,
            ppo_options=PPO_OPTIONS,
            distill_options=DISTILL_OPTIONS,
        ),
    ],
)
class RummageHotConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(
        num_envs=4096,
        num_eval_envs=16,
        env_spacing=(BIN_SIZE + 0.2, BIN_SIZE + 0.2),
        episode_length_s=20.0,
        sim_dt=SIM_DT,
        sim_substeps=1,
        decimation=DECIMATION,
        max_collision_pairs=64,
        multiplier_collision_broad_phase=10,
    )

    scene_options: SceneOptions = SceneOptions(
        bin=Bin(
            scale=0.5,
            default_root_quat=BIN_QUAT,
            is_fixed_base=True,
            material=RigidMaterialOptions(rho=500.0, friction=0.01),
        ),
        obj=BallyCube(
            scale=BALL_SIZE,
        ),
    )
    sensors_options: SensorsOptions = SensorsOptions(
        # tactile_sensors set by TactileSensorsMod
        # priv_surface_distance_* / priv_contact_* set by RobotHandWithPrivSensorsMod
        #   (surface-distance probes track every ball, see BALL_ENTITY_NAMES)
        # temp_* set by RummageHotTaskMod
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
        # Student-only: temperature sensors on the finger links.
        temp_sensors=ObservationGroupOptions(
            temp_sensors=TemperatureDiffReading.configure(
                sensor_names=["temp_*"],
                scale=5.0,
            ),
            history_length=TEMPERATURE_HISTORY_LENGTH,
        ),
        # Teacher-only: per-fingertip surface distance to the nearest ball.
        proximity=ObservationGroupOptions(
            proximity=en.observations.SensorRead.configure(
                sensor_names=["priv_surface_distance_*"],
            ),
        ),
        # Teacher-only: the grasp-target command — one-hot of the currently
        # commanded ball, plus that ball's position relative to the robot base.
        grasp_target=ObservationGroupOptions(
            target_idx=ObsTerm.configure(
                func=en.observations.generated_commands,
                params={"command_name": "grasp_target"},
            ),
            target_pos=CurrentTargetPositionObs.configure(
                robot_name="robot",
                command_name="grasp_target",
                ball_names=BALL_ENTITY_NAMES,
            ),
        ),
        # Teacher/critic-only: per-fingertip continuous-contact duration plus a
        # hot-obj-touch flag, so the windowed ball_contact reward is observable.
        contact_duration=ObservationGroupOptions(
            contact_duration=ContactDurationObs.configure(
                robot_name="robot",
                ball_names=BALL_ENTITY_NAMES,
                contact_threshold=CONTACT_THRESHOLD_M,
            ),
        ),
        priv_obj_state=ObservationGroupOptions(
            obj_pos=ObsTerm.configure(
                func=en.observations.base_pos,
                params={"entity_name": "obj"},
            ),
            obj_lin_vel=ObsTerm.configure(
                func=en.observations.base_lin_vel,
                params={"entity_name": "obj"},
            ),
            surface_distance=en.observations.SensorRead.configure(
                sensor_names=["priv_surface_distance_*"],
            ),
        ),
    )

    action_options: ActionManagerOptions = ActionManagerOptions(
        dofs_pos_controller=HAND_CONTROLLER,
        ee_controller=RootPoseController.configure(
            entity_name="robot",
            dofs_name=["*"],
            kp=100.0,
            kd=20.0,
            pos_x_range=(-BIN_HALF_INTERIOR, BIN_HALF_INTERIOR - BALL_SIZE * 2.0),
            pos_y_range=(-BIN_HALF_INTERIOR + BALL_SIZE, BIN_HALF_INTERIOR - BALL_SIZE),
            pos_z_range=(BALL_SIZE, BIN_HEIGHT + BALL_SIZE * 2.0),
            euler_x_range=(-45.0, 45.0),
            euler_y_range=(-45.0, 45.0),
            euler_z_range=(-45.0, 45.0),
        ),
    )

    reward_options: RewardManagerOptions = RewardManagerOptions(
        target_advance=RewardTerm.configure(
            func=target_contact_decay_reward,
            weight=20.0,
            params={
                "command_name": "grasp_target",
                "hold_seconds": HOLD_SECONDS,
            },
        ),
        success_bonus=RewardTerm.configure(
            func=command_hot_success_reward,
            weight=200.0,
            params={"command_name": "grasp_target"},
        ),
        lift=TargetLiftReward.configure(
            command_name="grasp_target",
            ball_names=BALL_ENTITY_NAMES,
            lift_delta=LIFT_DELTA_M,
            weight=40.0,
        ),
        surface_distance_reward=RewardTerm.configure(
            func=surface_distance_reward,
            weight=0.1,
            params={
                "obs_name": "surface_distance",
                "nearest_k": 4,
                "sigma": 0.1,
            },
        ),
        action_rate_penalty=RewardTerm.configure(
            func=en.rewards.action_rate_l2,
            weight=-5e-3,
        ),
        joint_pos_limits_penalty=RewardTerm.configure(
            func=en.rewards.dofs_pos_limits,
            weight=-40.0,
            params={"entity_name": "robot"},
        ),
        work_penalty=RewardTerm.configure(
            func=work_penalty,
            weight=-1e-4,
            params={"entity_name": "robot"},
        ),
        ee_work_penalty=RewardTerm.configure(
            func=ee_work_penalty,
            weight=-1e-6,
            params={"entity_name": "robot"},
        ),
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        objective=MetricTerm.configure(
            func=episode_reward_metric,
            params={
                "reward_names": ["success_bonus", "target_advance", "lift", "action_rate_penalty"],
                "weights": [1.0, 1.0, 1.0, -1e-4],
            },
        ),
        temperature=MetricTerm.configure(
            func=max_temperature_metric,
            params={"obs_name": "temp_sensors"},
        ),
    )

    termination_options: TerminationManagerOptions = TerminationManagerOptions(
        timeout=DoneTerm.configure(
            func=en.terminations.time_out,
            time_out=True,
        ),
        obj_fallen=DoneTerm.configure(
            func=obj_below_height,
            time_out=False,
            params={
                "entity_name": "obj",
                "threshold": -0.2,
            },
        ),
        hot_obj_lifted=DoneTerm.configure(
            func=command_hot_success,
            time_out=False,
            params={"command_name": "grasp_target"},
        ),
    )

    command_options: CommandManagerOptions = CommandManagerOptions(
        grasp_target=BallGraspSequenceCommand.configure(
            robot_name="robot",
            ball_names=BALL_ENTITY_NAMES,
            hot_ball_idx=BALL_ENTITY_NAMES.index("obj"),
            lift_delta=LIFT_DELTA_M,
            hold_seconds=HOLD_SECONDS,
            contact_threshold=CONTACT_THRESHOLD_M,
        ),
    )

    event_options: EventManagerOptions = EventManagerOptions(
        scatter_bin_contents=RandomlyPlaceInGrid.configure(
            mode=EventMode.RESET,
            entity_names=["ball_*", "obj"],
            range_x=(-BIN_HALF_INTERIOR, BIN_HALF_INTERIOR),
            range_y=(-BIN_HALF_INTERIOR, BIN_HALF_INTERIOR),
            range_z=(0.0, BIN_HEIGHT),
            bin_quat=BIN_QUAT,
            spacing=BALL_GRID_SPACING,
            jiggle_std=0.005,
        ),
        reset_hand_dofs=EventTerm.configure(
            func=en.events.reset_dofs_by_offset,
            mode=EventMode.RESET,
            params={
                "entity_name": "robot",
                "dofs_pos_range": (-0.02, 0.02),
            },
        ),
    )

    runner_options: RslRlBaseRunnerOptions = RslRlBaseRunnerOptions(
        # runner is set by RslRlRunnerMod
    )
