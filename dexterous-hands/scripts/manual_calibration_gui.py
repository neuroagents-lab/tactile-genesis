"""Manual dexterous-hand calibration GUI.

Replay a recorded sysid trajectory while interactively tuning the sim
twin's per-joint gains and passive parameters. Works for any registered
hand via ``--robot``; ``--deploy`` (mirroring commands to the real hand)
is XHand1-only because that is the only hand with a deployment class so
far. Without ``--deploy`` the GUI runs the sim twin alone, for any hand.

Example
-------
python scripts/manual_calibration_gui.py --robot xhand1 \
    --trajectory data/xhand_sysid/trajectories/prbs_range20.npz
"""

from __future__ import annotations

import pathlib
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, cast

import eden as en
import genesis as gs
import numpy as np
import torch
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.utils.state import RobotCommand
from eden.extensions.sysid import Trajectory
from eden.extensions.sysid.modifier import apply_parameters
from eden.extensions.sysid.rollout import reset_to_initial_state

from calibration.action_mod_sysid import PROPERTIES, get_dofs_pos_controller, install_action_mod_sysid_patch
from calibration.sysid_config import (
    ParameterSet,
    load_params_yaml,
    make_parameter,
    make_sim_twin_config,
    save_params_yaml,
)
from calibration.sysid_rollout import refresh_pd_gains
from registry import get_argparser, get_task_config_from_args
from tactile_compare import (
    FINGER_NAMES,
    TactileComparisonRecorder,
    apply_isolation,
    finger_indices,
    fingertip_link_order,
    read_real_tactile,
    read_sim_tactile,
    reset_plotter,
    resolve_sim_sensor_names,
    start_tactile_plot,
)

PROPERTY_NAMES = (
    "kp",
    "kd",
    "stiffness",
    "armature",
    "frictionloss",
    "damping",
    "deadband_epsilon",
    "gear_backlash",
    "gear_reversal_threshold",
    "gear_takeup_rate",
    "gear_initial_side",
    "torque_kick",
    "activation_epsilon",
)
PLOT_LABELS = ("commanded", "measured", "sim")


def _to_1d_numpy(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


CommandName = Literal[
    "apply_current",
    "apply_all",
    "command_target",
    "load_trajectory",
    "load_params",
    "random_targets",
    "save_params",
    "reset_replay",
    "reset_env",
    "single_step",
    "zero_targets",
    "isolate_tactile",
]


@dataclass
class PendingCommand:
    name: CommandName
    payload: Any = None


@dataclass
class CalibrationState:
    dof_names: list[str]
    params_out: pathlib.Path
    trajectory_path: str = ""
    params_path: str = ""
    selected_dof: int = 0
    replay_idx: int = 0
    mode_idx: int = 0
    trajectory: Trajectory | None = None
    plot_values: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_positions: np.ndarray | None = None
    dof_lower: np.ndarray | None = None
    dof_upper: np.ndarray | None = None
    values: dict[str, np.ndarray] = field(default_factory=dict)
    commands: deque[PendingCommand] = field(default_factory=deque)
    status: str = "Select a trajectory to begin replay."
    plotter: Any = None
    tactile_recorder: Any = None
    tactile_plotter: Any = None
    sim_sensor_names: dict[int, str] = field(default_factory=dict)
    tactile_enabled: bool = False
    selected_tactile_finger: int = 0
    # Maps each controlled-DOF index -> its index in the real hand's full DOF
    # vector. Set only for a partial/frozen controller; None means 1:1.
    ctrl_to_full: list[int] | None = None

    def __post_init__(self) -> None:
        self.params_path = str(self.params_out)
        if self.target_positions is None:
            self.target_positions = np.zeros(len(self.dof_names), dtype=np.float64)
        if self.dof_lower is None:
            self.dof_lower = np.full(len(self.dof_names), -np.pi, dtype=np.float64)
        if self.dof_upper is None:
            self.dof_upper = np.full(len(self.dof_names), np.pi, dtype=np.float64)
        if not self.values:
            self.values = {name: np.zeros(len(self.dof_names), dtype=np.float64) for name in PROPERTY_NAMES}

    def queue(self, name: CommandName, payload: Any = None) -> None:
        self.commands.append(PendingCommand(name, payload))

    def pop_commands(self) -> list[PendingCommand]:
        commands = list(self.commands)
        self.commands.clear()
        return commands

    @property
    def selected_dof_name(self) -> str:
        return self.dof_names[self.selected_dof]

    @property
    def replay_mode(self) -> bool:
        return self.mode_idx == 0


class NPZFileBrowser:
    def __init__(self, start_dir: pathlib.Path, extension: str) -> None:
        self.open = False
        self.directory = str(start_dir)
        self.extension = extension
        self.target: Literal["trajectory", "params"] = "trajectory"
        self.selected = -1

    def request(self, target: Literal["trajectory", "params"], start_path: str) -> None:
        self.target = target
        path = pathlib.Path(start_path).expanduser()
        if path.is_file():
            self.directory = str(path.parent)
        elif path.is_dir():
            self.directory = str(path)
        self.selected = -1
        self.open = True

    def draw(self, imgui, state: CalibrationState) -> None:
        if not self.open:
            return

        title = f"Select {self.extension} file##manual_calibration_file_browser"
        imgui.open_popup(title)
        imgui.set_next_window_size((560, 420))
        visible, _ = imgui.begin_popup_modal(title)
        if not visible:
            self.open = False
            return

        if imgui.button("^##parent_dir"):
            parent = pathlib.Path(self.directory).parent
            if str(parent) != self.directory:
                self.directory = str(parent)
                self.selected = -1
        imgui.same_line()
        imgui.text(self.directory)
        imgui.separator()

        try:
            entries = sorted(pathlib.Path(self.directory).iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            entries = []
        dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
        files = [p for p in entries if p.is_file() and p.suffix == self.extension]
        items = [(p.name + "/", p) for p in dirs] + [(p.name, p) for p in files]

        if imgui.begin_child("manual_calibration_file_list", size=(0, -34)):
            for idx, (label, path) in enumerate(items):
                clicked, _ = imgui.selectable(label, idx == self.selected)
                if clicked:
                    if path.is_dir():
                        self.directory = str(path)
                        self.selected = -1
                    else:
                        self.selected = idx
                if path.is_file() and imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
                    self._accept(path, state)
                    imgui.close_current_popup()
                    break
            imgui.end_child()

        chosen = (
            items[self.selected][1] if 0 <= self.selected < len(items) and items[self.selected][1].is_file() else None
        )
        if imgui.button("OK", size=(80, 0)) and chosen is not None:
            self._accept(chosen, state)
            imgui.close_current_popup()
        imgui.same_line()
        if imgui.button("Cancel", size=(80, 0)):
            self.open = False
            imgui.close_current_popup()

        imgui.end_popup()

    def _accept(self, path: pathlib.Path, state: CalibrationState) -> None:
        if self.target == "trajectory":
            state.trajectory_path = str(path)
            state.queue("load_trajectory", path)
        else:
            state.params_path = str(path)
            state.queue("load_params", path)
        self.open = False


class ManualCalibrationPanel:
    def __init__(self, state: CalibrationState) -> None:
        self.state = state
        self.browser = NPZFileBrowser(pathlib.Path.cwd(), ".npz")
        self.param_browser = NPZFileBrowser(pathlib.Path.cwd(), ".yaml")

    def __call__(self, imgui) -> None:
        state = self.state
        imgui.text("Manual Calibration")
        changed_mode, new_mode = imgui.combo(
            "Mode##manual_mode", state.mode_idx, ("Trajectory replay", "Manual target")
        )
        if changed_mode:
            state.mode_idx = int(new_mode)
        if imgui.button("Step##manual_step", size=(80, 0)):
            state.queue("single_step")
        imgui.same_line()
        if imgui.button("Reset Replay##manual_reset", size=(120, 0)):
            state.queue("reset_replay")
        imgui.same_line()
        if imgui.button("Reset Env##manual_reset_env", size=(110, 0)):
            state.queue("reset_env")

        imgui.separator()
        _, state.trajectory_path = imgui.input_text("Trajectory##manual_traj", state.trajectory_path, 512)
        imgui.same_line()
        if imgui.button("Browse##manual_traj_browse"):
            self.browser.request("trajectory", state.trajectory_path)
        if imgui.button("Load Trajectory##manual_traj_load"):
            state.queue("load_trajectory", pathlib.Path(state.trajectory_path).expanduser())

        _, state.params_path = imgui.input_text("Params YAML##manual_params", state.params_path, 512)
        imgui.same_line()
        if imgui.button("Browse##manual_params_browse"):
            self.param_browser.request("params", state.params_path)
        if imgui.button("Load Params YAML##manual_params_load"):
            state.queue("load_params", pathlib.Path(state.params_path).expanduser())
        imgui.same_line()
        if imgui.button("Save Params YAML##manual_params_save"):
            state.queue("save_params", pathlib.Path(state.params_path).expanduser())

        imgui.separator()
        changed_dof, new_dof = imgui.combo("Current DOF##manual_dof", state.selected_dof, state.dof_names)
        if changed_dof:
            state.selected_dof = int(new_dof)

        if state.tactile_enabled and state.tactile_recorder is not None:
            rec = state.tactile_recorder
            entries = ["All fingers"] + list(rec.display_names)
            changed_f, new_f = imgui.combo("Tactile finger##manual_tac_finger", state.selected_tactile_finger, entries)
            if changed_f:
                state.selected_tactile_finger = int(new_f)
                state.queue("isolate_tactile", int(new_f))
            if state.selected_tactile_finger > 0:
                sample = rec.plot_data().get(entries[state.selected_tactile_finger])
                if sample is not None:
                    imgui.text("sim  fx/fy/fz: %+.3f %+.3f %+.3f" % tuple(sample[:3]))
                    imgui.text("real fx/fy/fz: %+.3f %+.3f %+.3f" % tuple(sample[3:]))

        idx = state.selected_dof
        target = cast(np.ndarray, state.target_positions)
        lower = cast(np.ndarray, state.dof_lower)
        upper = cast(np.ndarray, state.dof_upper)
        changed_target, new_target = imgui.drag_float(
            "Target Position##manual_target_pos",
            float(target[idx]),
            0.01,
            float(lower[idx]),
            float(upper[idx]),
            "%.5f",
        )
        if changed_target:
            target[idx] = float(np.clip(new_target, lower[idx], upper[idx]))
        if imgui.button("Command Target##manual_command_target", size=(150, 0)):
            state.queue("command_target")
        imgui.same_line()
        if imgui.button("All 0##manual_zero_targets", size=(80, 0)):
            state.queue("zero_targets")
        imgui.same_line()
        if imgui.button("Random All##manual_random_targets", size=(110, 0)):
            state.queue("random_targets")

        imgui.separator()
        for name in PROPERTY_NAMES:
            current = float(state.values[name][idx])
            lo, hi = PROPERTIES[name].bounds
            changed, new_value = imgui.drag_float(
                f"{name}##manual_{name}",
                current,
                max((hi - lo) * 0.002, 1e-4),
                lo,
                max(hi, current * 1.5 + 1.0),
                "%.5f",
            )
            if changed:
                state.values[name][idx] = float(np.clip(new_value, lo, hi))

        if imgui.button("Apply Current Joint##manual_apply_current", size=(160, 0)):
            state.queue("apply_current", idx)
        imgui.same_line()
        if imgui.button("Apply All Joints##manual_apply_all", size=(140, 0)):
            state.queue("apply_all", idx)

        imgui.text_wrapped(f"Status: {state.status}")
        self.browser.draw(imgui, state)
        self.param_browser.draw(imgui, state)


def _load_trajectory(path: pathlib.Path, dof_names: Sequence[str], sim_step_dt: float) -> Trajectory:
    trajectory = Trajectory.load(path)
    if trajectory.action is None:
        raise ValueError(f"{path} is missing required Trajectory.action")
    if trajectory.dofs_pos is None:
        raise ValueError(f"{path} is missing required Trajectory.dofs_pos")
    if trajectory.action.shape[1] != len(dof_names):
        raise ValueError(
            f"{path} action width {trajectory.action.shape[1]} does not match {len(dof_names)} robot DOFs."
        )
    if trajectory.dofs_pos.shape[1] != len(dof_names):
        raise ValueError(
            f"{path} dofs_pos width {trajectory.dofs_pos.shape[1]} does not match {len(dof_names)} robot DOFs."
        )

    if trajectory.dof_names:
        name_to_idx = {name: i for i, name in enumerate(trajectory.dof_names)}
        missing = [name for name in dof_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"{path} is missing robot DOFs: {missing}")
        order = [name_to_idx[name] for name in dof_names]
        trajectory.action = trajectory.action[:, order]
        trajectory.dofs_pos = trajectory.dofs_pos[:, order]
        if trajectory.dofs_vel is not None and trajectory.dofs_vel.shape[1] == len(order):
            trajectory.dofs_vel = trajectory.dofs_vel[:, order]
        if trajectory.dofs_torque is not None and trajectory.dofs_torque.shape[1] == len(order):
            trajectory.dofs_torque = trajectory.dofs_torque[:, order]
        trajectory.dof_names = tuple(dof_names)

    if "qpos" not in trajectory.initial_state:
        trajectory.initial_state["qpos"] = trajectory.dofs_pos[0].copy()
    if "dofs_vel" not in trajectory.initial_state:
        trajectory.initial_state["dofs_vel"] = np.zeros(len(dof_names), dtype=np.float64)

    if sim_step_dt > 0.0 and trajectory.dt > 0.0:
        rel_err = abs(trajectory.dt - sim_step_dt) / sim_step_dt
        if rel_err > 0.02:
            en.logger.warning(
                f"trajectory dt={trajectory.dt * 1000:.2f} ms but sim replay dt={sim_step_dt * 1000:.2f} ms "
                f"({rel_err * 100:.1f}% mismatch)."
            )
    return trajectory


def _apply_property_values(
    env,
    state: CalibrationState,
    names: Sequence[str],
    selected_idx: int | None = None,
) -> None:
    """Push slider values into the sim via the same path identify uses.

    Builds a ``ParameterSet`` from ``state.values[name]`` and calls
    ``apply_parameters``; the patched ``_apply_one`` routes solver fields
    to ``entity.set_dofs_<prop>`` and modifier fields to their per-tensor
    writers. ``refresh_pd_gains`` is invoked once if any kp/kd was touched
    (``ExplicitPDController`` caches gains and otherwise only refreshes on
    ``env.reset``).
    """
    if selected_idx is None:
        target_dof_names = state.dof_names
        values_for = lambda name: state.values[name]
    else:
        target_dof_names = [state.dof_names[selected_idx]]
        values_for = lambda name: np.asarray([state.values[name][selected_idx]], dtype=np.float64)

    relevant = [name for name in names if name in PROPERTIES]
    if not relevant:
        return
    params = ParameterSet([make_parameter(name, target_dof_names, values_for(name)) for name in relevant])
    apply_parameters(env, params)
    if any(PROPERTIES[name].needs_pd_refresh for name in relevant):
        refresh_pd_gains(env)


def _read_current_values(env, state: CalibrationState, dof_indices: Sequence[int]) -> None:
    robot = env.entities["robot"]
    lower, upper = robot.get_dofs_limit(dofs_idx_local=dof_indices)
    if isinstance(lower, torch.Tensor) and lower.ndim == 2:
        lower = lower[0]
    if isinstance(upper, torch.Tensor) and upper.ndim == 2:
        upper = upper[0]
    lower_np = _to_1d_numpy(lower)
    upper_np = _to_1d_numpy(upper)
    state.dof_lower = np.where(np.isfinite(lower_np), lower_np, -np.pi)
    state.dof_upper = np.where(np.isfinite(upper_np), upper_np, np.pi)

    current_pos = robot.get_dofs_pos(dofs_idx_local=dof_indices)
    if isinstance(current_pos, torch.Tensor) and current_pos.ndim == 2:
        current_pos = current_pos[0]
    state.target_positions = np.clip(_to_1d_numpy(current_pos), state.dof_lower, state.dof_upper)

    for name in PROPERTY_NAMES:
        state.values[name] = PROPERTIES[name].read_row(env, state.dof_names)


def _deploy_send(deployer, dofs_pos: np.ndarray, *, ctrl_to_full: list[int] | None = None) -> None:
    if deployer is None:
        return
    n = deployer.num_dofs
    pos = np.asarray(dofs_pos, dtype=np.float64).reshape(-1)
    if len(pos) != n:
        # Partial/frozen controller: the GUI commands only a DOF subset, but the
        # real hand has all n motors. Scatter the subset into a full-width command;
        # uncontrolled motors default to 0.0 (the xhand rest pose).
        if ctrl_to_full is None or len(ctrl_to_full) != len(pos):
            en.logger.warning(f"_deploy_send: action width {len(pos)} != hand DOFs {n} and no DOF map; skipping.")
            return
        full = np.zeros(n, dtype=np.float64)
        full[ctrl_to_full] = pos
        pos = full
    # RoboTeraXHandDeployment.send_payload only forwards dofs_pos to the SDK;
    # dofs_kp/dofs_kd are dropped on the floor (gains live on the deployer
    # class attrs set at connect()). Don't bother filling them.
    try:
        deployer.send_payload(
            RobotCommand(
                dofs_pos=pos,
                dofs_vel=np.zeros(n, dtype=np.float64),
                dofs_torque=np.zeros(n, dtype=np.float64),
                dofs_kp=np.zeros(n, dtype=np.float64),
                dofs_kd=np.zeros(n, dtype=np.float64),
            )
        )
    except Exception as exc:  # noqa: BLE001 - keep GUI alive on transient comm errors.
        en.logger.warning(f"deployer.send_payload() failed: {type(exc).__name__}: {exc}")


def _reset_views(env, state: CalibrationState, deployer=None) -> None:
    """Refresh the live plots after an env reset.

    Pushes the post-reset sample to the joint and tactile plots FIRST, so the
    next point the recorder thread appends is the new pose rather than a stale
    pre-reset value, then fully resets both plotters. ``reset_plotter`` also
    rescales the axes: the live plotter only ever extends its x/y limits, so
    without that the window stays stuck at the old range and the fresh data is
    drawn off-screen.
    """
    deploy_state = _read_deploy_state(deployer)
    _update_plot_values(env, state, deploy_state=deploy_state)
    if state.tactile_enabled:
        _update_tactile(env, state, deploy_state)
    reset_plotter(state.plotter)
    reset_plotter(state.tactile_plotter)


def _reset_replay(env, state: CalibrationState, deployer=None) -> None:
    if state.trajectory is None:
        return
    # Match the shared rollout: env.reset() runs action_manager.reset, which
    # refreshes ExplicitPDController's cached kp/kd from the entity. Then
    # reset_to_initial_state overrides the default pose with the trajectory's.
    env.reset()
    reset_to_initial_state(env, state.trajectory, entity_name="robot")
    state.replay_idx = 0
    _reset_views(env, state, deployer)


def _read_deploy_state(deployer):
    """Read one RobotState from the real hand; None on no-deployer / comm error.

    Returning the whole RobotState (not just dofs_pos) lets one hardware read
    feed both the joint-position plot and the tactile recorder per step.
    """
    if deployer is None:
        return None
    try:
        return deployer.read_state()
    except Exception as exc:  # noqa: BLE001 - keep GUI alive on transient comm errors.
        en.logger.warning(f"deployer.read_state() failed: {type(exc).__name__}: {exc}")
        return None


def _update_plot_values(env, state: CalibrationState, deploy_state=None) -> None:
    idx = state.selected_dof
    trajectory = state.trajectory
    sim = env.entities["robot"].get_dofs_pos()
    if isinstance(sim, torch.Tensor) and sim.ndim == 2:
        sim = sim[0]
    sim_values = _to_1d_numpy(sim)
    target_positions = cast(np.ndarray, state.target_positions)
    deploy_measured = np.asarray(deploy_state.dofs_pos, dtype=np.float64) if deploy_state is not None else None
    if deploy_measured is not None and state.ctrl_to_full is not None:
        # read_state() reports all hand motors; narrow to the controlled subset
        # so it aligns with state.dof_names / selected_dof.
        deploy_measured = deploy_measured[state.ctrl_to_full]

    if not state.replay_mode:
        if deploy_measured is not None:
            measured = float(deploy_measured[idx])
        elif trajectory is not None and trajectory.dofs_pos is not None:
            measured = float(trajectory.dofs_pos[min(state.replay_idx, len(trajectory) - 1), idx])
        else:
            measured = float(target_positions[idx])
        state.plot_values = (float(target_positions[idx]), measured, float(sim_values[idx]))
        return

    if trajectory is None or trajectory.action is None or trajectory.dofs_pos is None:
        commanded = float(target_positions[idx])
        measured = float(deploy_measured[idx]) if deploy_measured is not None else commanded
        state.plot_values = (commanded, measured, float(sim_values[idx]))
        return

    t = min(state.replay_idx, len(trajectory) - 1)
    commanded = float(trajectory.action[t, idx])
    measured = float(deploy_measured[idx]) if deploy_measured is not None else float(trajectory.dofs_pos[t, idx])
    state.plot_values = (commanded, measured, float(sim_values[idx]))


def _plot_data_func(state: CalibrationState) -> tuple[float, float, float]:
    return state.plot_values


def _set_plot_colors(plotter) -> None:
    if plotter is None or not hasattr(plotter, "lines"):
        return
    colors_by_label = {
        "commanded": "g",
        "measured": "k",
        "sim": "r",
    }
    lines = plotter.lines.get("main", [])
    for line in lines:
        color = colors_by_label.get(line.get_label())
        if color is not None:
            line.set_color(color)

    if hasattr(plotter, "fig"):
        for legend in plotter.fig.legends:
            handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
            for handle, text in zip(handles, legend.get_texts(), strict=False):
                color = colors_by_label.get(text.get_text())
                if color is not None:
                    handle.set_color(color)
        plotter.fig.canvas.draw()


def _process_commands(env, state: CalibrationState, sim_step_dt: float, deployer=None) -> bool:
    step_once = False
    for command in state.pop_commands():
        try:
            if command.name == "apply_current":
                idx = int(command.payload)
                _apply_property_values(env, state, PROPERTY_NAMES, selected_idx=idx)
                state.status = f"Applied values for {state.dof_names[idx]}."
            elif command.name == "apply_all":
                idx = int(command.payload)
                for name in PROPERTY_NAMES:
                    state.values[name][:] = state.values[name][idx]
                _apply_property_values(env, state, PROPERTY_NAMES)
                state.status = f"Applied {state.dof_names[idx]} values to all joints."
            elif command.name == "command_target":
                _target_step(env, state, deployer=deployer)
                state.status = "Commanded manual target positions."
            elif command.name == "load_trajectory":
                path = pathlib.Path(command.payload).expanduser()
                state.trajectory = _load_trajectory(path, state.dof_names, sim_step_dt)
                state.trajectory_path = str(path)
                _reset_replay(env, state, deployer=deployer)
                state.status = f"Loaded {len(state.trajectory)} samples from {path}."
            elif command.name == "load_params":
                path = pathlib.Path(command.payload).expanduser()
                loaded = load_params_yaml(path, state.dof_names)
                for name, vals in loaded.items():
                    state.values[name] = vals
                _apply_property_values(env, state, PROPERTY_NAMES)
                state.params_path = str(path)
                state.status = f"Loaded and applied parameters from {path}."
            elif command.name == "random_targets":
                lower = cast(np.ndarray, state.dof_lower)
                upper = cast(np.ndarray, state.dof_upper)
                state.target_positions = np.random.default_rng().uniform(lower, upper)
                _target_step(env, state, deployer=deployer)
                state.status = "Randomized and commanded all target positions."
            elif command.name == "save_params":
                path = pathlib.Path(command.payload).expanduser()
                save_params_yaml(path, state.dof_names, {name: state.values[name] for name in PROPERTY_NAMES})
                state.params_path = str(path)
                state.status = f"Saved parameters to {path}."
            elif command.name == "reset_replay":
                _reset_replay(env, state, deployer=deployer)
                state.status = "Replay reset to trajectory start."
            elif command.name == "reset_env":
                env.reset()
                state.replay_idx = 0
                _reset_views(env, state, deployer=deployer)
                state.status = "Environment reset."
            elif command.name == "single_step":
                step_once = True
            elif command.name == "zero_targets":
                lower = cast(np.ndarray, state.dof_lower)
                upper = cast(np.ndarray, state.dof_upper)
                state.target_positions = np.clip(np.zeros(len(state.dof_names), dtype=np.float64), lower, upper)
                _target_step(env, state, deployer=deployer)
                state.status = "Zeroed and commanded all target positions."
            elif command.name == "isolate_tactile":
                rec = state.tactile_recorder
                if rec is not None:
                    combo_idx = int(command.payload)
                    rec.isolated_finger = None if combo_idx <= 0 else rec.display_indices[combo_idx - 1]
                    apply_isolation(state.tactile_plotter, rec)
                    reset_plotter(state.tactile_plotter)
                    state.status = (
                        "Tactile view: all fingers"
                        if combo_idx <= 0
                        else f"Tactile view: {rec.display_names[combo_idx - 1]}"
                    )
        except Exception as exc:  # noqa: BLE001 - keep GUI alive and show the actionable error.
            state.status = f"{type(exc).__name__}: {exc}"
            en.logger.warning(state.status)
    return step_once


def _update_tactile(env, state: CalibrationState, deploy_state) -> None:
    """Record one real-vs-sim tactile sample; disable recording on error."""
    recorder = state.tactile_recorder
    if recorder is None:
        return
    try:
        sim = read_sim_tactile(env, state.sim_sensor_names, device=env.device) if state.sim_sensor_names else None
        real = read_real_tactile(deploy_state) if deploy_state is not None else None
        recorder.record(t=env.scene.t * env.scene.dt, sim=sim, real=real)
    except Exception as exc:  # noqa: BLE001 - keep the GUI alive; disable on error.
        en.logger.warning(f"tactile recording disabled after error: {type(exc).__name__}: {exc}")
        state.tactile_enabled = False


def _first_env_flag(flag: Any) -> bool:
    """Truthiness of env 0's entry in a per-env done/terminated flag."""
    if flag is None:
        return False
    if isinstance(flag, torch.Tensor):
        return bool(flag.reshape(-1)[0].item()) if flag.numel() else False
    if isinstance(flag, np.ndarray):
        return bool(flag.reshape(-1)[0]) if flag.size else False
    if isinstance(flag, (list, tuple)):
        return bool(flag[0]) if flag else False
    return bool(flag)


def _env_step_terminated(step_out: Any) -> bool:
    """True when the last RLEnvBase.step terminated/truncated env 0.

    RLEnvBase.step returns ``(obs, reward, terminated, truncated, extras)`` and
    auto-resets done envs internally, so a True here means the env already reset.
    """
    if not isinstance(step_out, (tuple, list)) or len(step_out) < 5:
        return False
    return _first_env_flag(step_out[2]) or _first_env_flag(step_out[3])


def _step(env, action_np: np.ndarray, state: CalibrationState, deployer=None) -> None:
    """Step the sim with ``action_np``, mirror it to the real hand, refresh plots."""
    action = torch.as_tensor(action_np, dtype=torch.float32, device=env.device)
    step_out = env.step(action.unsqueeze(0))
    _deploy_send(deployer, action_np, ctrl_to_full=state.ctrl_to_full)
    # One hardware read per step feeds both the joint-position and tactile plots.
    deploy_state = _read_deploy_state(deployer)
    _update_plot_values(env, state, deploy_state=deploy_state)
    if state.tactile_enabled:
        _update_tactile(env, state, deploy_state)
    # A task termination auto-resets the env inside env.step(); restart the plot
    # windows so they do not stay stuck at the pre-reset range.
    if _env_step_terminated(step_out):
        reset_plotter(state.plotter)
        reset_plotter(state.tactile_plotter)
        state.replay_idx = 0
        state.status = "Env auto-reset (task termination)."


def _replay_step(env, state: CalibrationState, deployer=None) -> None:
    trajectory = state.trajectory
    if trajectory is None or trajectory.action is None:
        _target_step(env, state, deployer=deployer)
        return
    if state.replay_idx >= len(trajectory):
        _reset_replay(env, state, deployer=deployer)
    _step(env, np.asarray(trajectory.action[state.replay_idx], dtype=np.float64), state, deployer=deployer)
    state.replay_idx += 1


def _target_step(env, state: CalibrationState, deployer=None) -> None:
    _step(env, cast(np.ndarray, state.target_positions), state, deployer=deployer)


def _resize_state_dofs(state: CalibrationState, dof_names: Sequence[str]) -> None:
    """Re-fit per-DOF GUI state to a new DOF list (the controller's controlled subset).

    Per-DOF arrays are zeroed; ``_read_current_values`` repopulates them from the
    entity/controller right after this is called.
    """
    state.dof_names = list(dof_names)
    n = len(state.dof_names)
    state.selected_dof = min(state.selected_dof, max(n - 1, 0))
    state.target_positions = np.zeros(n, dtype=np.float64)
    state.dof_lower = np.full(n, -np.pi, dtype=np.float64)
    state.dof_upper = np.full(n, np.pi, dtype=np.float64)
    state.values = {name: np.zeros(n, dtype=np.float64) for name in PROPERTY_NAMES}


def _build_deployer(env, args):
    """Instantiate and connect the XHand1 deployment, leaving the sim entity intact.

    Why: DeploymentBase.__init__ swaps env.entities[entity_name] for a RobotStateEntity
    wrapper so observation terms read real-robot state. The manual GUI keeps stepping
    the sim independently, so we save and restore the original sim entity reference.
    """
    from eden.extensions.deployment.base import DEPLOYMENT_REGISTRY
    from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
    from eden.options.extensions.deployment import DeploymentOptions

    if "robo_tera_x_hand_deployment" not in DEPLOYMENT_REGISTRY:
        raise SystemExit("RoboTeraXHandDeployment is not registered.")

    sim_entity = env.entities["robot"]
    deployer = RoboTeraXHandDeployment(
        env=env,
        options=DeploymentOptions(
            entity_name="robot",
            control_freq=args.deploy_control_freq,
            finger_mode=args.deploy_finger_mode,
        ),
    )
    deployer.protocol = args.deploy_protocol
    deployer.serial_port = args.deploy_serial_port
    deployer.hand_id = args.deploy_hand_id
    env.entities["robot"] = sim_entity

    en.logger.info(f"Connecting to XHand1 via {args.deploy_protocol} ({args.deploy_serial_port}) …")
    deployer.connect()
    en.logger.info("Running deployment init sequence — robot will hold its default pose.")
    deployer.init_sequence()
    return deployer


def build_env(args, *, show_viewer: bool):
    sysid_cfg = make_sim_twin_config(args.robot)
    if args.use_task:
        config = get_task_config_from_args(args, upload_logs=False)
        config.env_options.num_envs = 1
        config.recorder_options = sysid_cfg.recorder_options
    else:
        config = sysid_cfg
    config.env_options.background_color = (1.0, 1.0, 1.0)

    env = RLEnvBase.from_config(config, show_viewer=show_viewer)

    return env, config


def main() -> int:
    parser = get_argparser(description=__doc__)
    parser.add_argument("--trajectory", type=pathlib.Path, default=None, help="Trajectory .npz to load at startup.")
    parser.add_argument("--params-in", type=pathlib.Path, default=None, help="Optional ParameterSet YAML to load.")
    parser.add_argument(
        "--params-out",
        type=pathlib.Path,
        default=pathlib.Path("data/xhand_sysid/manual_params.yaml"),
        help="Default output YAML path for Save Params YAML.",
    )
    parser.add_argument("--history-length", type=int, default=1000, help="Live plot history length.")
    parser.add_argument(
        "--record-tactile",
        action="store_true",
        help="Record + live-plot real-vs-sim fingertip tactile; saves a CSV to the log dir. "
        "Pair with --use-task (for a sim trace) and/or --deploy (for the real-hand trace).",
    )
    parser.add_argument("--tactile-history", type=int, default=500, help="Live tactile plot history length.")
    parser.add_argument(
        "--tactile-fingers",
        nargs="+",
        choices=["thumb", "index", "mid", "ring", "pinky"],
        default=None,
        help="Which fingertips to plot (default: all with data). CSV still records all.",
    )
    parser.add_argument("--use-task", action="store_true", help="Use task config instead of sim twin config.")
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Also control the real XHand1 alongside the sim. In replay/manual mode, the GUI commands the hardware "
        "every step and the 'measured' plot trace reads from the live hand instead of the recorded trajectory.",
    )
    parser.add_argument("--deploy-protocol", choices=["RS485", "EtherCAT"], default="RS485")
    parser.add_argument("--deploy-serial-port", default="/dev/ttyUSB0")
    parser.add_argument("--deploy-hand-id", type=int, default=0)
    parser.add_argument("--deploy-control-freq", type=float, default=None, help="Optional control-loop frequency (Hz).")
    parser.add_argument(
        "--deploy-finger-mode",
        choices=[0, 3, 5],
        default=3,
        type=int,
        help="Hand control mode: 0 powerless, 3 position (default), 5 powerful.",
    )
    args = parser.parse_args()

    if args.deploy and args.robot != "xhand1":
        raise SystemExit(
            f"--deploy is only implemented for the xhand1 hand (RoboTeraXHandDeployment); got --robot {args.robot}. "
            f"Drop --deploy to run the {args.robot} sim twin alone."
        )

    en.init(backend=gs.cpu if args.cpu else gs.gpu, performance_mode=False, log_root_path="logs/temp")
    env, config = build_env(args, show_viewer=True)
    dof_names = list(config.scene_options.robot.dofs_name)
    state = CalibrationState(dof_names=dof_names, params_out=args.params_out)
    if args.trajectory is not None:
        state.trajectory_path = str(args.trajectory)
    if args.params_in is not None:
        state.params_path = str(args.params_in)

    plotter = None
    try:
        from genesis.recorders.plotters import IS_MATPLOTLIB_AVAILABLE

        if IS_MATPLOTLIB_AVAILABLE:
            plotter = env.scene.start_recording(
                lambda: _plot_data_func(state),
                gs.recorders.MPLLinePlot(
                    title=f"{args.robot} manual calibration",
                    labels=PLOT_LABELS,
                    x_label="step",
                    y_label="joint position (rad)",
                    history_length=args.history_length,
                ),
            )
        else:
            state.status = "Matplotlib is unavailable; live plot disabled."
    except Exception as exc:  # noqa: BLE001 - plotting is helpful, not required for calibration.
        state.status = f"Live plot disabled: {type(exc).__name__}: {exc}"
    state.plotter = plotter

    # Tactile recording + plot. start_recording() must run before env.build()
    # (it is @assert_unbuilt), so set this up in the same pre-build block as the
    # joint-position plot. env.sensors is already populated by from_config.
    if args.record_tactile:
        state.sim_sensor_names = resolve_sim_sensor_names(
            env, fingertip_links=fingertip_link_order(config.scene_options.robot)
        )
        if not state.sim_sensor_names and not args.deploy:
            tactile_keys = sorted(k for k in getattr(env, "sensors", {}) if k.startswith("tactile_"))
            en.logger.warning(
                "--record-tactile: no agg_force fingertip sim sensors found and --deploy is off; "
                f"nothing to record. tactile_* sensors in env: {tactile_keys or 'none'}. "
                "Use --use-task with a agg_force --sensors config and/or --deploy."
            )
        else:
            if not state.sim_sensor_names:
                tactile_keys = sorted(k for k in getattr(env, "sensors", {}) if k.startswith("tactile_"))
                en.logger.warning(
                    "--record-tactile: no agg_force fingertip sim sensors found; recording the real hand "
                    f"only (sim columns -> nan). tactile_* sensors in env: {tactile_keys or 'none'}."
                )
            if args.tactile_fingers:
                display = set(finger_indices(args.tactile_fingers))
            else:
                # Plot fingers with data: sim-sensored links, plus all five when
                # the real hand is connected. Frozen/unsensored links are dropped.
                display = set(state.sim_sensor_names)
                if args.deploy:
                    display |= set(range(len(FINGER_NAMES)))
            state.tactile_recorder = TactileComparisonRecorder(en.log_dir, display_fingers=sorted(display))
            state.tactile_plotter = start_tactile_plot(
                env.scene,
                state.tactile_recorder,
                title=f"{args.robot} tactile: real vs sim",
                history_length=args.tactile_history,
            )
            state.tactile_enabled = True

    env.build()
    install_action_mod_sysid_patch()
    robot = env.entities["robot"]

    # A partial/frozen controller drives only a subset of the robot's DOFs. The
    # calibration GUI tunes exactly those: actions, modifier params and reads are
    # all controller-DOF indexed, so re-fit per-DOF state to the controlled subset
    # (otherwise modifier reads/writes KeyError on uncontrolled DOFs).
    try:
        controlled_dof_names = list(get_dofs_pos_controller(env).dofs_name)
    except (KeyError, TypeError) as exc:
        controlled_dof_names = list(state.dof_names)
        en.logger.warning(f"Could not resolve dofs_pos_controller ({exc}); calibrating all robot DOFs.")
    if controlled_dof_names != state.dof_names:
        en.logger.info(
            f"Controller drives {len(controlled_dof_names)}/{len(state.dof_names)} DOFs; "
            f"calibrating only the controlled subset: {controlled_dof_names}"
        )
        state.ctrl_to_full = [dof_names.index(n) for n in controlled_dof_names]
        _resize_state_dofs(state, controlled_dof_names)

    _, dof_indices = robot.find_named_dofs_idx_local(state.dof_names, name_scope=robot.dofs_name, preserve_order=True)
    _read_current_values(env, state, dof_indices)
    _apply_property_values(env, state, PROPERTY_NAMES)

    deployer = _build_deployer(env, args) if args.deploy else None
    if deployer is not None:
        state.status = f"Deployment connected on {args.deploy_protocol} ({args.deploy_serial_port})."

    _update_plot_values(env, state, deploy_state=_read_deploy_state(deployer))
    _set_plot_colors(plotter)

    sim_step_dt = config.env_options.sim_dt * config.env_options.decimation
    if args.params_in is not None:
        state.queue("load_params", args.params_in)
    if args.trajectory is not None:
        state.queue("load_trajectory", args.trajectory)
    _process_commands(env, state, sim_step_dt, deployer=deployer)

    from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

    plugin = ImGuiOverlayPlugin(
        show_sim_controls=False,
        show_entity_browser=False,
        show_visualization=True,
        show_camera_controls=True,
        panel_width=520,
    )
    env.scene.viewer.add_plugin(plugin)
    plugin.register_panel(ManualCalibrationPanel(state))

    try:
        while env.scene.viewer.is_alive():
            step_once = _process_commands(env, state, sim_step_dt, deployer=deployer)
            if state.replay_mode:
                _replay_step(env, state, deployer=deployer)
            else:
                _target_step(env, state, deployer=deployer)
            if step_once:
                deploy_state = _read_deploy_state(deployer)
                _update_plot_values(env, state, deploy_state=deploy_state)
                if state.tactile_enabled:
                    _update_tactile(env, state, deploy_state)
            time.sleep(0.005)
    finally:
        if state.tactile_recorder is not None:
            state.tactile_recorder.close()
            en.logger.info(f"Tactile CSV saved to {state.tactile_recorder.csv_path}")
        if deployer is not None:
            try:
                deployer.close()
            except Exception as exc:  # noqa: BLE001 - close errors shouldn't mask real failures.
                en.logger.warning(f"deployer.close() failed: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
