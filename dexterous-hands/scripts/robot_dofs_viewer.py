"""
Interactive visualizer for controlling robot joints, base pose, and (optionally) object pose.
Exports a YAML snippet suitable for use as a grasp-generator config override.

Usage:
    python scripts/robot_dofs_viewer.py --task screwdriver --robot xhand1 --cpu
    python scripts/robot_dofs_viewer.py --task in_hand_repose --cpu
    python scripts/robot_dofs_viewer.py --task="$TASK" --robot="$ROBOT" --grasp-path "$GRASPS_OUTPUT" --grasp-index 0 --cpu
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING

import eden as en
import genesis as gs
import numpy as np
import torch
import yaml
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
from eden.extensions.visualization.viser import ViserViewer
from genesis.utils.geom import quat_to_xyz, xyz_to_quat

from registry import get_argparser, get_task_config_from_args
from tasks.screwdriver.custom_terms import get_grasp_target_pos, resolve_grasp_target_link
from utils import get_entity_metadata

if TYPE_CHECKING:
    from eden.entities.base import Entity


def _format_vec(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.6f}" for v in values) + "]"


def _to_dofs_idx_list(dofs_idx_local) -> list[int]:
    """Normalize dofs_idx_local to a plain list; Eden may return None, a tensor, or a list."""
    if dofs_idx_local is None:
        return []
    if hasattr(dofs_idx_local, "tolist"):
        return list(dofs_idx_local.tolist())
    return list(dofs_idx_local)


def _load_grasp_target_metadata_from_yaml(config_path: str | None) -> dict | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.is_file():
        return None

    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    scene = data.get("scene")
    if not isinstance(scene, dict):
        return None
    obj = scene.get("obj")
    if not isinstance(obj, dict):
        return None
    metadata = obj.get("metadata")
    if not isinstance(metadata, dict):
        return None
    return metadata


def _apply_precomputed_grasp(
    robot: "Entity",
    obj: "Entity | None",
    *,
    grasp_path: str,
    grasp_index: int,
) -> str:
    """Load one saved grasp sample onto the current robot/object state."""
    grasps_path = grasp_path
    if not Path(grasps_path).is_file():
        raise FileNotFoundError(f"Precomputed grasp file not found: {grasps_path}")

    grasp_data = torch.load(grasps_path, map_location="cpu")
    joint_pos = grasp_data["joint_pos"]
    num_grasps = joint_pos.shape[0]
    if grasp_index < 0 or grasp_index >= num_grasps:
        raise IndexError(f"grasp_index={grasp_index} out of range for {num_grasps} saved grasps in {grasps_path}")

    robot_pos = grasp_data.get("robot_pos")
    robot_quat = grasp_data.get("robot_quat")
    obj_pos = grasp_data.get("obj_pos")
    obj_quat = grasp_data.get("obj_quat")
    obj_pose_frame = grasp_data.get("obj_pose_frame", "root")

    device = robot.get_pos().device

    if robot_pos is not None and robot_quat is not None:
        robot.set_pos(robot_pos[grasp_index].to(device))
        robot.set_quat(robot_quat[grasp_index].to(device))
    robot.set_dofs_pos(joint_pos[grasp_index].to(device), robot.dofs_idx_local, zero_velocity=True)

    if obj is not None and obj_pos is not None and obj_quat is not None:
        if obj_pose_frame != "root":
            raise ValueError(
                f"robot_dofs_viewer only supports precomputed grasps with obj_pose_frame='root'. Got: {obj_pose_frame}"
            )
        obj.set_pos(obj_pos[grasp_index].to(device))
        obj.set_quat(obj_quat[grasp_index].to(device))

    return grasps_path


class RobotDofsViewer(ViserViewer):
    def __init__(
        self,
        env,
        robot: "Entity",
        obj: "Entity | None" = None,
        grasp_target_metadata: dict | None = None,
        host: str = "localhost",
        port: int = 8080,
        enable_gui: bool = True,
    ):
        super().__init__(env, host=host, port=port, enable_gui=enable_gui)
        self.robot = robot
        self.obj = obj

        # Robot DOF config
        self.robot_dofs_idx = _to_dofs_idx_list(robot.dofs_idx_local)
        self.motors_name = robot.dofs_name
        self.robot_dofs_min_limit = []
        self.robot_dofs_max_limit = []
        self.current_dof_positions = None
        if self.robot_dofs_idx:
            dofs_limit = robot.get_dofs_limit(self.robot_dofs_idx)
            self.robot_dofs_min_limit = dofs_limit[0].tolist()[0]
            self.robot_dofs_max_limit = dofs_limit[1].tolist()[0]
            self.current_dof_positions = robot.get_dofs_pos(self.robot_dofs_idx).clone()

        # Robot base pose
        robot_quat = robot.get_quat()[0].cpu()
        self.init_pos = robot.get_pos()[0].cpu().numpy()
        self.init_euler = (quat_to_xyz(robot_quat).cpu().numpy() * 180.0) / np.pi
        self.current_base_pos = self.init_pos.copy()
        self.current_base_euler = self.init_euler.copy()

        # Object base pose (optional)
        self.obj_dofs_idx = []
        self.obj_dofs_name = []
        self.obj_dofs_min_limit = []
        self.obj_dofs_max_limit = []
        self.init_obj_base_pos = None
        self.init_obj_base_euler = None
        self.current_obj_base_pos = None
        self.current_obj_base_euler = None
        self.current_obj_dof_pos = None
        self.init_obj_dof_pos = None
        self.grasp_target_link_name = None
        self.grasp_target_use_aabb_center = False
        self.init_grasp_target_offset = np.zeros(3, dtype=np.float32)
        self.current_grasp_target_offset = np.zeros(3, dtype=np.float32)
        self._grasp_target_offset_sliders: list = []
        self._grasp_target_link_marker = None
        self._grasp_target_point_marker = None
        if self.obj is not None:
            obj_quat = obj.get_quat()[0].cpu()
            self.init_obj_base_pos = obj.get_pos()[0].cpu().numpy()
            self.init_obj_base_euler = (quat_to_xyz(obj_quat).cpu().numpy() * 180.0) / np.pi
            self.current_obj_base_pos = self.init_obj_base_pos.copy()
            self.current_obj_base_euler = self.init_obj_base_euler.copy()

            obj_metadata = get_entity_metadata(obj)
            if obj_metadata is not None:
                self.grasp_target_link_name = getattr(obj_metadata, "grasp_target_link", None)
                self.grasp_target_use_aabb_center = bool(getattr(obj_metadata, "grasp_target_use_aabb_center", False))
                if hasattr(obj_metadata, "grasp_target_local_offset"):
                    init_offset = np.asarray(getattr(obj_metadata, "grasp_target_local_offset"), dtype=np.float32)
                    if init_offset.shape == (3,):
                        self.init_grasp_target_offset = init_offset.copy()
                        self.current_grasp_target_offset = init_offset.copy()

            if grasp_target_metadata:
                yaml_link_name = grasp_target_metadata.get("grasp_target_link")
                if isinstance(yaml_link_name, str) and yaml_link_name.strip():
                    self.grasp_target_link_name = yaml_link_name.strip()
                if "grasp_target_use_aabb_center" in grasp_target_metadata:
                    self.grasp_target_use_aabb_center = bool(grasp_target_metadata["grasp_target_use_aabb_center"])
                yaml_offset = grasp_target_metadata.get("grasp_target_local_offset")
                if isinstance(yaml_offset, (list, tuple)) and len(yaml_offset) == 3:
                    init_offset = np.asarray(yaml_offset, dtype=np.float32)
                    self.init_grasp_target_offset = init_offset.copy()
                    self.current_grasp_target_offset = init_offset.copy()

            if self.grasp_target_link_name is None:
                obj_link_names = {link.name for link in obj.links}
                if "handle" in obj_link_names:
                    self.grasp_target_link_name = "handle"

            self.obj_dofs_idx = _to_dofs_idx_list(obj.dofs_idx_local)
            self.obj_dofs_name = obj.dofs_name or []
            if self.obj_dofs_idx:
                obj_dofs_limit = obj.get_dofs_limit(self.obj_dofs_idx)
                self.obj_dofs_min_limit = obj_dofs_limit[0].tolist()[0]
                self.obj_dofs_max_limit = obj_dofs_limit[1].tolist()[0]
                self.init_obj_dof_pos = obj.get_dofs_pos(self.obj_dofs_idx).clone()
                self.current_obj_dof_pos = self.init_obj_dof_pos.clone()

        self._dof_sliders: list = []
        self._pos_sliders: list = []
        self._rot_sliders: list = []
        self._obj_pos_sliders: list = []
        self._obj_rot_sliders: list = []
        self._obj_dof_sliders: list = []

    def _add_grasp_target_controls(self):
        if self.obj is None or self.grasp_target_link_name is None:
            return

        with self.server.gui.add_folder("Grasp Target"):
            self.server.gui.add_button(f"Target Link: {self.grasp_target_link_name}", disabled=True)
            axis_names = ["Offset X", "Offset Y", "Offset Z"]
            for i, axis_name in enumerate(axis_names):
                slider = self.server.gui.add_slider(
                    axis_name,
                    min=-0.150,
                    max=0.150,
                    step=0.001,
                    initial_value=float(self.init_grasp_target_offset[i]),
                )

                def make_offset_callback(idx, slider_obj):
                    def callback(_) -> None:
                        self.current_grasp_target_offset[idx] = slider_obj.value

                    return callback

                slider.on_update(make_offset_callback(i, slider))
                self._grasp_target_offset_sliders.append(slider)

            btn_reset_target = self.server.gui.add_button("Reset Grasp Target Offset")

            @btn_reset_target.on_click
            def _(_) -> None:
                for i, slider in enumerate(self._grasp_target_offset_sliders):
                    slider.value = float(self.init_grasp_target_offset[i])
                    self.current_grasp_target_offset[i] = self.init_grasp_target_offset[i]

    def _add_pose_controls(
        self,
        folder_name: str,
        init_pos: np.ndarray,
        init_euler: np.ndarray,
        current_pos: np.ndarray,
        current_euler: np.ndarray,
        pos_sliders: list,
        rot_sliders: list,
    ):
        with self.server.gui.add_folder(folder_name):
            with self.server.gui.add_folder("Position (XYZ)"):
                pos_names = ["X", "Y", "Z"]
                pos_limits = [(-1.0, 1.0), (-1.0, 1.0), (-0.1, 2.0)]

                for i, (name, (min_val, max_val)) in enumerate(zip(pos_names, pos_limits)):
                    slider = self.server.gui.add_slider(
                        name,
                        min=min_val,
                        max=max_val,
                        step=0.001,
                        initial_value=float(init_pos[i]),
                    )

                    # Factory captures idx and slider_obj by value, avoiding Python loop-closure capture bug.
                    def make_pos_callback(idx, slider_obj):
                        def callback(_) -> None:
                            current_pos[idx] = slider_obj.value

                        return callback

                    slider.on_update(make_pos_callback(i, slider))
                    pos_sliders.append(slider)

                btn_reset_pos = self.server.gui.add_button("Reset Position")

                @btn_reset_pos.on_click
                def _(_) -> None:
                    for i, slider in enumerate(pos_sliders):
                        slider.value = float(init_pos[i])
                        current_pos[i] = init_pos[i]

            with self.server.gui.add_folder("Rotation (deg)"):
                rot_names = ["Roll", "Pitch", "Yaw"]

                for i, name in enumerate(rot_names):
                    slider = self.server.gui.add_slider(
                        name,
                        min=-180.0,
                        max=180.0,
                        step=0.1,
                        initial_value=float(init_euler[i]),
                    )

                    # Same factory pattern as make_pos_callback above.
                    def make_rot_callback(idx, slider_obj):
                        def callback(_) -> None:
                            current_euler[idx] = slider_obj.value

                        return callback

                    slider.on_update(make_rot_callback(i, slider))
                    rot_sliders.append(slider)

                btn_reset_rot = self.server.gui.add_button("Reset Rotation")

                @btn_reset_rot.on_click
                def _(_) -> None:
                    for i, slider in enumerate(rot_sliders):
                        slider.value = float(init_euler[i])
                        current_euler[i] = init_euler[i]

    def _setup_gui(self):
        super()._setup_gui()

        # Robot base pose
        self._add_pose_controls(
            folder_name="Robot Base",
            init_pos=self.init_pos,
            init_euler=self.init_euler,
            current_pos=self.current_base_pos,
            current_euler=self.current_base_euler,
            pos_sliders=self._pos_sliders,
            rot_sliders=self._rot_sliders,
        )

        # Robot DOF sliders
        if self.robot_dofs_idx and self.current_dof_positions is not None:
            with self.server.gui.add_folder("Joint Positions"):
                btn_reset_all = self.server.gui.add_button("Reset All Joints")

                @btn_reset_all.on_click
                def _(_) -> None:
                    for i, slider in enumerate(self._dof_sliders):
                        val = float(self.current_dof_positions[0, i].item())
                        slider.value = val
                        self.current_dof_positions[0, i] = val

                for i, name in enumerate(self.motors_name):
                    min_limit = self.robot_dofs_min_limit[i]
                    max_limit = self.robot_dofs_max_limit[i]
                    init_val = float(self.current_dof_positions[0, i].item())
                    slider = self.server.gui.add_slider(
                        name,
                        min=min_limit,
                        max=max_limit,
                        step=0.01,
                        initial_value=init_val,
                    )

                    def make_dof_callback(idx, slider_obj):
                        def callback(_) -> None:
                            self.current_dof_positions[0, idx] = slider_obj.value

                        return callback

                    slider.on_update(make_dof_callback(i, slider))
                    self._dof_sliders.append(slider)

        # Object controls (optional)
        if self.obj is not None:
            self._add_pose_controls(
                folder_name="Object Base",
                init_pos=self.init_obj_base_pos,
                init_euler=self.init_obj_base_euler,
                current_pos=self.current_obj_base_pos,
                current_euler=self.current_obj_base_euler,
                pos_sliders=self._obj_pos_sliders,
                rot_sliders=self._obj_rot_sliders,
            )

            if self.obj_dofs_idx and self.current_obj_dof_pos is not None:
                with self.server.gui.add_folder("Object Joints"):
                    btn_reset_obj_joints = self.server.gui.add_button("Reset Object Joints")

                    @btn_reset_obj_joints.on_click
                    def _(_) -> None:
                        self.current_obj_dof_pos[:] = self.init_obj_dof_pos
                        for i, slider in enumerate(self._obj_dof_sliders):
                            slider.value = float(self.init_obj_dof_pos[0, i].item())

                    for i, dof_name in enumerate(self.obj_dofs_name):
                        min_limit = self.obj_dofs_min_limit[i]
                        max_limit = self.obj_dofs_max_limit[i]
                        init_val = float(self.current_obj_dof_pos[0, i].item())
                        slider = self.server.gui.add_slider(
                            dof_name,
                            min=min_limit,
                            max=max_limit,
                            step=0.01,
                            initial_value=init_val,
                        )

                        def make_obj_dof_callback(idx, slider_obj):
                            def callback(_) -> None:
                                self.current_obj_dof_pos[0, idx] = slider_obj.value

                            return callback

                        slider.on_update(make_obj_dof_callback(i, slider))
                        self._obj_dof_sliders.append(slider)

            self._add_grasp_target_controls()

        with self.server.gui.add_folder("Export"):
            btn_print = self.server.gui.add_button("Print YAML Snippet")

            @btn_print.on_click
            def _(_) -> None:
                print(self.get_yaml_snippet())

    def update_robot_state(self):
        device = self.env.device

        # zero_velocity=True prevents residual velocities from accumulating while the
        # scene is stepped without any physics (viewer-only mode).
        if self.robot_dofs_idx and self.current_dof_positions is not None:
            self.robot.set_dofs_pos(
                position=self.current_dof_positions,
                dofs_idx_local=self.robot_dofs_idx,
                zero_velocity=True,
            )

        self.robot.set_pos(torch.tensor(self.current_base_pos, dtype=torch.float32, device=device))
        robot_quat = xyz_to_quat(torch.tensor(self.current_base_euler, dtype=torch.float32) * np.pi / 180.0)
        self.robot.set_quat(robot_quat.to(device))

        if self.obj is not None:
            obj_pos = torch.tensor(self.current_obj_base_pos, dtype=torch.float32, device=device)
            obj_quat = xyz_to_quat(
                torch.tensor(self.current_obj_base_euler, dtype=torch.float32, device=device) * torch.pi / 180.0
            )
            self.obj.set_pos(obj_pos)
            self.obj.set_quat(obj_quat)

            if self.obj_dofs_idx and self.current_obj_dof_pos is not None:
                self.obj.set_dofs_pos(
                    position=self.current_obj_dof_pos,
                    dofs_idx_local=self.obj_dofs_idx,
                    zero_velocity=True,
                )

    def _update_grasp_target_markers(self) -> None:
        if self.obj is None or self.grasp_target_link_name is None:
            return

        device = self.env.device
        link = resolve_grasp_target_link(self.obj, link_name=self.grasp_target_link_name, default_link_name="handle")
        link_pos = link.get_pos()[0].detach().cpu().numpy().reshape(1, 3)
        target_pos = (
            get_grasp_target_pos(
                self.obj,
                link_name=self.grasp_target_link_name,
                use_aabb_center=self.grasp_target_use_aabb_center,
                local_offset=torch.tensor(self.current_grasp_target_offset, dtype=torch.float32, device=device),
                default_link_name="handle",
            )[0]
            .detach()
            .cpu()
            .numpy()
            .reshape(1, 3)
        )

        if self._grasp_target_link_marker is not None and hasattr(self._grasp_target_link_marker, "remove"):
            self._grasp_target_link_marker.remove()
        if self._grasp_target_point_marker is not None and hasattr(self._grasp_target_point_marker, "remove"):
            self._grasp_target_point_marker.remove()

        self._grasp_target_link_marker = self.server.scene.add_point_cloud(
            "/debug/grasp_target_link_origin",
            points=link_pos,
            colors=np.array([[0, 255, 0]], dtype=np.uint8),
            point_size=0.02,
        )
        self._grasp_target_point_marker = self.server.scene.add_point_cloud(
            "/debug/grasp_target_point",
            points=target_pos,
            colors=np.array([[255, 64, 64]], dtype=np.uint8),
            point_size=0.025,
        )

    def get_yaml_snippet(self) -> str:
        robot_quat = xyz_to_quat(torch.tensor(self.current_base_euler, dtype=torch.float32) * torch.pi / 180.0)

        lines = [
            "# Paste into your grasp-generator config override YAML",
            "scene:",
            "  robot:",
            f"    default_root_pos: {_format_vec(self.current_base_pos.tolist())}",
            f"    default_root_quat: {_format_vec(robot_quat.cpu().tolist())}",
        ]
        if self.robot_dofs_idx and self.current_dof_positions is not None:
            lines.append("    default_dofs_pos:")
            for name, value in zip(self.motors_name, self.current_dof_positions[0].cpu().tolist()):
                lines.append(f"      {name}: {value:.6f}")

        if self.obj is not None:
            obj_quat = xyz_to_quat(torch.tensor(self.current_obj_base_euler, dtype=torch.float32) * torch.pi / 180.0)
            lines.extend(
                [
                    "  obj:",
                    "    metadata:",
                    f"      grasp_target_link: {self.grasp_target_link_name or 'handle'}",
                    f"      grasp_target_use_aabb_center: {str(self.grasp_target_use_aabb_center).lower()}",
                    f"      grasp_target_local_offset: {_format_vec(self.current_grasp_target_offset.tolist())}",
                    f"    default_root_pos: {_format_vec(self.current_obj_base_pos.tolist())}",
                    f"    default_root_quat: {_format_vec(obj_quat.cpu().tolist())}",
                ]
            )
            if self.obj_dofs_idx and self.current_obj_dof_pos is not None:
                lines.append("# Object DOFs (not all task object options support YAML override):")
                for name, value in zip(self.obj_dofs_name, self.current_obj_dof_pos[0].cpu().tolist()):
                    lines.append(f"#   {name}: {value:.6f}")

        return "\n".join(lines)

    def update(self):
        self.update_robot_state()
        self._update_grasp_target_markers()
        super().update()


def main():
    parser = get_argparser(description="Visualize and control robot joints and object pose.")
    parser.add_argument(
        "--grasp-path",
        type=str,
        default=None,
        help="Path to a saved grasp `.pt` file to load instead of a YAML seed config.",
    )
    parser.add_argument(
        "--grasp-index",
        type=int,
        default=0,
        help="Index of the saved precomputed grasp to load when --grasp-path is set.",
    )
    parser.add_argument(
        "--num_envs",
        "-b",
        type=int,
        default=1,
        help="Number of environments to show.",
    )
    args = parser.parse_args()

    # Prefer the repo's maintained sample-grasp override when available so the
    # viewer starts from the same validated seed pose we use by default.
    sample_grasp_config = None
    if args.task and args.robot:
        sample_grasp_config = Path("conf/sample_grasps") / f"{args.task}_{args.robot}.yaml"

    if args.grasp_path is None and args.config is None and sample_grasp_config is not None:
        if sample_grasp_config.is_file():
            args.config = str(sample_grasp_config)
            print(f"Viewer: using default sample grasp config {args.config}")
    elif args.grasp_path is not None and args.config is None and sample_grasp_config is not None:
        if sample_grasp_config.is_file():
            print(
                "Viewer warning: --grasp-path was provided without --config, so the viewer is using the task default "
                f"scene instead of {sample_grasp_config}. If the grasp was generated from that YAML, pass "
                f"--config={sample_grasp_config} as well."
            )

    # Initialize Eden
    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path="logs/temp",
    )

    # Load task config
    config = get_task_config_from_args(args, upload_logs=False)
    config.env_options.num_envs = args.num_envs
    config.env_options.num_eval_envs = args.num_envs

    # Disable reset-time grasp sampling in this viewer. When a grasp file is supplied
    # we apply it directly after environment creation; otherwise the viewer shows the
    # configured seed pose from the YAML/task config.
    if hasattr(config, "event_options") and hasattr(config.event_options, "load_grasp"):
        del config.event_options.load_grasp
        if args.grasp_path is not None:
            print("Viewer: bypassing events.load_grasp and applying the requested saved grasp directly.")
        else:
            print("Viewer: disabled events.load_grasp to keep YAML default_dofs_pos as initial joint state.")

    # print("\nViewer config:")
    # print(config.model_dump_json(indent=2))

    # Create environment
    env = RslRlVecEnvWrapper.from_config(config, show_viewer=False, eval_mode=True)

    print("Environment created successfully")

    # Get robot entity from environment
    robot = env.unwrapped.entities["robot"]
    obj = env.unwrapped.entities.get("obj", None)

    if args.grasp_path is not None:
        grasps_path = _apply_precomputed_grasp(
            robot,
            obj,
            grasp_path=args.grasp_path,
            grasp_index=args.grasp_index,
        )
        print(f"Viewer: loaded precomputed grasp index {args.grasp_index} from {grasps_path}")

    print("Links:")
    for link in robot.links:
        print(f'  "{link.name}",')

    # Create viewer
    grasp_target_metadata = _load_grasp_target_metadata_from_yaml(args.config)
    viewer = RobotDofsViewer(env.unwrapped, robot, obj, grasp_target_metadata=grasp_target_metadata)
    viewer.build()

    if obj is not None and not viewer.obj_dofs_idx:
        print("Note: object has no controllable local DOFs; only object base pose controls are available.")

    last_time = time.perf_counter()
    dt = 1.0 / 100
    try:
        while True:
            cur_time = time.perf_counter()
            if (cur_time - last_time) < dt:
                time.sleep(dt - (cur_time - last_time))
            last_time = cur_time
            viewer.update()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("--------------------------------")
        print(viewer.get_yaml_snippet())
        print("--------------------------------")
        viewer.close()


if __name__ == "__main__":
    main()
