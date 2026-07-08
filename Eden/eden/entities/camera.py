"""Camera entity wrappers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import genesis as gs
from genesis.typing import Vec2IType, Vec3FType
from genesis.options.renderers import BatchRenderer
from genesis.utils.geom import T_to_pos_lookat_up, trans_quat_to_T
from genesis.utils.misc import tensor_to_array
import numpy as np
from PIL import Image
import torch

import eden as en
from eden.utils.common import ConfigurableMixin
from eden.utils.misc import get_now

from eden.options.camera import CameraOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


class Camera(ConfigurableMixin[CameraOptions]):
    """Wrapper around a Genesis camera, configured from :class:`CameraOptions`.

    Parameters
    ----------
    model: Literal["pinhole", "thinlens", "fisheye"]
        Camera model.
    cam_pos: array-like[float, float, float]
        Camera position in world frame.
    cam_lookat: array-like[float, float, float]
        Camera lookat in world frame.
    cam_resolution: array-like[int, int]
        Camera resolution in (width, height).
    cam_up: array-like[float, float, float]
        Camera up vector.
    cam_aperture: float
        Camera aperture.
    cam_fov: float
        The vertical field of view of the camera in degrees.
    debug: bool
        Whether this is a debug camera. A debug camera will use non-batch renderer.
    follow_entity_name: str
        Name of the entity to follow.
    attach_entity_name: str
        Name of the entity to attach the camera to.
    attach_link_name: str
        Name of the link to attach the camera to.
    background_image_path: str | None
        Replace the background with the image at the given path. Use this for SimplerEnv style fake backgrounds.
    background_entities: list[str]
        List of names of entities treated as background and thus overlayed by the background image.
    """

    model: Literal["pinhole", "thinlens", "fisheye"] = "pinhole"
    cam_pos: Vec3FType = (5.0, 0.0, 1.8)
    cam_lookat: Vec3FType = (0.0, 0.0, 0.5)
    cam_resolution: Vec2IType = (1280, 720)
    cam_up: Vec3FType = (0.0, 0.0, 1.0)
    cam_aperture: float = 2.0
    cam_fov: float = 55.0
    debug: bool = False
    follow_entity_name: str = ""
    attach_entity_name: str = ""
    attach_link_name: str = ""
    background_image_path: str | None = None
    background_entities: list[str] = []

    if TYPE_CHECKING:
        attach_T: np.ndarray  # 4x4 transformation matrix

    def __init__(self, env: EnvBase, options: CameraOptions):
        self._uid = gs.UID()
        self._options = options
        self._env = env

        # Copy fields from options to self (without mutating options)
        for name in self.get_parameter_names():
            if name in self._options.model_dump():
                setattr(self, name, getattr(self._options, name))
            else:
                setattr(self, name, getattr(self, name))

        # Resolve cam_intrinsic → override cam_fov / cam_resolution on self
        if self._options.cam_intrinsic is not None:
            intrinsic = self._options.cam_intrinsic
            assert intrinsic.shape == (3, 3), "`cam_intrinsic` should be a 3x3 matrix"
            en.logger.warning("`cam_intrinsic` is provided, ignoring `cam_fov` and `cam_resolution`")
            self.cam_fov = float(2 * np.arctan(intrinsic[1, 2] / intrinsic[1, 1]) * 180 / np.pi)
            self.cam_resolution = (
                int(math.ceil(intrinsic[0, 2] * 2)),
                int(math.ceil(intrinsic[1, 2] * 2)),
            )
            en.logger.info(
                f"Using `cam_intrinsic` to set `cam_fov`: {self.cam_fov} and `cam_resolution`: {self.cam_resolution}"
            )

        # Resolve cam_extrinsic → override cam_pos / cam_lookat / cam_up on self
        if self._options.cam_extrinsic is not None:
            extrinsic = self._options.cam_extrinsic
            assert extrinsic.shape == (4, 4), "`cam_extrinsic` should be a 4x4 matrix"
            en.logger.warning("`cam_extrinsic` is provided, ignoring `cam_pos`, `cam_lookat`, and `cam_up`")
            T = np.linalg.inv(extrinsic)
            pos, lookat, up = T_to_pos_lookat_up(T)
            self.cam_pos = tuple(pos)
            self.cam_lookat = tuple(lookat)
            self.cam_up = tuple(up)

        # Validate follow vs attach
        if self._options.follow_entity_name != "":
            assert self._options.attach_entity_name == "", (
                "`attach_entity_name` should be empty if `follow_entity_name` is provided"
            )
            assert self._options.attach_link_name == "", (
                "`attach_link_name` should be empty if `follow_entity_name` is provided"
            )
            assert self._options.attach_offset is None, (
                "`attach_offset` should be None if `follow_entity_name` is provided"
            )
            assert self._options.attach_quat is None, "`attach_quat` should be None if `follow_entity_name` is provided"
        elif self._options.attach_entity_name != "":
            assert self._options.follow_entity_name == "", (
                "`follow_entity_name` should be empty if `attach_entity_name` is provided"
            )
            assert self._options.attach_link_name != "", (
                "`attach_link_name` should be provided if `attach_entity_name` is provided"
            )
            assert self._options.attach_offset is not None, (
                "`attach_offset` should be provided if `attach_entity_name` is provided"
            )
            assert self._options.attach_quat is not None, (
                "`attach_quat` should be provided if `attach_entity_name` is provided"
            )

        # Compute attach_T from offset/quat (stored on self, not options)
        if self._options.attach_entity_name != "" and self._options.attach_offset is not None:
            self.attach_T = trans_quat_to_T(
                np.array(self._options.attach_offset),
                np.array(self._options.attach_quat),
            )

        # Determine backend: sensor (add_sensor) vs legacy (add_camera)
        self._sensor_options = self._options.sensor_options
        self._cam = None  # Legacy camera object (set in pre_build)
        self._sensor = None  # Sensor object (set in pre_build)
        self._recording_frames = []  # For sensor-backend video recording
        self._is_recording = False

        self.background_image = None
        if self.background_image_path is not None:
            if self._sensor_options is not None:
                en.logger.warning(
                    "background_image_path requires segmentation which is not available "
                    "with the sensor backend. Background overlay will be disabled."
                )
            else:
                self.background_image = np.array(
                    Image.open(self.background_image_path).convert("RGB").resize(self.cam_resolution)
                )
                if isinstance(self._env.renderer_options, BatchRenderer):
                    self.background_image = torch.tensor(self.background_image, device=self._env.device)

    @property
    def _uses_sensor_backend(self) -> bool:
        """Whether this camera uses the sensor API (add_sensor) instead of legacy add_camera."""
        return self._sensor_options is not None

    def pre_build(self):
        if self._uses_sensor_backend:
            self._pre_build_sensor()
        else:
            self._pre_build_legacy()

    def _pre_build_legacy(self):
        """Build camera using the legacy scene.add_camera() API."""
        self._cam = self._env.scene.add_camera(
            model=self.model,
            pos=self.cam_pos,
            lookat=self.cam_lookat,
            res=self.cam_resolution,
            fov=self.cam_fov,
            up=self.cam_up,
            aperture=self.cam_aperture,
            GUI=False,
            debug=self.debug,
        )

        # NOTE: moving camera for moving entity
        if self.follow_entity_name != "":
            if self.follow_entity_name not in self._env.entities:
                raise ValueError(
                    f"Camera references non-existent follow entity: '{self.follow_entity_name}'. "
                    f"Available entities: {list(self._env.entities.keys())}"
                )
            self._cam.follow_entity(
                self._env.entities[self.follow_entity_name]._entity,
                fix_orientation=True,
            )

        if self.attach_entity_name != "":
            if self.attach_entity_name not in self._env.entities:
                raise ValueError(
                    f"Camera references non-existent attach entity: '{self.attach_entity_name}'. "
                    f"Available entities: {list(self._env.entities.keys())}"
                )
            rigid_link = self._env.entities[self.attach_entity_name].get_link(self.attach_link_name)
            self._cam.attach(rigid_link, self.attach_T)

    def _pre_build_sensor(self):
        """Build camera using the sensor API (scene.add_sensor())."""
        sensor_opts = self._sensor_options.model_copy(deep=True)

        # Auto-populate sensor_options fields from CameraOptions for any field
        # the user did not explicitly set when constructing sensor_options.
        _defaults = {
            "pos": list(self.cam_pos),
            "lookat": list(self.cam_lookat),
            "up": list(self.cam_up),
            "fov": self.cam_fov,
            "res": self.cam_resolution,
        }
        for attr, val in _defaults.items():
            if hasattr(sensor_opts, attr) and attr not in sensor_opts.model_fields_set:
                setattr(sensor_opts, attr, val)

        # Inject entity attachment if configured via CameraOptions
        if self.attach_entity_name != "":
            if self.attach_entity_name not in self._env.entities:
                raise ValueError(
                    f"Camera sensor references non-existent entity: '{self.attach_entity_name}'. "
                    f"Available entities: {list(self._env.entities.keys())}"
                )
            entity = self._env.entities[self.attach_entity_name]
            sensor_opts.entity_idx = entity.idx
            if self.attach_link_name:
                link = entity.get_link(self.attach_link_name)
                sensor_opts.link_idx_local = link.idx_local
            if hasattr(self, "attach_T"):
                sensor_opts.offset_T = self.attach_T
        elif self.attach_link_name:
            raise ValueError("Should provide attach_entity_name if attach_link_name is provided.")

        self._sensor = self._env.scene.add_sensor(sensor_opts)

    def post_build(self):
        if self._uses_sensor_backend:
            # Sensor backend doesn't support segmentation, skip background_ids setup
            return

        seg_idx_dict = {}
        for seg_idx, scene_elem in self._env.segmentation_idx_dict.items():
            if isinstance(scene_elem, (list, tuple)):
                entity_idx = scene_elem[0]  # NOTE: take out entity idx
            else:
                entity_idx = scene_elem
            seg_idx_dict[entity_idx] = seg_idx

        if self.background_entities:
            self.background_ids = [0] + [
                seg_idx_dict[self._env.entities[entity_name].idx] for entity_name in self.background_entities
            ]
            if isinstance(self._env.renderer_options, BatchRenderer):
                self.background_ids = torch.tensor(self.background_ids, device=self._env.device)
        else:
            # NOTE: segmentation idx 0 is the background
            self.background_ids = [0]
            if isinstance(self._env.renderer_options, BatchRenderer):
                self.background_ids = torch.tensor(self.background_ids, device=self._env.device)

    @property
    def segmentation_idx_dict(self):
        return self._env.scene.segmentation_idx_dict

    def get_pos(self, envs_idx=None):
        """Return the current position of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "get_pos() is not supported with the sensor backend. Use sensor.read() to get rendered data."
            )
        return self._cam.get_pos(envs_idx=envs_idx)

    def get_quat(self, envs_idx=None):
        """Return the current quaternion of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "get_quat() is not supported with the sensor backend. Use sensor.read() to get rendered data."
            )
        return self._cam.get_quat(envs_idx=envs_idx)

    def get_lookat(self, envs_idx=None):
        """Return the current lookat point of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_lookat() is not supported with the sensor backend.")
        return self._cam.get_lookat(envs_idx=envs_idx)

    def get_up(self, envs_idx=None):
        """Return the current up vector of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_up() is not supported with the sensor backend.")
        return self._cam.get_up(envs_idx=envs_idx)

    def get_transform(self, envs_idx=None):
        """Return the current transform matrix of the camera."""
        if self._uses_sensor_backend:
            raise NotImplementedError("get_transform() is not supported with the sensor backend.")
        return self._cam.get_transform(envs_idx=envs_idx)

    def set_pose(self, transform=None, pos=None, lookat=None, up=None, envs_idx=None):
        """Set the camera pose.

        Parameters
        ----------
        transform : np.ndarray, shape (4, 4) or (N, 4, 4), optional
            The transform matrix of the camera.
        pos : array-like, shape (3,) or (N, 3), optional
            The position of the camera.
        lookat : array-like, shape (3,) or (N, 3), optional
            The lookat point of the camera.
        up : array-like, shape (3,) or (N, 3), optional
            The up vector of the camera.
        envs_idx : array of indices in integers, optional
            The environment indices for which to update the pose. If not provided, the camera pose will be set for the
            specific environment bound to the camera if any, all the environments otherwise.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_pose() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set pose on an attached camera. Detach it first or modify the attachment offset."
            )
        self._cam.set_pose(transform=transform, pos=pos, lookat=lookat, up=up, envs_idx=envs_idx)

    def set_pos(self, pos, envs_idx=None):
        """Set the camera position, keeping current lookat and up.

        Parameters
        ----------
        pos : array-like, shape (3,) or (N, 3)
            The position of the camera.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_pos() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set position on an attached camera. Detach it first or modify the attachment offset."
            )
        self._cam.set_pose(pos=pos, envs_idx=envs_idx)

    def set_quat(self, quat, envs_idx=None):
        """Set the camera orientation from a quaternion (wxyz), keeping current position.

        Parameters
        ----------
        quat : array-like, shape (4,) or (N, 4)
            Quaternion in (w, x, y, z) format.
        envs_idx : array-like, optional
            Environment indices. If None, applies to all environments.
        """
        if self._uses_sensor_backend:
            raise NotImplementedError("set_quat() is not supported with the sensor backend.")
        if self.is_attached:
            raise RuntimeError(
                "Cannot set orientation on an attached camera. Detach it first or modify the attachment offset."
            )
        quat = torch.as_tensor(quat, dtype=torch.float32, device=self._env.device)
        # Use internal _pos (without envs_offset) to match set_pose's coordinate space
        idx = () if envs_idx is None else envs_idx
        pos = self._cam._pos[idx]
        T = trans_quat_to_T(pos, quat)
        self._cam.set_pose(transform=T, envs_idx=envs_idx)

    @property
    def is_attached(self) -> bool:
        """Whether this camera is attached to an entity link."""
        if self._uses_sensor_backend:
            entity_idx = getattr(self._sensor_options, "entity_idx", None)
            return entity_idx is not None and entity_idx != -1
        return bool(self.attach_entity_name)

    def start_recording(self):
        if self._uses_sensor_backend:
            self._is_recording = True
            self._recording_frames = []
            return
        self._cam.start_recording()

    def stop_recording(self, video_path: str | None = None, fps: int = 60):
        """
        Stop recording and save accumulated frames to a video file.

        Parameters
        ----------
        video_path : str, optional
            Output video file path. Defaults to ``{log_dir}/video_{timestamp}.mp4``.
        fps : int, optional
            Frames per second of the output video.
        """
        video_path = video_path or f"{en.log_dir}/video_{get_now()}.mp4"
        if self._uses_sensor_backend:
            self._is_recording = False
            if self._recording_frames:
                gs.tools.animate(self._recording_frames, video_path, fps)
            self._recording_frames = []
            return
        self._cam.stop_recording(save_to_filename=video_path, fps=fps)

    def read(self, envs_idx=None):
        """Read sensor data. Only available with the sensor backend.

        Returns
        -------
        data : CameraData
            Named tuple with `rgb` field containing the rendered image.
        """
        if not self._uses_sensor_backend:
            raise RuntimeError(
                "read() is only available with the sensor backend. "
                "Use render_rgb()/render_depth()/etc. for legacy cameras."
            )
        return self._sensor.read(envs_idx=envs_idx)

    def render(
        self,
        rgb: bool = True,
        depth: bool = False,
        segmentation: bool = False,
        colorize_seg: bool = False,
        normal: bool = False,
    ):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "render() with multiple outputs is not supported with the sensor backend. "
                "Use render_rgb() or read() instead."
            )
        rgb_out, depth_out, segmentation_out, normal_out = self._cam.render(
            rgb=rgb,
            depth=depth,
            segmentation=segmentation,
            colorize_seg=colorize_seg,
            normal=normal,
        )
        return rgb_out, depth_out, segmentation_out, normal_out

    def render_rgb(self):
        if self._uses_sensor_backend:
            data = self._sensor.read()
            rgb_out = data.rgb
            if self._is_recording:
                frame = tensor_to_array(rgb_out[0] if rgb_out.ndim > 3 else rgb_out)
                self._recording_frames.append(frame)
            return rgb_out

        if self.background_image is not None:
            rgb_out, _, segm, _ = self.render(
                rgb=True,
                depth=False,
                segmentation=True,
                colorize_seg=False,
                normal=False,
            )
            if isinstance(segm, np.ndarray):
                mask = np.isin(segm, self.background_ids)[..., None]
                ops = np.where
            elif isinstance(segm, torch.Tensor):
                mask = torch.isin(segm, self.background_ids)[..., None]
                ops = torch.where
            else:
                raise ValueError(f"Invalid type of segm: {type(segm)}")
            if rgb_out.ndim == 4:
                rgb_out = ops(mask, self.background_image[None], rgb_out)
            elif rgb_out.ndim == 3:
                rgb_out = ops(mask, self.background_image, rgb_out)
            else:
                raise ValueError(f"Invalid shape of rgb_out: {rgb_out.shape}")

            is_recording = self._cam._in_recording and self._cam._recorded_t_prev != self._env.scene._t
            if is_recording:
                self._cam._recorded_imgs[-1] = tensor_to_array(rgb_out)
        else:
            rgb_out, _, _, _ = self.render(
                rgb=True,
                depth=False,
                segmentation=False,
                colorize_seg=False,
                normal=False,
            )
        return rgb_out

    def render_depth(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Depth rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, depth_out, _, _ = self.render(rgb=False, depth=True, segmentation=False, colorize_seg=False, normal=False)
        return depth_out

    def render_pointcloud(self, world_frame=True):
        """Render a point cloud from the camera's current view.

        Parameters
        ----------
        world_frame : bool, optional
            Whether the point cloud is on camera frame or world frame.

        Returns
        -------
        pc : np.ndarray
            Numpy array of shape (res[0], res[1], 3) or (N, res[0], res[1], 3).
            Represents the point cloud in each pixel.
        mask_arr : np.ndarray
            The valid depth mask. boolean array of same shape as depth_arr
        """
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Pointcloud rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        pc, valid_mask = self._cam.render_pointcloud(world_frame=world_frame)
        return pc, valid_mask

    def render_segm(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Segmentation rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, _, segmentation_out, _ = self.render(
            rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False
        )
        return segmentation_out

    def render_normal(self):
        if self._uses_sensor_backend:
            raise NotImplementedError(
                "Normal rendering is not supported with the sensor backend. "
                "Use the legacy camera backend (remove sensor_options from CameraOptions)."
            )
        _, _, _, normal_out = self.render(rgb=False, depth=False, segmentation=False, colorize_seg=False, normal=True)
        return normal_out

    def snapshot(self, image_path: str | None = None):
        image_path = image_path or f"{en.log_dir}/snapshot_{get_now()}.png"
        rgb = self.render_rgb()
        if rgb.ndim == 4:
            rgb = rgb[0]
        rgb = tensor_to_array(rgb)
        gs.tools.save_img_arr(rgb, image_path)

    @property
    def sensor(self):
        """Access the underlying Genesis sensor object. Only available with sensor backend."""
        if not self._uses_sensor_backend:
            raise RuntimeError("sensor property is only available with the sensor backend.")
        return self._sensor
