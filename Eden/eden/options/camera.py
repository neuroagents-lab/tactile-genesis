"""Camera and multi-camera configuration options."""

from typing import Any, Literal

from genesis.typing import UnitVec4FType, Vec2IType, Vec3FType
from numpydantic import NDArray, Shape

from eden.options.options import ConfigurableOptions


class CameraOptions(ConfigurableOptions):
    """Configuration for a single camera.

    Parameters
    ----------
    model: Literal["pinhole", "thinlens"]
        Camera model.
    cam_pos: array-like[float, float, float]
        Camera position.
    cam_lookat: array-like[float, float, float]
        Camera lookat.
    cam_resolution: array-like[int, int]
        Camera resolution in (width, height).
    cam_up: array-like[float, float, float]
        Camera up vector.
    cam_intrinsic: NDArray]
        Camera intrinsic matrix. This is used to set the camera's field of view and resolution.
        If provided, `cam_fov` and `cam_resolution` will be ignored.
    cam_extrinsic: NDArray]
        Camera extrinsic matrix. This is used to set the camera's position, lookat, and up.
        The extrinsic matrix is the inverse of the transform from the camera frame to the world frame.
        If provided, `cam_pos`, `cam_lookat`, and `cam_up` will be ignored.
    cam_aperture: float
        Camera aperture.
    cam_fov: float
        Camera field of view.
    debug: bool
        Whether to enable debug mode.
    follow_entity_name: str
        Name of the entity to follow.
    attach_entity_name: str
        Name of the entity to attach the camera to.
    attach_link_name: str
        Name of the link to attach the camera to.
    attach_offset: array-like[float, float, float] | None
        Offset of the camera from the link.
    attach_quat: array-like[float, float, float, float] | None
        Quaternion of the camera from the link (w, x, y, z).
    background_image_path: str | None
        Replace the background with the image at the given path. Use this for SimplerEnv style fake backgrounds.
    background_entities: list[str]
        List of names of entities treated as background and thus overlayed by the background image.
    sensor_options: Any | None
        Genesis sensor options (e.g. RasterizerCameraOptions, ApolloCameraOptions).
        When provided, the camera uses scene.add_sensor() instead of scene.add_camera().
        The sensor backend only supports RGB rendering; depth/segmentation/normal/pointcloud
        are not available. Fields pos/lookat/fov/res/up are auto-populated from
        cam_pos/cam_lookat/cam_fov/cam_resolution/cam_up during build if not explicitly set.
    """

    model: Literal["pinhole", "thinlens"] = "pinhole"
    cam_pos: Vec3FType = (1.5, 0.0, 1)
    cam_lookat: Vec3FType = (0.0, 0.0, 0.5)
    cam_resolution: Vec2IType = (720, 480)
    cam_up: Vec3FType = (0.0, 0.0, 1.0)
    cam_intrinsic: NDArray[Shape["3, 3"], float] | None = None
    cam_extrinsic: NDArray[Shape["4, 4"], float] | None = None
    cam_aperture: float = 2.0
    cam_fov: float = 80.0
    debug: bool = False

    follow_entity_name: str = ""
    attach_entity_name: str = ""
    attach_link_name: str = ""
    attach_offset: Vec3FType | None = None
    attach_quat: UnitVec4FType | None = None

    background_image_path: str | None = None
    background_entities: list[str] = []

    sensor_options: Any | None = None


class CamerasOptions(ConfigurableOptions):
    """Container mapping camera names to their :class:`CameraOptions`.

    Parameters
    ----------
    <camera_name>: CameraOptions
        The camera configuration to be used for the given camera.
    """

    def __init__(self, **data):
        entity_data = {}
        for key, entity in data.items():
            if isinstance(entity, CameraOptions):
                entity_data[key] = entity
            else:
                entity_data[key] = CameraOptions(**entity)

        super().__init__(**entity_data)
