from __future__ import annotations

from eden.options.entities import BoxOptions, CylinderOptions, EntityOptions
from eden.options.materials import RigidMaterialOptions

from utils import get_asset_path

PASTEL_COLORS = (
    (1.0, 0.6, 0.7),
    (0.6, 1.0, 0.7),
    (0.7, 0.6, 1.0),
    (0.5, 1.0, 1.0),
    (1.0, 0.5, 1.0),
    (1.0, 1.0, 0.5),
)


class BallyCube(EntityOptions):
    """A beveled cube. Base size is 1m^3, so use scale to set the size."""

    file: str = get_asset_path("objects/bally.obj")
    material: RigidMaterialOptions = RigidMaterialOptions(rho=500.0, friction=1.0)
    scale: float = 0.05
    recompute_inertia: bool = True  # compute inertia from material


class DexCube(EntityOptions):
    """A colored cube with default size 5cm^3."""

    file: str = get_asset_path("objects/dex_cube.urdf")
    material: RigidMaterialOptions = RigidMaterialOptions(rho=500.0, friction=1.0)
    scale: float = 0.05
    recompute_inertia: bool = True  # compute inertia from material


class Bin(EntityOptions):
    """A bin with default size 1m^3."""

    file: str = get_asset_path("objects/bin.obj")
    scale: float = 1.0


OBJECTS_4CM_16 = [
    BoxOptions(size=(0.036, 0.040, 0.042)),
    BoxOptions(size=(0.042, 0.042, 0.042)),
    BoxOptions(size=(0.035, 0.035, 0.035)),
    BoxOptions(size=(0.040, 0.040, 0.040)),
    BoxOptions(size=(0.045, 0.045, 0.045)),
    BoxOptions(size=(0.042, 0.042, 0.042)),
    BoxOptions(size=(0.034, 0.035, 0.040)),
    BoxOptions(size=(0.031, 0.038, 0.042)),
    BoxOptions(size=(0.038, 0.045, 0.041)),
    BoxOptions(size=(0.039, 0.033, 0.044)),
    BoxOptions(size=(0.041, 0.037, 0.030)),
    BoxOptions(size=(0.042, 0.034, 0.043)),
    BoxOptions(size=(0.043, 0.039, 0.035)),
    BoxOptions(size=(0.040, 0.041, 0.036)),
    BoxOptions(size=(0.044, 0.039, 0.035)),
    CylinderOptions(height=0.042, radius=0.020),
    CylinderOptions(height=0.038, radius=0.017),
]
