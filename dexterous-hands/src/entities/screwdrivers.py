from eden.options.entities import EntityOptions, GroupedEntityOptions
from eden.options.materials import RigidMaterialOptions
from genesis.utils.geom import euler_to_quat

from utils import get_asset_path

UPRIGHT_ROOT_QUAT = tuple(euler_to_quat((90.0, 0.0, 0.0)))


class ScrewdriverObj(EntityOptions):
    material: RigidMaterialOptions = RigidMaterialOptions(rho=4000.0, friction=1.0)
    scale: float = 0.2032  # 8 inches in meters
    recompute_inertia: bool = True  # compute inertia from material


class FatScrewdrivers(GroupedEntityOptions):
    """Screwdriver objects with 1.5x x/y scale to make it suitable for the fat xhand1 fingers."""

    default_root_quat: tuple[float, float, float, float] = UPRIGHT_ROOT_QUAT
    grouped_entities: list[EntityOptions] = [
        ScrewdriverObj(file=get_asset_path("screwdriver/screwdriver0_fat.obj")),
        ScrewdriverObj(file=get_asset_path("screwdriver/screwdriver1_fat.obj")),
        ScrewdriverObj(file=get_asset_path("screwdriver/screwdriver2_fat.obj")),
        ScrewdriverObj(file=get_asset_path("screwdriver/screwdriver3_fat.obj")),
    ]
