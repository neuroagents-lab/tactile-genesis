"""Tabletop and kitchen-countertop scene options."""

from pydantic import Field

from eden.options.entities import EntityOptions, PlaneOptions, SceneOptions


class TabletopSceneOptions(SceneOptions):
    """
    Tabletop scene options.

    Parameters
    ----------
    robot: EntityOptions
        The entity configuration to be used for the robot.
    plane: PrimitiveOptions
        The plane entity configuration.
    table: EntityOptions
        The table entity configuration.
    """

    robot: EntityOptions
    plane: PlaneOptions = Field(default_factory=PlaneOptions)
    table: EntityOptions = Field(
        default_factory=lambda: EntityOptions(
            file="work_table.obj",
            is_fixed_base=True,
            default_root_pos=(0.5, 0.0, 0.0),
        )
    )


class KitchenCountertopSceneOptions(SceneOptions):
    """
    Kitchen countertop scene options.

    Parameters
    ----------
    robot: EntityOptions
        The entity configuration to be used for the robot.
    plane: PrimitiveOptions
        The plane entity configuration.
    countertop: EntityOptions
        The countertop entity configuration.
    """

    robot: EntityOptions
    plane: PlaneOptions = Field(default_factory=PlaneOptions)
    countertop: EntityOptions = Field(
        default_factory=lambda: EntityOptions(
            file="kitchen_counter.glb",
            is_fixed_base=True,
            up=(0, 0, 1),
            front=(0, -1, 0),
            default_root_pos=(0.5, 0.0, 0.0),
        )
    )
