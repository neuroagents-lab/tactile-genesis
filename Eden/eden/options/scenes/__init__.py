"""Scene configuration option presets."""

from .empty import BimanualEmptySceneOptions, EmptySceneOptions
from .tabletop import KitchenCountertopSceneOptions, TabletopSceneOptions

# Short aliases
Empty = EmptySceneOptions
BimanualEmpty = BimanualEmptySceneOptions
Tabletop = TabletopSceneOptions
KitchenCountertop = KitchenCountertopSceneOptions

__all__ = [
    "EmptySceneOptions",
    "BimanualEmptySceneOptions",
    "TabletopSceneOptions",
    "KitchenCountertopSceneOptions",
    "Empty",
    "BimanualEmpty",
    "Tabletop",
    "KitchenCountertop",
]
