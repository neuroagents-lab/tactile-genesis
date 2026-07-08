from pathlib import Path
from types import SimpleNamespace


def get_asset_path(asset_name: str) -> str:
    assets_dir = Path(__file__).parent / "assets"
    asset_path = assets_dir / asset_name
    if not asset_path.exists():
        print(f"WARNING: Asset '{asset_name}' not found in {assets_dir}")
    return str(asset_path)


def get_entity_metadata(entity):
    options = getattr(entity, "options", None)
    metadata = getattr(options, "metadata", None)
    if metadata is None:
        metadata = getattr(entity, "metadata", None)
    if isinstance(metadata, dict):
        metadata = SimpleNamespace(**metadata)
    return metadata


def get_entity_link_names(entity) -> tuple[str, ...]:
    """Return runtime link names exposed by an entity."""
    links = getattr(entity, "links", None)
    if links is None:
        links = getattr(getattr(entity, "_entity", None), "_links", None)

    names = []
    for link in links or ():
        name = getattr(link, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def resolve_entity_link(entity, link_name: str, *, fallback_link_names: tuple[str, ...] = ()):
    """Resolve a runtime entity link, trying fallbacks before raising."""
    candidate_names = []
    for candidate in (link_name, *fallback_link_names):
        if candidate and candidate not in candidate_names:
            candidate_names.append(candidate)

    last_error = None
    available_link_names = get_entity_link_names(entity)
    for candidate in candidate_names:
        try:
            return entity.get_link(candidate)
        except Exception as exc:  # pragma: no cover - backend-dependent
            last_error = exc
        suffix_matches = [name for name in available_link_names if name.endswith(f"_{candidate}")]
        if len(suffix_matches) == 1:
            return entity.get_link(suffix_matches[0])

    available_suffix = f" Available link names: {list(available_link_names)}." if available_link_names else ""
    requested = candidate_names[0] if candidate_names else link_name
    raise ValueError(f"Could not resolve link '{requested}'.{available_suffix}") from last_error
