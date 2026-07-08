"""Asset download (DDP-gated) and dependency extraction from model files."""

import os
import glob
import re
import xml.etree.ElementTree as ET

import yaml
import json
from huggingface_hub import snapshot_download

from eden.utils.distributed import barrier, is_distributed, is_main_process


def _snapshot_download_ddp(**kwargs) -> str:
    """Wrap ``snapshot_download``, gating downloads by DDP rank.

    In multi-GPU (DDP) mode, only rank 0 performs the download while other
    ranks wait at a barrier. In single-GPU mode, calls snapshot_download directly.
    """
    if is_distributed():
        if is_main_process():
            result = snapshot_download(**kwargs)
        barrier()
        # After barrier, all ranks can read from the local cache.
        # Non-rank-0 processes call snapshot_download which will find the
        # cached files (no actual network request).
        if not is_main_process():
            result = snapshot_download(**kwargs)
    else:
        result = snapshot_download(**kwargs)
    return result


def _extract_asset_paths_from_xml(xml_path: str) -> set[str]:
    """Extract file dependencies (meshes, textures, hfields) from a MuJoCo XML file.

    Resolves paths using the ``meshdir`` and ``texturedir`` compiler directives,
    and normalizes ``../`` traversals so the returned paths are clean relative paths.

    Parameters
    ----------
    xml_path : str
        Path to the XML file.

    Returns
    -------
    set[str]
        Asset file paths relative to the XML file directory.
    """
    asset_paths: set[str] = set()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get directory overrides from <compiler> element
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        texturedir = compiler.get("texturedir", "") if compiler is not None else ""

        asset_section = root.find("asset")
        if asset_section is not None:
            # Mesh files
            for mesh in asset_section.findall("mesh"):
                mesh_file = mesh.get("file")
                if mesh_file:
                    raw = os.path.join(meshdir, mesh_file) if meshdir else mesh_file
                    asset_paths.add(os.path.normpath(raw))

            # Texture files (only those with a `file` attribute)
            for texture in asset_section.findall("texture"):
                tex_file = texture.get("file")
                if tex_file:
                    raw = os.path.join(texturedir, tex_file) if texturedir else tex_file
                    asset_paths.add(os.path.normpath(raw))

            # Heightfield files
            for hfield in asset_section.findall("hfield"):
                hfield_file = hfield.get("file")
                if hfield_file:
                    asset_paths.add(os.path.normpath(hfield_file))

    except (ET.ParseError, FileNotFoundError, PermissionError):
        pass

    return asset_paths


def _extract_asset_paths_from_urdf(urdf_path: str) -> set[str]:
    """Extract mesh file paths from a URDF file.

    Handles ``package://`` and ``file://`` URI prefixes by stripping them.

    Parameters
    ----------
    urdf_path : str
        Path to the URDF file.

    Returns
    -------
    set[str]
        Mesh file paths relative to the URDF file directory.
    """
    asset_paths: set[str] = set()

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for mesh in root.findall(".//mesh"):
            mesh_filename = mesh.get("filename")
            if mesh_filename:
                # Strip URI prefixes
                if mesh_filename.startswith("package://"):
                    mesh_filename = mesh_filename[len("package://") :]
                elif mesh_filename.startswith("file://"):
                    mesh_filename = mesh_filename[len("file://") :]
                asset_paths.add(os.path.normpath(mesh_filename))

    except (ET.ParseError, FileNotFoundError, PermissionError):
        pass

    return asset_paths


def _extract_asset_paths_from_obj(obj_path: str) -> set[str]:
    """Extract material library (.mtl) file paths from a Wavefront OBJ file.

    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.

    Returns
    -------
    set[str]
        Material file paths relative to the OBJ file directory.
    """
    asset_paths: set[str] = set()

    try:
        with open(obj_path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line.startswith("mtllib "):
                    mtl_file = line[len("mtllib ") :].strip()
                    if mtl_file:
                        asset_paths.add(os.path.normpath(mtl_file))
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass

    return asset_paths


def _extract_asset_paths_from_mtl(mtl_path: str) -> set[str]:
    """Extract texture file paths from a Wavefront MTL file.

    Parameters
    ----------
    mtl_path : str
        Path to the MTL file.

    Returns
    -------
    set[str]
        Texture file paths relative to the MTL file directory.
    """
    asset_paths: set[str] = set()
    texture_directives = {
        "map_ka",
        "map_kd",
        "map_ks",
        "map_ns",
        "map_d",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "map_pr",
        "map_pm",
        "norm",
    }

    try:
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0]
                parts = line.split()
                if len(parts) >= 2 and parts[0].lower() in texture_directives:
                    # Texture path is the last token (options may precede it)
                    tex_file = parts[-1]
                    asset_paths.add(os.path.normpath(tex_file))
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass

    return asset_paths


def _extract_asset_paths_from_gltf(gltf_path: str) -> set[str]:
    """Extract external buffer and image URIs from a glTF JSON file.

    glTF JSONs reference their binary buffer(s) and texture image(s) via
    ``buffers[*].uri`` and ``images[*].uri``. Data URIs (``data:...``) are
    skipped because they're inlined. ``.glb`` files self-contain everything,
    so they're not handled here.

    Parameters
    ----------
    gltf_path : str
        Path to the glTF JSON file.

    Returns
    -------
    set[str]
        Asset file paths relative to the glTF file's directory.
    """
    asset_paths: set[str] = set()

    try:
        with open(gltf_path, "r") as f:
            doc = json.load(f)
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, json.JSONDecodeError):
        return asset_paths

    for section in ("buffers", "images"):
        for entry in doc.get(section, []) or []:
            uri = entry.get("uri") if isinstance(entry, dict) else None
            if not uri or uri.startswith("data:"):
                continue
            asset_paths.add(os.path.normpath(uri))

    return asset_paths


def _get_asset_dependencies(file_path: str) -> set[str]:
    """Get file dependencies (meshes, textures, materials) from a scene/model file.

    Parameters
    ----------
    file_path : str
        Path to the XML, URDF, OBJ, MTL, or glTF file.

    Returns
    -------
    set[str]
        Asset file paths relative to the descriptor file's directory.
    """
    if not os.path.exists(file_path):
        return set()

    file_lower = file_path.lower()
    if file_lower.endswith(".xml"):
        return _extract_asset_paths_from_xml(file_path)
    elif file_lower.endswith(".urdf"):
        return _extract_asset_paths_from_urdf(file_path)
    elif file_lower.endswith(".obj"):
        return _extract_asset_paths_from_obj(file_path)
    elif file_lower.endswith(".mtl"):
        return _extract_asset_paths_from_mtl(file_path)
    elif file_lower.endswith(".gltf"):
        return _extract_asset_paths_from_gltf(file_path)
    else:
        return set()


def _resolve_dep(dep_path: str, file_dir: str, file_dir_relative: str, asset_path: str) -> list[tuple[str, str]]:
    """Resolve a dependency path to a list of (dep_pattern, dep_local_path) candidates.

    Returns the path relative to the referring file's directory as the primary
    candidate. When the dependency path contains subdirectories (e.g.
    ``assets/base.dae``), a basename-only fallback is also included so that
    OBJ files with absolute-style mtllib paths are handled correctly.

    The caller should try candidates in order: use the first whose local path
    exists, or include all patterns when requesting a download.
    """
    if file_dir_relative != ".":
        dep_pattern = os.path.join(file_dir_relative, dep_path)
    else:
        dep_pattern = dep_path
    dep_pattern = os.path.normpath(dep_pattern)
    dep_local_path = os.path.join(asset_path, dep_pattern)

    primary = (dep_pattern, dep_local_path)

    if os.sep in dep_path:
        basename = os.path.basename(dep_path)
        fallback_local = os.path.join(file_dir, basename)
        # If the basename already exists locally, use it directly
        if os.path.exists(fallback_local):
            fb_pattern = os.path.relpath(fallback_local, asset_path)
            return [(fb_pattern, fallback_local)]
        # Otherwise return both primary and basename fallback for download
        if file_dir_relative != ".":
            fb_pattern = os.path.join(file_dir_relative, basename)
        else:
            fb_pattern = basename
        fb_pattern = os.path.normpath(fb_pattern)
        fb_local_path = os.path.join(asset_path, fb_pattern)
        if fb_pattern != dep_pattern:
            return [primary, (fb_pattern, fb_local_path)]

    return [primary]


def get_asset_path(
    file: str,
    registry: str | None = "Kashu7100",
    dataset: str | None = "eden_assets",
    local_dir: str | None = None,
) -> str:
    local_dir = local_dir or os.path.join(os.path.dirname(os.path.dirname((__file__))), "assets")

    if os.path.exists(file):
        return file
    elif registry is not None and dataset is not None:
        asset_path = local_dir
        matched_file_path = None

        # Try to find the file locally before downloading
        if not any(ch in file for ch in ["*", "?", "["]):
            if os.sep in file:
                repo_file = file
                local_dir_abs = os.path.abspath(local_dir)
                file_abs = os.path.abspath(file)
                if file_abs.startswith(local_dir_abs + os.sep):
                    repo_file = os.path.relpath(file_abs, local_dir_abs)
                candidate = os.path.join(local_dir, repo_file)
                if os.path.isfile(candidate):
                    matched_file_path = candidate
            else:
                # Plain filename - search under local_dir
                for path in glob.glob(os.path.join(local_dir, "**", file), recursive=True):
                    if os.path.isfile(path):
                        matched_file_path = path
                        break

        if matched_file_path is None:
            # File not found locally, download from HuggingFace
            # NOTE: HF allow_patterns uses fnmatch, where **/ translates to .*/
            # in regex and requires at least one char before /, so avoid **/ prefix.
            if any(ch in file for ch in ["*", "?", "["]):
                allow_patterns = file
            elif os.sep in file:
                repo_file = file
                local_dir_abs = os.path.abspath(local_dir)
                file_abs = os.path.abspath(file)
                if file_abs.startswith(local_dir_abs + os.sep):
                    repo_file = os.path.relpath(file_abs, local_dir_abs)
                # Use both exact and wildcard patterns to handle files at repo
                # root (exact match) and files nested under subdirectories
                # (wildcard match). fnmatch treats */ as .*/ in regex, which
                # requires at least one char before /, so it only matches nested.
                allow_patterns = [repo_file, f"*/{repo_file}"]
            else:
                allow_patterns = f"*{file}*"
            asset_path = _snapshot_download_ddp(
                repo_type="dataset",
                repo_id=f"{registry}/{dataset}",
                allow_patterns=allow_patterns,
                local_dir=local_dir,
            )
            # search for the matched asset path under asset_path
            for path in glob.glob(os.path.join(asset_path, "**"), recursive=True):
                if os.path.isdir(path):
                    continue
                if file in path:
                    matched_file_path = path
                    break

            if matched_file_path is None:
                raise FileNotFoundError(f"Asset {file} not found in {asset_path}")

        # Download asset dependencies (meshes, textures, materials, etc.)
        # Walk the dep graph layer by layer, BFS: every file at the current
        # depth is parsed for refs, then ALL missing refs are downloaded in a
        # single batched ``snapshot_download`` call before advancing to the
        # next layer. URDFs in particular reference dozens of ``.obj`` files,
        # which each reference one ``.mtl`` — batching avoids one HF round-trip
        # per file (laptop URDF: 1 + 37 + 37 + textures ≈ 4 batched calls vs
        # the previous ~100 single-file calls).
        checked_files: set[str] = {matched_file_path}
        current_layer: list[str] = [matched_file_path]

        while current_layer:
            next_missing_patterns: list[str] = []
            next_local_paths: list[str] = []
            already_local: list[str] = []

            for current_file in current_layer:
                dep_paths = _get_asset_dependencies(current_file)
                if not dep_paths:
                    continue

                file_dir = os.path.dirname(current_file)
                file_dir_relative = os.path.relpath(file_dir, asset_path)

                for dep_path in dep_paths:
                    candidates = _resolve_dep(dep_path, file_dir, file_dir_relative, asset_path)
                    found_local = False
                    for _dep_pattern, dep_local_path in candidates:
                        if os.path.exists(dep_local_path):
                            if dep_local_path not in checked_files:
                                already_local.append(dep_local_path)
                            found_local = True
                            break
                    if not found_local:
                        for dep_pattern, dep_local_path in candidates:
                            next_missing_patterns.append(dep_pattern.replace(os.sep, "/"))
                            next_local_paths.append(dep_local_path)

            # Dedup patterns + their matched local paths, preserving order.
            seen: set[str] = set()
            unique_patterns: list[str] = []
            unique_local_paths: list[str] = []
            for pattern, local_path in zip(next_missing_patterns, next_local_paths):
                if pattern not in seen:
                    seen.add(pattern)
                    unique_patterns.append(pattern)
                    unique_local_paths.append(local_path)

            if unique_patterns:
                _snapshot_download_ddp(
                    repo_type="dataset",
                    repo_id=f"{registry}/{dataset}",
                    allow_patterns=unique_patterns,
                    local_dir=local_dir,
                )

            next_layer: list[str] = []
            for p in already_local:
                if p not in checked_files:
                    checked_files.add(p)
                    next_layer.append(p)
            for p in unique_local_paths:
                if p not in checked_files and os.path.exists(p):
                    checked_files.add(p)
                    next_layer.append(p)
            current_layer = next_layer

        return matched_file_path


def find_latest_checkpoint(log_dir: str, return_path: bool = True) -> str:
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} not found")
    if not os.path.isdir(log_dir):
        raise NotADirectoryError(f"Log directory {log_dir} is not a directory")
    if not os.listdir(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} is empty")
    if not any(filename.endswith(".pt") for filename in os.listdir(log_dir)):
        raise FileNotFoundError(f"No model checkpoints in {log_dir}")

    model_pattern = re.compile(r"^model_(\d+)\.pt$")
    candidates = []
    for filename in os.listdir(log_dir):
        match = model_pattern.match(filename)
        if match:
            candidates.append((int(match.group(1)), filename))
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints matching 'model_{{itr}}.pt' in {log_dir}")
    if return_path:
        return os.path.join(log_dir, max(candidates, key=lambda item: item[0])[1])
    else:
        return max(candidates, key=lambda item: item[0])[1]


def load_text(filepath: str) -> dict:
    try:
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except OSError as e:
        # Provide a clearer error message while preserving the original exception context.
        raise FileNotFoundError(f"Unable to open file '{filepath}': {e}") from e
