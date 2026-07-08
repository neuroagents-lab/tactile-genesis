"""Miscellaneous helpers (env-index sanitization, hashing, snake_case, key loading)."""

import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from importlib import metadata
from urllib.parse import unquote, urlparse

import numpy as np
import torch
from genesis.utils.misc import sanitize_index as gs_sanitize_index


def _selected_env_count(idx: slice | torch.Tensor, num_envs: int) -> int:
    """Return the number of environments selected by a normalized index."""
    if isinstance(idx, slice):
        start, stop, step = idx.indices(num_envs)
        return len(range(start, stop, step))
    if idx.dtype == torch.bool:
        return int(idx.sum().item())
    return int(idx.numel())


def sanitize_envs_idx(
    envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None,
    num_envs: int,
    *,
    prefer_slice: bool = True,
    name: str = "envs_idx",
    return_n_envs: bool = False,
) -> tuple[slice | torch.Tensor, int] | slice | torch.Tensor:
    """Normalize environment indices with an optional slice fast-path.

    Parameters
    ----------
    envs_idx: int | range | slice | tuple[int, ...] | list[int] | torch.Tensor | np.ndarray | None
        Environment selector.
    num_envs: int
        Total number of environments.
    prefer_slice: bool
        If True, preserve slices for faster backend indexing when possible.
    name: str
        Name used in validation errors.
    return_n_envs: bool
        If True, also return the number of environments selected by the index.

    Returns
    -------
    envs_idx: slice | torch.Tensor
        Normalized environment indices.
    n_envs: int, optional
        Number of environments selected by the index. Only returned if return_n_envs is True.
    """
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}")

    if isinstance(envs_idx, torch.Tensor) and envs_idx.dtype == torch.bool:
        out = envs_idx
    elif envs_idx is None:
        if prefer_slice:
            out = slice(None)
        else:
            out = gs_sanitize_index(None, -1, num_envs, 0, name)
    elif isinstance(envs_idx, slice):
        start, stop, step = envs_idx.indices(num_envs)
        if prefer_slice:
            out = slice(start, stop, step)
        else:
            out = gs_sanitize_index(range(start, stop, step), -1, num_envs, 0, name)
    else:
        # Delegate range checks and contiguous int tensor conversion to Genesis util.
        out = gs_sanitize_index(envs_idx, -1, num_envs, 0, name)

    if return_n_envs:
        return out, _selected_env_count(out, num_envs)
    return out


def verify_file_hash(path: str, expected_hash: str) -> bool:
    """Verify file integrity using SHA256."""
    if not os.path.exists(path):
        return False

    sha256_hash = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest() == expected_hash


def string_to_sha256_hex(text: str) -> str:
    """Convert a string to its SHA-256 hash as a hex string."""
    # Hash functions require bytes as input, so we encode the string.
    # 'utf-8' is a standard and safe choice.
    encoded_string = text.encode("utf-8")
    # Create a sha256 hash object
    hasher = hashlib.sha256()
    # Update the hash object with the bytes
    hasher.update(encoded_string)
    # Get the hexadecimal representation of the hash
    hex_digest = hasher.hexdigest()
    return hex_digest


def to_snake_case(camel_case_string):
    """Convert a CamelCase string to snake_case using regular expressions."""
    if not camel_case_string:
        return ""

    # Add an underscore before any uppercase letter.
    s1 = re.sub("([^_])([A-Z][a-z]+)", r"\1_\2", camel_case_string)
    # Handle cases like "HTTP" by adding an underscore between consecutive uppercase letters.
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _display_greeting(INFO_length):
    from eden import logger

    try:
        terminal_size = os.get_terminal_size()[0]
    except OSError:
        terminal_size = 80
    wave_width = int((terminal_size - INFO_length - 11) / 2)
    if wave_width % 2 == 0:
        wave_width -= 1
    wave_width = max(0, min(38, wave_width))
    bar_width = wave_width * 2 + 8
    wave = ("┈┉" * wave_width)[:wave_width]
    logger.info(f"~<╭{'─' * (bar_width)}╮>~")
    logger.info(f"~<│{wave}>~ ~~~~< Eden >~~~~ ~<{wave}│>~")
    logger.info(f"~<╰{'─' * (bar_width)}╯>~")


def get_editable_package_commit(package_name: str) -> str:
    """Get the package's git commit hash (editable install) or its version otherwise.

    Returns the git commit hash if the package is installed via ``pip install -e .``,
    otherwise the package version.
    """
    try:
        # Prefer Python metadata first. This works for pip and uv environments.
        dist = metadata.distribution(package_name)
        version = dist.version or ""

        # PEP 610 metadata is available for direct URL / VCS / editable installs.
        direct_url_text = dist.read_text("direct_url.json")
        if direct_url_text:
            direct_url = json.loads(direct_url_text)
            vcs_info = direct_url.get("vcs_info", {})
            commit_id = vcs_info.get("commit_id") or vcs_info.get("requested_revision")
            if commit_id:
                return str(commit_id)

            # Editable local installs can expose the project path through file:// URL.
            dir_info = direct_url.get("dir_info", {})
            if dir_info.get("editable"):
                parsed_url = urlparse(direct_url.get("url", ""))
                if parsed_url.scheme == "file":
                    package_dir = unquote(parsed_url.path)
                    if os.path.isdir(package_dir):
                        return (
                            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_dir)
                            .strip()
                            .decode("utf-8")
                        )

        if version:
            return version
    except Exception:
        pass

    # Fallback for environments where metadata lookup fails.
    try:
        package_path = subprocess.check_output([sys.executable, "-m", "pip", "show", package_name]).decode("utf-8")

        location = None
        version = None
        for line in package_path.splitlines():
            if line.startswith("Location:"):
                location = line.split(": ", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(": ", 1)[1].strip()

        if location:
            dist_info_pattern = os.path.join(
                location,
                package_name.replace("-", "_") + "-*.dist-info",
                "direct_url.json",
            )
            matches = glob.glob(dist_info_pattern)
            if matches:
                try:
                    with open(matches[0], "r") as f:
                        direct_url = json.load(f)
                    commit_id = direct_url.get("vcs_info", {}).get("commit_id") or direct_url.get("vcs_info", {}).get(
                        "requested_revision"
                    )
                    if commit_id:
                        return str(commit_id)
                except Exception:
                    pass

        for line in package_path.splitlines():
            if line.startswith("Editable project location: "):
                package_dir = line.split(": ", 1)[1].strip()
                return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_dir).strip().decode("utf-8")

        if version:
            return version

        return ""
    except Exception:
        return ""


# Logging Utils
def get_git_hash():
    return subprocess.check_output(["git", "log", "-n", "1", "--pretty=tformat:%H"]).strip().decode("utf-8")


def get_git_commit_date():
    return subprocess.check_output(["git", "log", "-n", "1", "--pretty=tformat:%cd"]).strip().decode("utf-8")


def get_now():
    return datetime.datetime.now().strftime("%y%b%dT%H%M")


def create_symlink(src, dst):
    if not os.path.exists(src):
        raise FileNotFoundError(f"[*] {src} not found")
    if os.path.islink(dst):
        os.remove(dst)
    os.symlink(src, dst)


def setup_workdir(config):
    log_dir = f"runs/{config.robot_name.lower()}_{config.run_group}/{get_now()}"

    os.makedirs(log_dir, exist_ok=True)
    config.log_dir = log_dir
    return config


def set_api_keys_from_files():
    """Load API keys from ``eden/generators/keys/`` into environment variables.

    Reads all ``*.key`` files from the ``eden/generators/keys/`` directory,
    extracts the API key from each file, and sets them as environment variables.

    The environment variable name will be derived from the filename.
    For example, a file named 'my_api.key' will set an environment
    variable named 'MY_API_KEY'.
    """
    keys_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), "generators", "keys")
    key_files_pattern = os.path.join(keys_directory, "*.key")

    if not os.path.isdir(keys_directory):
        print(f"Error: Directory not found - {keys_directory}")
        return

    key_files = glob.glob(key_files_pattern)

    if not key_files:
        print(f"No API key files found in {keys_directory}")
        return

    for key_file_path in key_files:
        try:
            # Derive the API name from the filename
            # e.g., /path/to/eden/generators/keys/some_api_name.key -> some_api_name
            base_name = os.path.basename(key_file_path)
            api_name_parts = base_name.split(".")[:-1]  # Remove .key extension
            api_name = "_".join(api_name_parts)

            if not api_name:
                print(f"Warning: Could not derive API name from file: {key_file_path}")
                continue

            # Construct the environment variable name (e.g., SOME_API_NAME_KEY)
            env_var_name = f"{api_name.upper()}"

            with open(key_file_path, "r") as f:
                api_key = f.read().strip()

            if api_key:
                os.environ[env_var_name] = api_key
                print(f"Successfully set environment variable: {env_var_name}")
            else:
                print(f"Warning: API key file is empty: {key_file_path}")

        except Exception as e:
            print(f"Error processing file {key_file_path}: {e}")


def get_src_dir():
    return os.path.dirname(os.path.dirname(__file__))


def get_assets_dir():
    return os.path.join(get_src_dir(), "assets")


def get_gs_assets_dir():
    # Resolve the actual Genesis package root (e.g. .../site-packages/genesis)
    # instead of matching arbitrary nested paths that merely contain "genesis".
    try:
        genesis_module = __import__("genesis")
        genesis_dir = os.path.dirname(os.path.abspath(genesis_module.__file__))
        assets_dir = os.path.join(genesis_dir, "assets")
        if os.path.isdir(assets_dir):
            return assets_dir
    except Exception:
        pass

    # Fallback: recover package root from sys.path entries that may include
    # nested paths like ".../genesis/ext/.../bin".
    for path in sys.path:
        normalized_path = os.path.abspath(path)
        marker = f"{os.sep}genesis{os.sep}"
        if marker in normalized_path:
            genesis_root = normalized_path.split(marker, 1)[0] + marker.rstrip(os.sep)
        elif os.path.basename(normalized_path) == "genesis":
            genesis_root = normalized_path
        else:
            continue

        assets_dir = os.path.join(genesis_root, "assets")
        if os.path.isdir(assets_dir):
            return assets_dir

    raise FileNotFoundError(
        "Could not locate the Genesis 'assets' directory. "
        "Tried resolving it via the 'genesis' package location and entries in sys.path."
    )
