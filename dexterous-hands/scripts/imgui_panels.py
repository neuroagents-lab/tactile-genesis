"""Shared ImGui sub-panels for the viewer scripts.

Reused by:
- ``scripts/sensor_probes_selector.py`` (``ProbeControlPanel`` embeds ``DofSliderPanel``)
- ``scripts/hand_tactile_sandbox.py`` (manual DOF sliders + YAML sequence pause / load)

Each helper is designed to be composed inside a larger ImGui panel: it renders into
the caller's ``imgui`` context and (where applicable) returns whether anything changed
so the host can react (e.g. apply targets to the entity).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import genesis as gs
import numpy as np
from genesis.utils.misc import tensor_to_array


def get_dof_info(entity: Any) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-DOF ``(names, lower, upper)`` for a fixed-base articulated entity.

    Fixed joints are skipped; free / spherical joints are not expected for a fixed
    hand. Non-finite limits (continuous joints) are clamped to ``[-pi, pi]`` so they
    map onto a usable slider range.
    """
    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint in entity.joints:
        if joint.n_dofs == 0 or joint.type in (gs.JOINT_TYPE.FIXED, gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.SPHERICAL):
            continue
        for i in range(joint.n_dofs):
            names.append(joint.name if joint.n_dofs == 1 else f"{joint.name}[{i}]")
            lo = float(joint.dofs_limit[i, 0])
            hi = float(joint.dofs_limit[i, 1])
            lower.append(lo if np.isfinite(lo) else -np.pi)
            upper.append(hi if np.isfinite(hi) else np.pi)
    return names, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def format_dofs_as_yaml_pose(names: list[str], targets: np.ndarray) -> str:
    """Format ``(names, targets)`` as one YAML ``sequence:`` step ready to paste.

    Output matches the format consumed by ``hand_tactile_sandbox``: a ``- joint: val``
    block under a top-level ``sequence:`` list, two-space indented.
    """
    lines: list[str] = []
    for i, name in enumerate(names):
        prefix = "  - " if i == 0 else "    "
        lines.append(f"{prefix}{name}: {float(targets[i]):.6f}")
    return "\n".join(lines)


def format_camera_pose(viewer: Any) -> str:
    """Format the viewer's current camera pose as a paste-ready ``ViewerOptions`` snippet."""

    def fmt(vec: Any) -> str:
        return "(" + ", ".join(f"{float(x):.4f}" for x in np.asarray(vec, dtype=np.float64)) + ")"

    return (
        "Current camera pose -- paste into gs.options.ViewerOptions(...):\n"
        f"        camera_pos={fmt(viewer.camera_pos)},\n"
        f"        camera_lookat={fmt(viewer.camera_lookat)},\n"
        f"        camera_fov={viewer.camera_fov},"
    )


def render_camera_pose_button(imgui: Any, viewer: Any, *, id_suffix: str = "camera_pose") -> bool:
    """Render a 'Print camera pose' button; prints a ``ViewerOptions`` snippet to stdout.

    Returns ``True`` on the frame the button was pressed. ``viewer`` must expose
    ``camera_pos``, ``camera_lookat``, and ``camera_fov`` (Genesis's ``Viewer`` does).
    """
    if imgui.button(f"Print camera pose##{id_suffix}_print"):
        print(format_camera_pose(viewer))
        return True
    return False


class DofSliderPanel:
    """ImGui sub-panel: a per-DOF slider for every actuated joint of an articulated entity.

    The panel mutates a caller-owned ``targets`` numpy array in place via ``render()``;
    the host script reads that array and drives the entity. With ``readonly=True``
    the sliders are replaced by static text so a different source (e.g. a YAML cycle)
    can drive the entity while the panel still shows the live values.

    Slider DOF names / limits are cached and refreshed whenever ``get_entity()`` returns
    a new entity (e.g. after an ``InteractiveScene`` rebuild).
    """

    def __init__(
        self,
        get_entity: Callable[[], Any | None],
        *,
        header: str = "DOF positions",
        id_suffix: str = "dofs",
    ) -> None:
        self._get_entity = get_entity
        self._header = header
        self._id_suffix = id_suffix
        self._dof_entity: Any | None = None
        self._dof_names: list[str] = []
        self._dof_lower = np.zeros(0, dtype=np.float64)
        self._dof_upper = np.zeros(0, dtype=np.float64)

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def dof_lower(self) -> np.ndarray:
        return self._dof_lower

    @property
    def dof_upper(self) -> np.ndarray:
        return self._dof_upper

    def sync(self, entity: Any) -> None:
        """Refresh cached DOF metadata when ``entity`` differs from the last call."""
        if entity is self._dof_entity:
            return
        self._dof_entity = entity
        self._dof_names, self._dof_lower, self._dof_upper = get_dof_info(entity)

    def current_targets(self) -> np.ndarray:
        """The entity's current DOF positions (env 0) as a ``(n_dofs,)`` numpy array.

        Useful for seeding a freshly-allocated targets array right after a robot switch.
        Returns an empty array if no entity is available yet.
        """
        entity = self._get_entity()
        if entity is None:
            return np.zeros(0, dtype=np.float64)
        self.sync(entity)
        current = np.asarray(tensor_to_array(entity.get_dofs_position()), dtype=np.float64).reshape(-1)
        return current[: len(self._dof_names)].copy()

    def render(self, imgui: Any, targets: np.ndarray, *, readonly: bool = False) -> bool:
        """Render the sliders inside the host panel; mutates ``targets`` in place.

        Returns ``True`` if any slider was moved this frame (or the Zero button was
        pressed). The "Print YAML pose" button always prints the current ``targets``
        to stdout in the ``hand_tactile_sandbox`` YAML format.
        """
        entity = self._get_entity()
        if entity is None:
            return False
        self.sync(entity)
        n = len(self._dof_names)
        if n == 0 or len(targets) != n:
            return False
        changed = False
        if imgui.collapsing_header(f"{self._header} ({n})##{self._id_suffix}_header"):
            for i, name in enumerate(self._dof_names):
                if readonly:
                    imgui.text(f"{name}: {float(targets[i]):.3f}")
                    continue
                dof_changed, new_val = imgui.slider_float(
                    f"{name}##{self._id_suffix}_dof_{i}",
                    float(targets[i]),
                    float(self._dof_lower[i]),
                    float(self._dof_upper[i]),
                    "%.3f",
                )
                if dof_changed:
                    targets[i] = new_val
                    changed = True
            if not readonly and imgui.button(f"Zero DOFs##{self._id_suffix}_zero"):
                targets[:] = np.clip(0.0, self._dof_lower, self._dof_upper)
                changed = True
            if not readonly:
                imgui.same_line()
            if imgui.button(f"Print YAML pose##{self._id_suffix}_print"):
                print(format_dofs_as_yaml_pose(self._dof_names, targets))
        return changed


class DofSequencePanel:
    """ImGui sub-panel: pause / play + YAML file loader for a cycling DOF target sequence.

    Owns the full cycling state:

    - ``paused`` (bool): toggled by a checkbox; the host script consults it to decide
      whether to follow the cycle or let the slider panel drive.
    - ``sequence`` ((n_steps, n_dofs) ndarray or ``None``): the loaded YAML sequence.
    - ``seq_t`` (float): cycle clock advanced by ``advance(dt)``; reset whenever a new
      sequence is loaded so the host doesn't have to track that bookkeeping.
    - ``interval`` (float): seconds each pose is held for; used by ``current_step_target``.

    The ``load_callback`` parses a YAML path and returns a fresh sequence array. File
    discovery scans each entry in ``search_dirs`` for ``glob`` matches; defaults pick
    up the project's ``conf/dofs_*.yaml`` files.
    """

    def __init__(
        self,
        load_callback: Callable[[str], np.ndarray | None],
        *,
        interval: float = 5.0,
        search_dirs: tuple = ("conf",),
        glob: str = "dofs_*.yaml",
        id_suffix: str = "seq",
    ) -> None:
        self._load_cb = load_callback
        self.interval = float(interval)
        self._search_dirs = tuple(search_dirs)
        self._glob = glob
        self._id_suffix = id_suffix
        self._files: list[tuple[str, str]] = []  # (label, abs_path)
        self._file_idx = 0
        self.paused: bool = False
        self.sequence: np.ndarray | None = None
        self.current_path: str | None = None
        self.seq_t: float = 0.0
        self._status = ""
        self._refresh_files()

    def _refresh_files(self) -> None:
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for d in self._search_dirs:
            base = Path(d)
            if not base.is_dir():
                continue
            for p in sorted(base.glob(self._glob)):
                resolved = str(p.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append((p.name, resolved))
        self._files = found
        # Keep the currently-loaded file selected when present, otherwise reset.
        if self.current_path is not None:
            for i, (_, path) in enumerate(self._files):
                if path == self.current_path:
                    self._file_idx = i
                    return
        self._file_idx = 0

    def set_sequence(self, path: str | None, seq: np.ndarray | None) -> None:
        """Replace the active sequence and reset the cycle clock to 0."""
        self.sequence = seq
        self.current_path = path
        self.seq_t = 0.0
        if path and seq is not None:
            self._status = f"Loaded {os.path.basename(path)} ({seq.shape[0]} poses)"

    def advance(self, dt: float) -> None:
        """Step the cycle clock by ``dt`` seconds if a sequence is loaded and not paused."""
        if not self.paused and self.sequence is not None:
            self.seq_t += float(dt)

    def current_pose_index(self) -> int:
        """Index of the pose currently being held (``0`` when no sequence is loaded)."""
        if self.sequence is None or self.sequence.shape[0] == 0:
            return 0
        return int(self.seq_t // self.interval) % int(self.sequence.shape[0])

    def current_step_target(self) -> np.ndarray | None:
        """Active pose target as a ``(n_dofs,)`` slice of ``sequence``, or ``None`` if unloaded."""
        if self.sequence is None or self.sequence.shape[0] == 0:
            return None
        return self.sequence[self.current_pose_index()]

    def render(self, imgui: Any) -> None:
        loaded_label = os.path.basename(self.current_path) if self.current_path else "<none>"
        imgui.text(f"YAML: {loaded_label}")
        if self.sequence is not None:
            n_poses = int(self.sequence.shape[0])
            imgui.text(
                f"Pose {self.current_pose_index()} / {n_poses}  "
                f"(cycle {'PAUSED' if self.paused else 'active'}, interval {self.interval:g}s)"
            )
        else:
            imgui.text("Poses: 0  (no sequence loaded -- sliders drive the hand)")
        _, self.paused = imgui.checkbox(f"Pause cycle##{self._id_suffix}_pause", self.paused)
        if self._files:
            _, self._file_idx = imgui.combo(
                f"YAML file##{self._id_suffix}_file", self._file_idx, [label for label, _ in self._files]
            )
            if imgui.button(f"Load##{self._id_suffix}_load"):
                path = self._files[self._file_idx][1]
                seq = self._load_cb(path)
                if seq is not None:
                    self.set_sequence(path, seq)
                else:
                    self._status = f"Failed to load {os.path.basename(path)}"
            imgui.same_line()
        else:
            imgui.text("No YAML files found.")
            imgui.same_line()
        if imgui.button(f"Rescan##{self._id_suffix}_rescan"):
            self._refresh_files()
            self._status = f"Found {len(self._files)} YAML file(s)."
        if self._status:
            imgui.text_colored((0.6, 0.9, 0.6, 1.0), self._status)
