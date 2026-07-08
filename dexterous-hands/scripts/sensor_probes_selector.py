import datetime
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import genesis as gs
import genesis.utils.geom as gu
import genesis.vis.keybindings as kb
import numpy as np
from genesis.engine.interactive_scene import InteractiveScene
from genesis.ext.pyrender.camera import OrthographicCamera
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin
from genesis.utils.geom import quat_to_xyz
from genesis.utils.misc import tensor_to_array
from genesis.utils.raycast import Ray
from genesis.vis.viewer_plugins import EVENT_HANDLE_STATE, EVENT_HANDLED, RaycasterViewerPlugin
from typing_extensions import override

if TYPE_CHECKING:
    from genesis.engine.entities.rigid_entity import RigidLink
    from genesis.engine.scene import Scene
    from genesis.ext.pyrender.node import Node
    from genesis.ext.pyrender.viewer import Viewer

from eden.utils.assets import get_asset_path
from imgui_panels import DofSliderPanel, format_camera_pose, render_camera_pose_button

import entities.robots  # noqa: F401
from registry import ROBOT_REGISTRY, get_argparser

ProbeLayoutKind = Literal["grid", "line"]

# Degrees the viewer camera orbits per frame while a rotation key (5/6, 7/8, 9/0) is held.
CAMERA_ROTATE_DROT_DEG = 1.0

# Cycled (by committed-layout index) so adjacent probe grids/lines are visually distinguishable.
PROBE_LAYOUT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    # (0.10, 0.30, 1.00, 1.0),
    (0.20, 0.50, 1.00, 1.0),
    (1.00, 0.45, 0.20, 1.0),
    (0.25, 0.80, 0.35, 1.0),
    (0.85, 0.30, 0.80, 1.0),
    (0.95, 0.80, 0.15, 1.0),
    (0.30, 0.85, 0.85, 1.0),
)

# Each probe is drawn as a low-opacity probe-radius sphere plus a tiny opaque marker at its
# center, so the exact centerpoint stays visible through the (overlapping) probe spheres.
PROBE_CENTER_RADIUS = 0.0008
PROBE_SPHERE_OPACITY = 0.3


def _gl_camera_pose_from_eye_target(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """OpenGL camera-to-world 4×4 pose (columns: right, up, +Z / backward)."""
    forward = target.astype(np.float64, copy=False) - eye.astype(np.float64, copy=False)
    dist = float(np.linalg.norm(forward))
    if dist < 1e-12:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / dist
    back = -forward
    temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, temp_up))) > 0.95:
        temp_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(temp_up, back)
    rn = float(np.linalg.norm(right))
    if rn < 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up = np.cross(back, right)
    un = float(np.linalg.norm(up))
    if un < 1e-12:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        up = up / un
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = back
    pose[:3, 3] = eye
    return pose


def _snap_viewer_camera_neg_world_axis(pyrender_viewer: "Viewer", axis: int) -> None:
    """
    Orbit center (trackball target) unchanged; place the camera on the +X/+Y/+Z side so the view direction is −X/−Y/−Z.
    ``pyrender_viewer`` must be ``genesis.vis.viewer.Viewer._pyrender_viewer`` (the pyglet/pyrender window).
    """
    tb = pyrender_viewer._trackball
    target = np.asarray(tb._n_target, dtype=np.float64)
    eye = np.asarray(tb._n_pose[:3, 3], dtype=np.float64)
    dist = float(np.linalg.norm(eye - target))
    if dist < 1e-6:
        dist = max(0.5 * float(pyrender_viewer.scene.scale), 0.35)
    offset = np.zeros(3, dtype=np.float64)
    offset[axis] = dist
    new_eye = target + offset
    pose = _gl_camera_pose_from_eye_target(new_eye, target)
    tb._n_pose = pose
    tb._pose = pose.copy()


def _make_axis_snap_callback(pyrender_viewer: "Viewer", axis: int):
    def _cb() -> None:
        _snap_viewer_camera_neg_world_axis(pyrender_viewer, axis)

    return _cb


def _axis_rotation_matrix(axis: int, angle: float) -> np.ndarray:
    """3×3 rotation matrix for a rotation of ``angle`` radians about world axis 0/1/2 (X/Y/Z)."""
    c, s = np.cos(angle), np.sin(angle)
    rot = np.eye(3, dtype=np.float64)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    rot[i, i] = c
    rot[j, j] = c
    rot[i, j] = -s
    rot[j, i] = s
    return rot


def _rotate_viewer_camera_world_axis(pyrender_viewer: "Viewer", axis: int, angle: float) -> None:
    """
    Orbit the camera about the world ``axis`` (0/1/2 = X/Y/Z) through the trackball target by ``angle`` radians.
    The orbit center and distance are preserved; the camera orientation is rotated rigidly so roll is kept.
    """
    tb = pyrender_viewer._trackball
    target = np.asarray(tb._n_target, dtype=np.float64)
    pose = np.asarray(tb._n_pose, dtype=np.float64).copy()
    rot = _axis_rotation_matrix(axis, angle)
    pose[:3, :3] = rot @ pose[:3, :3]
    pose[:3, 3] = target + rot @ (pose[:3, 3] - target)
    tb._n_pose = pose
    tb._pose = pose.copy()


def _make_axis_rotate_callback(pyrender_viewer: "Viewer", axis: int, angle: float):
    def _cb() -> None:
        _rotate_viewer_camera_world_axis(pyrender_viewer, axis, angle)

    return _cb


class SingleProbeAction(NamedTuple):
    entry: dict[str, Any]
    probe: dict[str, Any]
    debug_object: Any


class ProbeLayoutAction(NamedTuple):
    entry: dict[str, Any]
    debug_object: Any | None


class DebugPreview(NamedTuple):
    sphere: Any | None
    arrow: Any | None


class ProbePointsSelectorPlugin(RaycasterViewerPlugin):
    """
    Interactive viewer plugin: **left-click** adds a single probe on the mesh (no click-to-delete; use **U** undo).

    Grid/line selection samples on the **plane through the anchor** whose normal is the **camera view direction**
    at anchor time — parallel to the image plane:

    - **L** toggles right-click placement between grid and line.
    - **Right-click** on the mesh to enter placement mode (anchor + frozen u/v plane).
    - **Left or right click** commits the grid/line (preview cleared), then exits placement mode.
    - **←/→** / **↓/↑** adjust resolution along u / v for grids; either pair adjusts point count for lines.
    - **U** undoes the last single-point add or whole grid/line commit (including JSON session entries).

    On viewer close, writes a **JSON** file: top-level **list** of dicts with ``link_name``, ``kind`` (``"single"`` or
    ``"grid"`` or ``"line"``), and ``probes`` — flat list of probe dicts for singles/lines, or **2D** list (rows ×
    cols) for grids. Each probe has ``radius``, ``pos`` ``[x,y,z]``, ``normal`` ``[x,y,z]`` in **link-local**
    coordinates. Singles for the same link merge into the latest singles entry when the last entry for that link is
    still ``kind: single``.
    """

    _GRID_AXIS_MIN = 2
    _GRID_AXIS_MAX = 100

    def __init__(
        self,
        radius: float = 0.005,
        color: tuple = (0.2, 0.5, 1.0, 0.6),
        grid_snap: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        output_file: str = "selected_points.json",
        default_grid_num_points: int = 3,
        save_on_close: bool = True,
    ) -> None:
        super().__init__()
        self.radius = radius
        self.color = color
        # When False (GUI mode), probes are written only on an explicit save_probes() call.
        self.save_on_close = save_on_close
        # When set (via the ImGui panel), overrides the per-layout cycled color for every probe.
        self.probe_color_override: tuple[float, float, float, float] | None = None
        self.hover_color = (color[0], color[1], color[2], color[3] * 0.5)
        self.grid_snap = grid_snap
        self.output_file = output_file

        n = int(default_grid_num_points)
        self.grid_num_points: tuple[int, int] = (n, n)
        self._right_click_probe_layout: ProbeLayoutKind = "grid"
        self._grid_selection_mode = False
        self._grid_anchor_world: np.ndarray | None = None
        self._grid_anchor_link_name: str | None = None
        self._grid_plane_normal: np.ndarray | None = None  # unit; camera forward at anchor time
        self._grid_plane_axes: tuple[np.ndarray, np.ndarray] | None = None  # (u, v) in anchor plane
        self._grid_hover_corner_world: np.ndarray | None = None
        self._grid_preview_color = (1.0, 0.55, 0.15, 0.55)

        self._probe_entries: list[dict[str, Any]] = []
        self._layout_commit_count = 0
        self._selections_stack: list[SingleProbeAction | ProbeLayoutAction] = []
        # (entry, debug_objects) pairs for probes loaded from file — paired so they can be recolored.
        self._loaded_debug_objects: list[tuple[dict[str, Any], list[Any]]] = []
        self._prev_mouse_pos: tuple[int, int] = (0, 0)
        self._debug_preview: DebugPreview = DebugPreview(None, None)
        self._debug_layout_spheres: Any | None = None
        # True once build() has run; a second build() means the scene was rebuilt (e.g. robot switch).
        self._built_once = False

    def _log(self, message: str) -> None:
        print(f"[ProbePointsSelectorPlugin] {message}")

    def _draw_probe_spheres(
        self, positions: np.ndarray, color: tuple[float, float, float, float], radius: float | None = None
    ) -> list[Any]:
        """Draw probe-radius spheres at low opacity plus tiny opaque centerpoint markers.

        ``radius`` defaults to ``self.radius`` (the plugin-level radius used for newly placed probes);
        pass an explicit value to honor a probe's own stored radius. Returns the list of debug objects
        so callers can clear them together.
        """
        if radius is None:
            radius = self.radius
        if self.probe_color_override is not None:
            color = self.probe_color_override
        faded = (color[0], color[1], color[2], PROBE_SPHERE_OPACITY)
        center = (color[0], color[1], color[2], 1.0)
        return [
            self.scene.draw_debug_spheres(positions, radius, faded),
            self.scene.draw_debug_spheres(positions, PROBE_CENTER_RADIUS, center),
        ]

    def _clear_probe_debug(self, debug_object: Any) -> None:
        """Clear a probe debug object, which may be a list (probe sphere + centerpoint marker)."""
        if debug_object is None:
            return
        for obj in debug_object if isinstance(debug_object, list) else (debug_object,):
            if obj is not None:
                self.scene.clear_debug_object(obj)

    @staticmethod
    def _entry_flat_probes(entry: dict[str, Any]) -> list[dict[str, Any]]:
        """Flat list of probe dicts for an entry (grids store probes as rows)."""
        probes = entry.get("probes")
        if not isinstance(probes, list):
            return []
        if entry.get("kind") == "grid":
            return [p for row in probes if isinstance(row, list) for p in row if isinstance(p, dict)]
        return [p for p in probes if isinstance(p, dict)]

    def _build_link_by_name(self) -> dict[str, Any]:
        """Map link name -> RigidLink across all scene entities."""
        link_by_name: dict[str, Any] = {}
        for entity in self.scene.entities:
            for link in getattr(entity, "_links", None) or ():
                link_by_name[link.name] = link
        return link_by_name

    def _probes_world_by_radius(self, probes: list[dict[str, Any]], link: Any) -> dict[float, list[np.ndarray]]:
        """Group a link's drawable (radius > 0) probes into ``{radius: [world_pos, ...]}``.

        ``draw_debug_spheres`` bakes a single radius into one instanced mesh, so each distinct radius
        needs its own draw call. radius-0 probes are grid padding and are skipped.
        """
        link_pos = tensor_to_array(link.get_pos())
        link_quat = tensor_to_array(link.get_quat())
        by_radius: dict[float, list[np.ndarray]] = {}
        for probe in probes:
            pos = probe.get("pos")
            if not isinstance(pos, list) or len(pos) != 3:
                continue
            probe_radius = float(probe.get("radius", 0.0))
            if probe_radius <= 0.0:
                continue
            local_pos = np.asarray(pos, dtype=np.float64)
            world_pos = np.asarray(gu.transform_by_trans_quat(local_pos, link_pos, link_quat), dtype=np.float64)
            by_radius.setdefault(probe_radius, []).append(world_pos)
        return by_radius

    def _draw_probes_for_link(
        self, probes: list[dict[str, Any]], link: Any, color: tuple[float, float, float, float]
    ) -> list[Any]:
        """Draw every drawable probe of one link — one ``_draw_probe_spheres`` call per distinct radius."""
        debug_objects: list[Any] = []
        for probe_radius, positions in self._probes_world_by_radius(probes, link).items():
            debug_objects.extend(self._draw_probe_spheres(np.stack(positions, axis=0), color, radius=probe_radius))
        return debug_objects

    def _draw_entry_debug(
        self, entry: dict[str, Any], link_by_name: dict[str, Any], color: tuple[float, float, float, float]
    ) -> list[Any]:
        """Draw all debug spheres for one probe entry; returns a flat list of debug objects."""
        link = link_by_name.get(entry.get("link_name"))
        if link is None:
            return []
        return self._draw_probes_for_link(self._entry_flat_probes(entry), link, color)

    def set_probe_color(self, rgb: Any) -> None:
        """Apply a uniform color to every probe debug sphere: existing committed/loaded probes are
        redrawn, and the color also becomes the default for probes placed afterward. ``rgb`` is an
        ``(r, g, b)`` triple in ``[0, 1]``."""
        self.probe_color_override = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        link_by_name = self._build_link_by_name()

        recolored_stack: list[SingleProbeAction | ProbeLayoutAction] = []
        for action in self._selections_stack:
            self._clear_probe_debug(action.debug_object)
            link = link_by_name.get(action.entry.get("link_name"))
            if link is None:
                recolored_stack.append(action._replace(debug_object=None))
                continue
            if isinstance(action, SingleProbeAction):
                debug_object = self._draw_probes_for_link([action.probe], link, self.probe_color_override)
            else:
                debug_object = self._draw_entry_debug(action.entry, link_by_name, self.probe_color_override)
            recolored_stack.append(action._replace(debug_object=debug_object))
        self._selections_stack = recolored_stack

        recolored_loaded: list[tuple[dict[str, Any], list[Any]]] = []
        for entry, debug_object in self._loaded_debug_objects:
            self._clear_probe_debug(debug_object)
            recolored_loaded.append((entry, self._draw_entry_debug(entry, link_by_name, self.probe_color_override)))
        self._loaded_debug_objects = recolored_loaded

    def _reset_probe_state(self) -> None:
        """Drop all probe state. Used on scene rebuild — the old scene's debug objects are already
        gone, so this only clears the Python-side bookkeeping (no ``clear_debug_object`` calls)."""
        self._probe_entries.clear()
        self._selections_stack.clear()
        self._loaded_debug_objects.clear()
        self._layout_commit_count = 0

    def build(self, viewer, camera: "Node", scene: "Scene"):
        super().build(viewer, camera, scene)
        # A second build() means the scene was rebuilt (robot switch): the previous scene's probes
        # no longer apply, and their debug objects died with that scene.
        if self._built_once:
            self._reset_probe_state()
        self._built_once = True
        self._prev_mouse_pos = (self.viewer._viewport_size[0] // 2, self.viewer._viewport_size[1] // 2)

        grid_keybind_specs = (
            ("probe_grid_+row", kb.Key.RIGHT, 0, +1),
            ("probe_grid_-row", kb.Key.LEFT, 0, -1),
            ("probe_grid_+col", kb.Key.UP, 1, +1),
            ("probe_grid_-col", kb.Key.DOWN, 1, -1),
        )
        self.viewer.register_keybinds(
            *(
                kb.Keybind(
                    name,
                    key,
                    kb.KeyAction.PRESS,
                    callback=(lambda axis=axis, delta=delta: self._adjust_grid_resolution(axis, delta)),
                )
                for name, key, axis, delta in grid_keybind_specs
            ),
            kb.Keybind(
                "probe_undo",
                kb.Key.U,
                kb.KeyAction.PRESS,
                callback=self._keybind_undo,
            ),
            kb.Keybind(
                "probe_toggle_grid_line",
                kb.Key.L,
                kb.KeyAction.PRESS,
                callback=self._toggle_probe_layout_mode,
            ),
            kb.Keybind(
                "probe_clear_all",
                kb.Key.BACKSPACE,
                kb.KeyAction.PRESS,
                callback=self._keybind_clear_all,
            ),
        )

    def _probe_layout_message(self) -> str:
        if self._right_click_probe_layout == "line":
            return f"Right-click probe mode: line ({self.grid_num_points[0]} points)"
        u, v = self.grid_num_points
        return f"Right-click probe mode: grid ({u}x{v})"

    def _toggle_probe_layout_mode(self) -> None:
        self._right_click_probe_layout = "line" if self._right_click_probe_layout == "grid" else "grid"
        self.viewer.set_message_text(self._probe_layout_message())

    def _adjust_grid_resolution(self, axis: int, delta: int) -> None:
        if not self._grid_selection_mode:
            return
        if self._right_click_probe_layout == "line":
            n = self._clamp_grid_axis(self.grid_num_points[0] + delta)
            self.grid_num_points = (n, self.grid_num_points[1])
            self.viewer.set_message_text(f"Probe line size: {n}")
            return
        counts = list(self.grid_num_points)
        counts[axis] = self._clamp_grid_axis(counts[axis] + delta)
        self.grid_num_points = (counts[0], counts[1])
        u, v = self.grid_num_points
        self.viewer.set_message_text(f"Probe grid size: {u}x{v}")

    def _keybind_undo(self) -> None:
        if not self._selections_stack:
            self.viewer.set_message_text("Nothing to undo.")
            return
        action = self._selections_stack.pop()
        if isinstance(action, SingleProbeAction):
            self._undo_single_action(action)
        else:
            self._undo_probe_layout_action(action)
        self.viewer.set_message_text("Undo")

    def _clear_all_probes(self) -> int:
        """Clear every committed/loaded probe (debug objects + entries). Returns the entry count cleared."""
        for action in self._selections_stack:
            self._clear_probe_debug(action.debug_object)
        for _entry, debug_object in self._loaded_debug_objects:
            self._clear_probe_debug(debug_object)
        n_cleared = len(self._probe_entries)
        self._selections_stack.clear()
        self._loaded_debug_objects.clear()
        self._probe_entries.clear()
        self._layout_commit_count = 0
        return n_cleared

    def _keybind_clear_all(self) -> None:
        if not self._selections_stack and not self._loaded_debug_objects and not self._probe_entries:
            self.viewer.set_message_text("Nothing to clear.")
            return
        n_cleared = self._clear_all_probes()
        self.viewer.set_message_text(f"Cleared {n_cleared} probe entr{'y' if n_cleared == 1 else 'ies'}.")

    def _undo_single_action(self, action: SingleProbeAction) -> None:
        self._clear_probe_debug(action.debug_object)
        entry = action.entry
        probes = entry.get("probes")
        if not isinstance(probes, list):
            return
        for i in range(len(probes) - 1, -1, -1):
            if probes[i] is action.probe:
                probes.pop(i)
                break
        else:
            if action.probe in probes:
                probes.remove(action.probe)
        if not probes and entry in self._probe_entries:
            self._probe_entries.remove(entry)

    def _undo_probe_layout_action(self, action: ProbeLayoutAction) -> None:
        self._clear_probe_debug(action.debug_object)
        if action.entry in self._probe_entries:
            self._probe_entries.remove(action.entry)
        self._layout_commit_count = max(0, self._layout_commit_count - 1)

    def _next_layout_color(self) -> tuple[float, float, float, float]:
        """Color for the next committed grid/line, cycling ``PROBE_LAYOUT_COLORS`` by commit index."""
        color = PROBE_LAYOUT_COLORS[self._layout_commit_count % len(PROBE_LAYOUT_COLORS)]
        self._layout_commit_count += 1
        return color

    def _probe_dict_payload(self, local_pos: np.ndarray, local_normal: np.ndarray, radius: float) -> dict[str, Any]:
        def fmt(x: Any) -> float:
            return round(float(x), 5)

        return {
            "radius": fmt(radius),
            "pos": [fmt(p) for p in local_pos],
            "normal": [fmt(n) for n in local_normal],
        }

    @staticmethod
    def _zero_probe() -> dict[str, Any]:
        """Padding probe for grid cells whose ray missed geometry (radius 0 -> not drawn).

        The normal is a unit ``+Z`` rather than the natural ``(0,0,0)`` so that Genesis's
        ``probe_local_normal`` validator (which runs ``_normalize`` on every probe regardless
        of radius) doesn't reject the layout when it's loaded by a probe-based sensor.
        """
        return {"radius": 0.0, "pos": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0]}

    def _entry_for_output(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Serialize an entry. ``line`` entries become a degenerate ``N×1`` grid; grid cells that
        missed geometry (``None``) are written as a radius-0 zero probe instead of ``null``."""
        kind = entry.get("kind")
        probes = entry.get("probes")
        if kind == "line":
            rows = [[p] for p in probes if isinstance(p, dict)] if isinstance(probes, list) else []
            return {"link_name": entry.get("link_name"), "kind": "grid", "probes": rows}
        if kind == "grid" and isinstance(probes, list):
            rows = [
                [p if isinstance(p, dict) else self._zero_probe() for p in row] if isinstance(row, list) else row
                for row in probes
            ]
            return {"link_name": entry.get("link_name"), "kind": "grid", "probes": rows}
        return entry

    def _clear_grid_preview_debug(self) -> None:
        if self._debug_layout_spheres is not None:
            self.scene.clear_debug_object(self._debug_layout_spheres)
            self._debug_layout_spheres = None
        if self._debug_preview.sphere is not None:
            self.scene.clear_debug_object(self._debug_preview.sphere)
        if self._debug_preview.arrow is not None:
            self.scene.clear_debug_object(self._debug_preview.arrow)
        self._debug_preview = DebugPreview(None, None)

    def _append_single_probe_session(self, link_name: str, payload: dict[str, Any], debug_object: Any) -> None:
        ent = None
        for candidate in reversed(self._probe_entries):
            if candidate.get("link_name") == link_name and candidate.get("kind") == "single":
                ent = candidate
                break
        if ent is None:
            ent = {"link_name": link_name, "kind": "single", "probes": []}
            self._probe_entries.append(ent)

        ent["probes"].append(payload)
        self._selections_stack.append(SingleProbeAction(entry=ent, probe=payload, debug_object=debug_object))

    def _raycast_from_camera_to_plane_target(self, cam_pos: np.ndarray, target: np.ndarray) -> Any | None:
        # Perspective: rays diverge from cam_pos toward each target on the anchor plane.
        # Orthographic: rays are parallel along camera forward (== anchor-plane normal); shift the
        # origin laterally to ``target`` projected onto the plane through cam_pos perpendicular to
        # forward, so the ray passes through ``target`` exactly like the screen pixel would.
        if isinstance(self.camera.camera, OrthographicCamera) and self._grid_plane_normal is not None:
            forward = self._grid_plane_normal
            origin = target - float(np.dot(target - cam_pos, forward)) * forward
            direction = forward
        else:
            dvec = target - cam_pos
            ln = float(np.linalg.norm(dvec))
            if ln < 1e-12:
                return None
            origin = cam_pos
            direction = dvec / ln
        rh = self._raycaster.cast(*Ray(origin, direction))
        if rh is not None and rh.geom:
            # rh.position / rh.normal may alias the raycaster's shared result buffer, which is
            # overwritten by the next cast(). Snapshot them now so every stored hit keeps its own
            # values instead of all collapsing to the most recent cast.
            return rh._replace(
                position=np.array(rh.position, dtype=np.float64, copy=True),
                normal=np.array(rh.normal, dtype=np.float64, copy=True),
            )
        return None

    def _grid_plane_hits_matrix(self, corner_b_world: np.ndarray) -> list[list[Any | None]]:
        """Return ``nu×nv`` matrix of ray hits; ``None`` where the ray missed geometry."""
        if self._grid_anchor_world is None or self._grid_plane_axes is None:
            return []
        nu = self._clamp_grid_axis(self.grid_num_points[0])
        nv = self._clamp_grid_axis(self.grid_num_points[1])
        anchor = self._grid_anchor_world
        u, v = self._grid_plane_axes
        d_corner = self._project_point_on_grid_plane(corner_b_world) - anchor
        s1 = float(np.dot(d_corner, u))
        t1 = float(np.dot(d_corner, v))
        s_lo, s_hi = (0.0, s1) if 0.0 <= s1 else (s1, 0.0)
        t_lo, t_hi = (0.0, t1) if 0.0 <= t1 else (t1, 0.0)
        eps = 1e-9
        if s_hi - s_lo < eps:
            s_hi = s_lo + eps
        if t_hi - t_lo < eps:
            t_hi = t_lo + eps
        s_vals = np.linspace(s_lo, s_hi, nu)
        t_vals = np.linspace(t_lo, t_hi, nv)
        cam_pos = np.asarray(self.camera.matrix[:3, 3], dtype=np.float64)
        matrix: list[list[Any | None]] = []
        for sv in s_vals:
            row: list[Any | None] = []
            for tv in t_vals:
                target = anchor + float(sv) * u + float(tv) * v
                row.append(self._raycast_from_camera_to_plane_target(cam_pos, target))
            matrix.append(row)
        return matrix

    def _line_plane_hits(self, end_world: np.ndarray) -> list[Any | None]:
        """Return ordered ray hits along the anchor-to-end segment."""
        if self._grid_anchor_world is None:
            return []
        n = self._clamp_grid_axis(self.grid_num_points[0])
        anchor = self._grid_anchor_world
        end = self._project_point_on_grid_plane(end_world)
        cam_pos = np.asarray(self.camera.matrix[:3, 3], dtype=np.float64)
        hits: list[Any | None] = []
        for alpha in np.linspace(0.0, 1.0, n):
            target = anchor + float(alpha) * (end - anchor)
            hits.append(self._raycast_from_camera_to_plane_target(cam_pos, target))
        return hits

    def _layout_endpoint_at(self, x: int, y: int) -> np.ndarray | None:
        ray = self._screen_position_to_ray(x, y)
        rh = self._raycaster.cast(*ray)
        if rh is not None:
            return self._project_point_on_grid_plane(np.asarray(rh.position, dtype=np.float64))
        return self._grid_hover_corner_world

    def _commit_probe_layout_at(self, x: int, y: int) -> None:
        if self._right_click_probe_layout == "line":
            self._commit_line_at(x, y)
        else:
            self._commit_grid_at(x, y)

    def _commit_grid_at(self, x: int, y: int) -> None:
        corner_b = self._layout_endpoint_at(x, y)
        link_name = self._grid_anchor_link_name or ""
        if corner_b is None:
            return
        matrix = self._grid_plane_hits_matrix(corner_b)
        grid_world_positions: list[np.ndarray] = []
        grid_rows: list[list[dict[str, Any] | None]] = []
        for row in matrix:
            grid_row: list[dict[str, Any] | None] = []
            for rh in row:
                if rh is None:
                    grid_row.append(None)
                    continue
                _, world_pos, _, local_pos, local_normal = self._hit_to_probe_data(rh)
                grid_world_positions.append(world_pos)
                grid_row.append(self._probe_dict_payload(local_pos, local_normal, self.radius))
            grid_rows.append(grid_row)

        if not grid_world_positions:
            self.viewer.set_message_text("No grid probes hit geometry.")
            return

        grid_debug_object = self._draw_probe_spheres(np.stack(grid_world_positions, axis=0), self._next_layout_color())

        entry: dict[str, Any] = {"link_name": link_name, "kind": "grid", "probes": grid_rows}
        self._probe_entries.append(entry)
        self._selections_stack.append(ProbeLayoutAction(entry=entry, debug_object=grid_debug_object))
        self.viewer.set_message_text(f"Added probe grid: {len(grid_world_positions)} points")

    def _commit_line_at(self, x: int, y: int) -> None:
        end_world = self._layout_endpoint_at(x, y)
        link_name = self._grid_anchor_link_name or ""
        if end_world is None:
            return
        hits = self._line_plane_hits(end_world)
        line_world_positions: list[np.ndarray] = []
        line_probes: list[dict[str, Any]] = []
        for rh in hits:
            if rh is None:
                continue
            _, world_pos, _, local_pos, local_normal = self._hit_to_probe_data(rh)
            line_world_positions.append(world_pos)
            line_probes.append(self._probe_dict_payload(local_pos, local_normal, self.radius))

        if not line_world_positions:
            self.viewer.set_message_text("No line probes hit geometry.")
            return

        line_debug_object = self._draw_probe_spheres(np.stack(line_world_positions, axis=0), self._next_layout_color())
        entry: dict[str, Any] = {"link_name": link_name, "kind": "line", "probes": line_probes}
        self._probe_entries.append(entry)
        self._selections_stack.append(ProbeLayoutAction(entry=entry, debug_object=line_debug_object))
        self.viewer.set_message_text(f"Added probe line: {len(line_world_positions)} points")

    def _snap_to_grid(self, point: np.ndarray) -> np.ndarray:
        """
        Snap a point to the grid based on grid_snap settings.

        Parameters
        ----------
        point : np.ndarray, shape (3,)
            The point to snap.

        Returns
        -------
        np.ndarray, shape (3,)
            The point snapped to the grid.
        """
        grid_snap = np.array(self.grid_snap)
        # Snap each axis if the snap value is non-negative
        return np.where(grid_snap >= 0, np.round(point / grid_snap) * grid_snap, point)

    def _clamp_grid_axis(self, n: int) -> int:
        return max(self._GRID_AXIS_MIN, min(self._GRID_AXIS_MAX, int(n)))

    def _project_point_on_grid_plane(self, p: np.ndarray) -> np.ndarray:
        a = self._grid_anchor_world
        n = self._grid_plane_normal
        if a is None or n is None:
            return np.asarray(p, dtype=np.float64)
        return p.astype(np.float64, copy=False) - float(np.dot(p - a, n)) * n

    def _hit_to_probe_data(self, ray_hit: Any) -> tuple["RigidLink", np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        link = ray_hit.geom.link
        world_pos = np.asarray(ray_hit.position, dtype=np.float64)
        world_normal = np.asarray(ray_hit.normal, dtype=np.float64)
        link_pos = tensor_to_array(link.get_pos())
        link_quat = tensor_to_array(link.get_quat())
        local_pos = gu.inv_transform_by_trans_quat(world_pos, link_pos, link_quat)
        local_normal = gu.inv_transform_by_quat(world_normal, link_quat)
        local_pos = self._snap_to_grid(local_pos)
        return link, world_pos, world_normal, local_pos, local_normal

    def _exit_grid_mode(self, *, cancelled: bool) -> None:
        self._grid_selection_mode = False
        self._grid_anchor_world = None
        self._grid_anchor_link_name = None
        self._grid_plane_normal = None
        self._grid_plane_axes = None
        self._grid_hover_corner_world = None
        self._debug_layout_spheres = None

    @override
    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_pos = (x, y)

    @override
    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> EVENT_HANDLE_STATE:
        if self._grid_selection_mode:
            if button in (kb.MouseButton.LEFT, kb.MouseButton.RIGHT):
                self._clear_grid_preview_debug()
                self._commit_probe_layout_at(x, y)
                self._exit_grid_mode(cancelled=False)
                return EVENT_HANDLED

        if button == kb.MouseButton.RIGHT:
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(*ray)
            if ray_hit is not None and ray_hit.geom:
                self._grid_selection_mode = True
                self._grid_anchor_world = np.asarray(ray_hit.position, dtype=np.float64)
                self._grid_anchor_link_name = ray_hit.geom.link.name
                mtx = np.asarray(self.camera.matrix, dtype=np.float64)
                forward = -mtx[:3, 2]
                fwd_norm = float(np.linalg.norm(forward))
                if fwd_norm < 1e-12:
                    plane_n = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                else:
                    plane_n = forward / fwd_norm
                right = mtx[:3, 0]
                cam_up = mtx[:3, 1]
                u = right - float(np.dot(right, plane_n)) * plane_n
                u_norm = float(np.linalg.norm(u))
                if u_norm < 1e-8:
                    u = cam_up - float(np.dot(cam_up, plane_n)) * plane_n
                    u_norm = float(np.linalg.norm(u))
                if u_norm < 1e-8:
                    alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                    if abs(float(np.dot(alt, plane_n))) > 0.9:
                        alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    u = alt - float(np.dot(alt, plane_n)) * plane_n
                    u_norm = float(np.linalg.norm(u))
                u = u / u_norm
                v = np.cross(u, plane_n)
                v = v / (float(np.linalg.norm(v)) + 1e-12)
                self._grid_plane_normal = plane_n
                self._grid_plane_axes = (u.astype(np.float64), v.astype(np.float64))
                self.viewer.set_message_text(self._probe_layout_message())
                return EVENT_HANDLED
            return None

        if button == kb.MouseButton.LEFT:
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(*ray)
            if ray_hit is not None and ray_hit.geom:
                link, world_pos, _, local_pos, local_normal = self._hit_to_probe_data(ray_hit)
                debug_object = self._draw_probe_spheres(world_pos, self.color)
                payload = self._probe_dict_payload(local_pos, local_normal, self.radius)
                self._append_single_probe_session(link.name, payload, debug_object)
                return EVENT_HANDLED
        return None

    @override
    def on_mouse_scroll(self, x: int, y: int, dx: int, dy: int) -> EVENT_HANDLE_STATE:
        if self._grid_selection_mode or self._debug_preview.sphere is not None:
            self.radius = max(0.001, self.radius + dy * 0.001)
            return EVENT_HANDLED
        return None

    @override
    def on_mouse_drag(self, x: int, y: int, dx: int, dy: int, buttons: int, modifiers: int) -> EVENT_HANDLE_STATE:
        self._prev_mouse_pos = (x, y)
        if self._grid_selection_mode and (buttons & int(kb.MouseButton.RIGHT) or buttons & int(kb.MouseButton.LEFT)):
            return EVENT_HANDLED
        if buttons & int(kb.MouseButton.LEFT):
            ray = self._screen_position_to_ray(x, y)
            ray_hit = self._raycaster.cast(*ray)
            if ray_hit is not None:
                return EVENT_HANDLED
        return None

    @override
    def on_draw(self) -> None:
        super().on_draw()
        if self.scene._visualizer is None or not self.scene._visualizer.is_built:
            return

        if self._debug_preview.sphere is not None:
            self.scene.clear_debug_object(self._debug_preview.sphere)
        if self._debug_preview.arrow is not None:
            self.scene.clear_debug_object(self._debug_preview.arrow)
        self._debug_preview = DebugPreview(None, None)

        if self._debug_layout_spheres is not None:
            self.scene.clear_debug_object(self._debug_layout_spheres)
            self._debug_layout_spheres = None

        if self._grid_selection_mode and self._grid_anchor_world is not None:
            end_ray = self._screen_position_to_ray(*self._prev_mouse_pos)
            end_hit = self._raycaster.cast(*end_ray)
            if end_hit is not None:
                ep = np.asarray(end_hit.position, dtype=np.float64)
                self._grid_hover_corner_world = self._project_point_on_grid_plane(ep)
                snap_pos = self._snap_to_grid(self._grid_hover_corner_world.copy())
                self._debug_preview = DebugPreview(
                    self.scene.draw_debug_sphere(snap_pos, self.radius, self.hover_color),
                    self._debug_preview.arrow,
                )
                nrm = np.asarray(end_hit.normal, dtype=np.float64)
                self._debug_preview = DebugPreview(
                    self._debug_preview.sphere,
                    self.scene.draw_debug_arrow(
                        snap_pos,
                        tuple(float(n * 0.05) for n in nrm),
                        0.002,
                        self.hover_color,
                    ),
                )

            if self._grid_hover_corner_world is not None:
                flat_pos: list[np.ndarray] = []
                if self._right_click_probe_layout == "line":
                    hits = self._line_plane_hits(self._grid_hover_corner_world)
                    for rh in hits:
                        if rh is not None:
                            flat_pos.append(np.asarray(rh.position, dtype=np.float64))
                else:
                    matrix = self._grid_plane_hits_matrix(self._grid_hover_corner_world)
                    for row in matrix:
                        for rh in row:
                            if rh is not None:
                                flat_pos.append(np.asarray(rh.position, dtype=np.float64))
                if flat_pos:
                    poss = np.stack(flat_pos, axis=0)
                    self._debug_layout_spheres = self.scene.draw_debug_spheres(
                        poss, self.radius, self._grid_preview_color
                    )
            return

        mouse_ray = self._screen_position_to_ray(*self._prev_mouse_pos)
        closest_hit = self._raycaster.cast(*mouse_ray)
        if closest_hit is not None:
            snap_pos = self._snap_to_grid(np.asarray(closest_hit.position, dtype=np.float64))
            self._debug_preview = DebugPreview(
                self.scene.draw_debug_sphere(snap_pos, self.radius, self.hover_color),
                self._debug_preview.arrow,
            )
            nrm = np.asarray(closest_hit.normal, dtype=np.float64)
            self._debug_preview = DebugPreview(
                self._debug_preview.sphere,
                self.scene.draw_debug_arrow(
                    snap_pos,
                    tuple(float(n * 0.05) for n in nrm),
                    0.002,
                    self.hover_color,
                ),
            )

    def load_probes_from_file(self, file_path: str) -> None:
        """Load probes from a previously written JSON file and visualize them in world space.

        Loaded entries are appended to ``self._probe_entries`` so they will be re-written on close,
        but they are intentionally not pushed onto the undo stack — treat the loaded file as the
        starting state. Edit the JSON directly if you need to remove pre-existing probes.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except FileNotFoundError:
            self._log(f"File not found: {file_path}")
            return
        except json.JSONDecodeError as e:
            self._log(f"Failed to parse '{file_path}': {e}")
            return
        if not isinstance(entries, list):
            self._log(f"Invalid file format (expected top-level list): {file_path}")
            return

        link_by_name = self._build_link_by_name()

        n_loaded = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            link_name = entry.get("link_name")
            kind = entry.get("kind")
            probes = entry.get("probes")
            if not isinstance(probes, list) or kind not in ("single", "line", "grid"):
                continue
            if link_name not in link_by_name:
                self._log(f"Skipping entry with unknown link: {link_name!r}")
                continue

            # radius-0 probes are grid padding; they stay in the entry but are not drawn.
            color = self.probe_color_override or self._next_layout_color()
            debug_objects = self._draw_entry_debug(entry, link_by_name, color)
            if not debug_objects:
                continue
            self._loaded_debug_objects.append((entry, debug_objects))
            self._probe_entries.append(entry)
            n_loaded += sum(1 for p in self._entry_flat_probes(entry) if float(p.get("radius", 0.0)) > 0.0)

        self._log(f"Loaded {n_loaded} probe(s) from '{file_path}'.")

    def load_probes_replace(self, file_path: str) -> None:
        """Clear all current probes, load ``file_path`` fresh, and redirect future saves to it."""
        self._clear_all_probes()
        self.load_probes_from_file(file_path)
        self.output_file = file_path

    def save_probes(self) -> str | None:
        """Write all probe entries to ``self.output_file`` as JSON. Returns the path on success."""
        if not self._probe_entries:
            self._log("No probe entries to write.")
            return None

        selected_points: dict[int, dict[str, Any]] = {}
        for ent in self._probe_entries:
            probes = ent.get("probes")
            if not isinstance(probes, list):
                continue

            if ent.get("kind") == "grid":
                probe_iter = []
                for row in probes:
                    if not isinstance(row, list):
                        continue
                    probe_iter.extend(row)
            else:
                probe_iter = probes

            for probe in probe_iter:
                if isinstance(probe, dict) and isinstance(probe.get("pos"), list):
                    pos = probe["pos"]
                    pos_hash = hash((round(float(pos[0]), 6), round(float(pos[1]), 6), round(float(pos[2]), 6)))
                    selected_points[pos_hash] = probe

        output_entries = [self._entry_for_output(ent) for ent in self._probe_entries]
        output_file = self.output_file
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_entries, f, indent=2, ensure_ascii=False)
            self._log(f"Wrote {len(output_entries)} JSON record(s) to '{output_file}'.")
            self._log(f"Unique selected points: {len(selected_points)}")
            return output_file
        except Exception as e:
            self._log(f"Error writing to '{output_file}': {e}")
            return None

    @override
    def on_close(self) -> None:
        super().on_close()
        if self.save_on_close:
            self.save_probes()
        else:
            self._log("GUI mode: auto-save disabled — use the panel's Save button.")


class SelectorApp:
    """Shared control state between the ImGui panel (viewer thread) and the main loop (main thread)."""

    def __init__(self) -> None:
        self.running = True
        self.pending_robot: str | None = None

    def stop(self) -> None:
        self.running = False

    def request_robot_switch(self, robot_name: str) -> None:
        """Ask the main loop to rebuild the scene for a different robot (handled between steps)."""
        self.pending_robot = robot_name


def make_robot_entity_kwargs(robot_name: str) -> dict[str, dict[str, Any]]:
    """``entities_kwargs`` for ``InteractiveScene.rebuild`` loading ``robot_name`` as a fixed hand.

    The dict key becomes the entity ``name``, so the live entity can be matched back to its robot.
    """
    robot = ROBOT_REGISTRY.get(robot_name)()
    # Resolve the asset through Eden so it is fetched from HuggingFace when missing locally.
    robot_file = get_asset_path(
        file=robot.file,
        registry=robot.registry,
        dataset=robot.dataset,
        local_dir=robot.local_dir,
    )
    # MJCF morphs don't accept ``fixed`` / ``links_to_keep`` — the fixed-base and
    # link-merging behavior is part of the XML itself, so just drop those kwargs.
    morph_kwargs = dict(
        file=robot_file,
        collision=True,
        pos=(0.1, 0.1, 0.1),
        euler=quat_to_xyz(np.array(robot.default_root_quat)),
    )
    if robot_file.lower().endswith(".xml"):
        morph = gs.morphs.MJCF(**morph_kwargs)
    else:
        morph = gs.morphs.URDF(
            **morph_kwargs,
            fixed=True,
            links_to_keep=robot.links_to_keep,
        )
    entity_kwargs: dict[str, Any] = dict(morph=morph)
    if robot.surface is not None:
        entity_kwargs["surface"] = robot.surface
    return {robot_name: entity_kwargs}


def save_viewer_screenshot(pyrender_viewer: "Viewer", out_dir: str = ".") -> str:
    """Read the viewer's current color buffer and write it to a timestamped PNG.

    Must be called on the viewer thread (it touches the live OpenGL context). Returns the file path.
    """
    import cv2

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(out_dir, f"probe_viewer_{timestamp}.png"))
    data = pyrender_viewer._renderer.jit.read_color_buf(*pyrender_viewer._viewport_size, rgba=False)
    cv2.imwrite(path, np.flip(data, axis=-1))
    return path


def find_probe_files(robot_name: str) -> list[tuple[str, str]]:
    """Discover probe-layout JSON files as ``(label, absolute_path)``, deduplicated.

    Searches ``src/assets/sensors`` recursively and the current working directory (non-recursively);
    files for ``robot_name`` are listed first.
    """
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        resolved = str(path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        found.append((f"{path.parent.name}/{path.name}", resolved))

    sensors_dir = Path(__file__).resolve().parent.parent / "src" / "assets" / "sensors"
    if sensors_dir.is_dir():
        for path in sorted(sensors_dir.rglob("*.json")):
            _add(path)
    for path in sorted(Path.cwd().glob("*.json")):
        if "probe" in path.name.lower():
            _add(path)

    found.sort(key=lambda label_path: (robot_name not in label_path[1], label_path[0]))
    return found


class ProbeControlPanel:
    """ImGui side panel: switch the hand (in-place rebuild), load probe-layout files, recolor probes,
    pose the hand's DOFs, and take screenshots.

    The panel holds no scene-specific objects — it reads the live entity/viewer from the
    ``InteractiveScene`` each frame, so it transparently survives a robot-switch rebuild.
    """

    def __init__(
        self,
        selector: ProbePointsSelectorPlugin,
        interactive: InteractiveScene,
        app: SelectorApp,
        robot_names: list[str],
        current_robot: str,
    ) -> None:
        self._selector = selector
        self._interactive = interactive
        self._app = app
        self._robot_names = robot_names
        self._current_robot = current_robot
        self._robot_idx = robot_names.index(current_robot) if current_robot in robot_names else 0
        self._files = find_probe_files(current_robot)
        self._file_idx = 0
        self._probe_color = [0.2, 0.5, 1.0]
        self._status = ""
        # DOF slider sub-panel + per-entity target cache, rebuilt whenever the live
        # hand entity changes (e.g. after a robot-switch rebuild).
        self._dof_entity: Any | None = None
        self._dof_panel = DofSliderPanel(self._hand_entity, header="DOF positions", id_suffix="probe")
        self._dof_targets = np.zeros(0, dtype=np.float64)

    def _hand_entity(self) -> Any | None:
        """The articulated hand entity in the current scene (first entity with DOFs)."""
        try:
            entities = self._interactive.scene.entities
        except Exception:  # noqa: BLE001 -- scene may be momentarily absent mid-rebuild
            return None
        for entity in entities:
            if getattr(entity, "n_dofs", 0) > 0:
                return entity
        return None

    def _sync(self, entity: Any) -> None:
        """Refresh cached DOF/robot state when the live entity changes (after a robot-switch rebuild)."""
        if entity is self._dof_entity:
            return
        self._dof_entity = entity
        self._dof_panel.sync(entity)
        self._dof_targets = self._dof_panel.current_targets()

        robot = entity.name
        if robot != self._current_robot:
            self._current_robot = robot
            if robot in self._robot_names:
                self._robot_idx = self._robot_names.index(robot)
            self._files = find_probe_files(robot)
            self._file_idx = 0
            self._status = f"Switched to '{robot}'."

    def __call__(self, imgui: Any) -> None:
        imgui.separator()
        imgui.text("Probe Selector")

        entity = self._hand_entity()
        if entity is not None:
            self._sync(entity)

        # Explicit save — in GUI mode probes are never auto-saved on close.
        if imgui.button("Save probes##probe_save"):
            path = self._selector.save_probes()
            self._status = f"Saved {os.path.basename(path)}" if path else "Nothing to save."

        imgui.separator()

        # Robot switcher (rebuilds the scene in place — no relaunch).
        _, self._robot_idx = imgui.combo("Robot##probe_robot", self._robot_idx, self._robot_names)
        if imgui.button("Switch Robot##probe_switch_robot"):
            target = self._robot_names[self._robot_idx]
            if target == self._current_robot:
                self._status = f"Already using '{target}'."
            else:
                self._status = f"Switching to '{target}'..."
                self._app.request_robot_switch(target)

        imgui.separator()

        # Probe-layout file loader (replaces the current probes and saves back to the loaded file).
        if self._files:
            _, self._file_idx = imgui.combo(
                "Layout file##probe_file", self._file_idx, [label for label, _ in self._files]
            )
            if imgui.button("Load##probe_load"):
                path = self._files[self._file_idx][1]
                self._selector.load_probes_replace(path)
                self._status = f"Loaded {os.path.basename(path)}"
        else:
            imgui.text("No probe layout files found.")
        imgui.same_line()
        if imgui.button("Rescan##probe_rescan"):
            self._files = find_probe_files(self._current_robot)
            self._file_idx = 0
            self._status = f"Found {len(self._files)} layout file(s)."

        imgui.separator()

        # Uniform probe color.
        _, self._probe_color = imgui.color_edit3("Probe color##probe_color", self._probe_color)
        if imgui.button("Apply color##probe_apply_color"):
            self._selector.set_probe_color(self._probe_color)
            self._status = "Recolored all probes."

        imgui.separator()

        # DOF position controller — pose the hand kinematically to place probes in any configuration.
        if entity is not None and self._dof_panel.dof_names:
            if self._dof_panel.render(imgui, self._dof_targets):
                self._interactive.set_entity_dofs_position(entity, self._dof_targets)

        imgui.separator()

        if imgui.button("Screenshot##probe_screenshot"):
            try:
                path = save_viewer_screenshot(self._interactive.viewer._pyrender_viewer)
                self._status = f"Saved {os.path.basename(path)}"
            except Exception as exc:  # noqa: BLE001
                self._status = f"Screenshot failed: {exc}"
        imgui.same_line()
        render_camera_pose_button(imgui, self._interactive.viewer, id_suffix="probe")

        if self._status:
            imgui.text_colored((0.6, 0.9, 0.6, 1.0), self._status)


if __name__ == "__main__":
    parser = get_argparser(description="Select points on a mesh (to be used with a tactile sensor).")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to an existing probe JSON file to load. The file is also used as the output path so edits save in place.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Enable the ImGui control panel (switch robot, load layout files, recolor, screenshot, save). "
        "In GUI mode probes are saved only via the panel's Save button, not automatically on close.",
    )
    args = parser.parse_args()

    gs.init(backend=gs.gpu if not args.cpu else gs.cpu)

    # InteractiveScene lets a robot switch rebuild the scene in place (no relaunch), reattaching
    # plugins and restoring the camera pose automatically.
    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4469, 0.2764, 0.3353),
            camera_lookat=(-0.4023, -0.1438, 0.0151),
            camera_fov=40.0,
            enable_default_keybinds=True,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            background_color=(1.0, 1.0, 1.0),
        ),
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
        show_viewer=True,
    )

    interactive = InteractiveScene()
    interactive.rebuild(scene_kwargs=scene_kwargs, entities_kwargs=make_robot_entity_kwargs(args.robot))

    output_file = args.file if args.file is not None else f"sensor_probes_{args.robot}.json"
    selector = ProbePointsSelectorPlugin(
        radius=0.004,
        output_file=output_file,
        # In GUI mode, probes are saved only via the panel's Save button.
        save_on_close=not args.gui,
    )
    interactive.viewer.add_plugin(selector)

    app = SelectorApp()

    # ImGui control panel (opt-in via --gui). Added after the selector so it gets first dibs on input
    # events that land on the panel. Degrades gracefully (no panel) if imgui-bundle is not installed.
    if args.gui:
        try:
            imgui_plugin = ImGuiOverlayPlugin(show_sim_controls=False)
            interactive.viewer.add_plugin(imgui_plugin)
            imgui_plugin.register_panel(
                ProbeControlPanel(selector, interactive, app, sorted(ROBOT_REGISTRY.keys()), args.robot),
                section="side",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sensor_probes_selector] ImGui control panel unavailable: {exc}")

    if args.file is not None:
        selector.load_probes_from_file(args.file)

    def register_tool_keybinds() -> None:
        """(Re-)register the tool's camera + quit keybinds on the current viewer.

        Must run after every ``InteractiveScene.rebuild()`` — a rebuild creates a fresh viewer that
        only carries the default keybinds.
        """
        viewer_wrapper = interactive.viewer
        for keybind in (
            "record_video",
            "save_image",
            "camera_rotation",
            "shadow",
            "face_normals",
            "vertex_normals",
            "link_frame",
            "wireframe",
            "camera_frustum",
            "reload_shader",
            "fullscreen_mode",
        ):
            viewer_wrapper.remove_keybind(keybind)

        pyrender_viewer = viewer_wrapper._pyrender_viewer
        viewer_wrapper.register_keybinds(
            *(
                kb.Keybind(
                    f"camera_{axis}_axis",
                    key,
                    kb.KeyAction.PRESS,
                    callback=_make_axis_snap_callback(pyrender_viewer, i),
                )
                for i, (key, axis) in enumerate(((kb.Key._1, "x"), (kb.Key._2, "y"), (kb.Key._3, "z")))
            ),
            *(
                kb.Keybind(
                    f"camera_rotate_{axis}_{sign_name}",
                    key,
                    kb.KeyAction.HOLD,
                    callback=_make_axis_rotate_callback(pyrender_viewer, i, sign * np.deg2rad(CAMERA_ROTATE_DROT_DEG)),
                )
                for i, (axis, neg_key, pos_key) in enumerate(
                    (("x", kb.Key._5, kb.Key._6), ("y", kb.Key._7, kb.Key._8), ("z", kb.Key._9, kb.Key._0))
                )
                for sign, sign_name, key in ((-1.0, "neg", neg_key), (+1.0, "pos", pos_key))
            ),
            kb.Keybind(
                "print_camera_pose",
                kb.Key.P,
                kb.KeyAction.PRESS,
                callback=lambda: print(format_camera_pose(interactive.viewer)),
            ),
            kb.Keybind("quit", kb.Key.ESCAPE, kb.KeyAction.PRESS, callback=app.stop),
        )

    register_tool_keybinds()

    try:
        while interactive.viewer.is_alive() and app.running:
            if app.pending_robot is not None:
                target = app.pending_robot
                app.pending_robot = None
                if target in ROBOT_REGISTRY.keys() and target != interactive.scene.entities[0].name:
                    print(f"[sensor_probes_selector] Switching robot -> '{target}'...")
                    # rebuild() reattaches the selector/ImGui plugins; the selector drops its stale
                    # probe state in build(). Keybinds belong to the new viewer, so re-register them.
                    interactive.rebuild(entities_kwargs=make_robot_entity_kwargs(target))
                    register_tool_keybinds()
                    selector.output_file = f"sensor_probes_{target}.json"
            interactive.scene.step()
    except KeyboardInterrupt:
        print("Simulation interrupted, exiting.")
    finally:
        print("Simulation finished.")
