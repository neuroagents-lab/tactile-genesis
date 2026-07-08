"""Tactile sensor type registry.

Each tactile sensor type is described by a :class:`TactileSensorSpec`, shared by
``TactileSensorsMod`` (which builds ``SensorOptions``) and ``TactileSensorRead``
(which reads + postprocesses sensor output and converts xhand1 hardware data to
sim format). Adding a sensor type means adding one entry to ``TACTILE_SENSORS``
-- no ``if/else`` chains to touch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import genesis as gs
import torch
import yaml

# ============================ postprocess funcs ============================
# Called right after ``sensor.read()``; map raw sensor output to a flat
# ``(num_envs, N)`` float tensor. Signature: (data, *, sensor, num_envs, device).
# ``sensor`` is the Eden sensor wrapper (``sensor._sensor`` is the gs sensor).
#
# When the gs sensor is configured with ``history_length > 0``, each per-env
# reading carries an extra leading history dim: shape ``(num_envs, H, *S)``.
# ``TactileSensorsMod`` sets ``history_length=DECIMATION`` so the postprocess
# can median-filter the within-step substep readings before flattening.


TemporalReductionMode = Literal["median", "none", "last"]


def _history_length(sensor: Any) -> int:
    """Return the gs sensor's configured history length, or 0 if absent."""
    gs_opts = getattr(getattr(sensor, "_options", None), "sensor", None)
    return int(getattr(gs_opts, "history_length", 0) or 0)


def _apply_temporal_reduction(tensor: torch.Tensor, history_length: int, mode: TemporalReductionMode) -> torch.Tensor:
    """Reduce/preserve the history axis (axis 1) of a ``(N, H, *S)`` sensor tensor.

    ``"median"`` collapses via per-element median (the historical default; cast
    to float so bool sensors go through). ``"last"`` keeps only the final substep.
    ``"none"`` folds the history axis into the trailing feature axis so each probe
    carries ``H * F`` features instead of ``F``, while preserving the probe axis
    in place so grid-aware encoders' spatial reshape still works.

    No-op when the tensor doesn't actually carry a history axis matching
    ``history_length`` (e.g. a sensor configured with ``history_length=0``).
    """
    if history_length <= 0 or tensor.ndim < 3 or tensor.shape[1] != history_length:
        return tensor
    if mode == "median":
        return tensor.float().median(dim=1).values
    if mode == "last":
        return tensor[:, -1]
    # mode == "none": (N, H, *S) -> (N, *S[:-1], H * S[-1])
    moved = tensor.movedim(1, -1)
    return moved.reshape(*moved.shape[:-2], moved.shape[-2] * moved.shape[-1])


def postprocess_generic(
    data: Any, *, sensor: Any, num_envs: int, device: Any, temporal_reduction: TemporalReductionMode = "none"
) -> torch.Tensor:
    """Flatten one or more raw sensor tensors into a single ``(num_envs, N)`` tensor."""
    tensors = data if isinstance(data, tuple) else (data,)
    history_length = _history_length(sensor)
    out: list[torch.Tensor] = []
    for tensor in tensors:
        tensor = _apply_temporal_reduction(tensor, history_length, temporal_reduction)
        if tensor.ndim > 2:
            tensor = tensor.flatten(start_dim=1)
        out.append(tensor.float())
    return torch.cat(out, dim=-1)


def postprocess_force(
    data: Any, *, sensor: Any, num_envs: int, device: Any, temporal_reduction: TemporalReductionMode = "none"
) -> torch.Tensor:
    """KinematicTaxel reading -> flattened per-probe force vectors."""
    force = _apply_temporal_reduction(data.force, _history_length(sensor), temporal_reduction)
    return force.flatten(start_dim=1)


AGG_BOOL_TAXEL_COUNT_THRESHOLD: int = 2
"""Minimum true-taxel count for ``agg_bool`` to report contact (strict >).

0 means any single taxel triggers the link-level bit; raise this to filter out
spurious single-taxel contacts. Lives at module scope rather than on the
``TactileSensorSpec`` so it is trivially overridable without touching the
registry.
"""


def postprocess_agg_bool(
    data: Any, *, sensor: Any, num_envs: int, device: Any, temporal_reduction: TemporalReductionMode = "none"
) -> torch.Tensor:
    """ContactProbe taxel bools -> per-link aggregate bool ``(N, 1)`` / ``(N, H)``.

    Mirrors :func:`postprocess_agg_force`: aggregate first (per substep), reduce
    after. For each substep we count probes on the link reading ``True`` and
    emit ``1.0`` iff the count is strictly greater than
    :data:`AGG_BOOL_TAXEL_COUNT_THRESHOLD`. Then ``temporal_reduction`` reduces
    the substep axis: ``"median"`` -> ``(N, 1)``; ``"last"`` -> ``(N, 1)``;
    ``"none"`` -> ``(N, H)``.
    """
    bools = data.float()
    if bools.ndim == 1:
        bools = bools.reshape(1, -1)
        if num_envs > 1:
            bools = bools.expand(num_envs, -1)

    threshold = AGG_BOOL_TAXEL_COUNT_THRESHOLD
    history_length = _history_length(sensor)
    has_history = history_length > 0 and bools.ndim >= 3 and bools.shape[1] == history_length
    if has_history:
        # bools: (N, H, *probe_layout) -> (N, H, n_probes_flat); aggregate per substep.
        bools_flat = bools.reshape(bools.shape[0], history_length, -1)
        agg = (bools_flat.sum(dim=-1) > threshold).float().unsqueeze(-1)  # (N, H, 1)
        return _apply_temporal_reduction(agg, history_length, temporal_reduction)

    bools_flat = bools.reshape(bools.shape[0], -1)
    return (bools_flat.sum(dim=-1, keepdim=True) > threshold).float()


def postprocess_agg_force(
    data: Any, *, sensor: Any, num_envs: int, device: Any, temporal_reduction: TemporalReductionMode = "none"
) -> torch.Tensor:
    """ContactDepthProbe depths -> per-link XYZ force, ZYX-swapped and scaled.

    Mirrors the real XHand ``calc_pressure`` patch convention (ZYX axis order,
    ~1e4 scale) so sim and deploy observations are interchangeable.

    Substep aggregation order is: aggregate first (per substep), reduce after.
    For each substep we contract depth across the per-probe axis against the
    probe-local normals, yielding one ``[fx, fy, fz]`` per link per substep.
    Then ``temporal_reduction`` reduces that ``(N, H, 3)`` tensor:
    ``"median"`` -> ``(N, 3)``; ``"last"`` -> ``(N, 3)``; ``"none"`` -> ``(N, 3*H)``.
    The non-history path stays untouched.
    """
    depth = data.float()
    if depth.ndim == 1:
        depth = depth.reshape(1, -1)
        if num_envs > 1:
            depth = depth.expand(num_envs, -1)

    # ContactDepthProbe has no probe-normal field, so TactileSensorsMod carries the
    # parsed normals on SensorOptions (see _get_sensors_dict).
    probe_local_normal = getattr(sensor._options, "tactile_probe_local_normal", None)
    if probe_local_normal is None:
        raise RuntimeError("agg_force tactile sensors must carry tactile_probe_local_normal on SensorOptions.")
    normals_xyz = torch.as_tensor(probe_local_normal, device=depth.device, dtype=depth.dtype).reshape(-1, 3)

    history_length = _history_length(sensor)
    has_history = history_length > 0 and depth.ndim >= 3 and depth.shape[1] == history_length
    if has_history:
        # depth: (N, H, *probe_layout) -> (N, H, n_probes_flat); aggregate per substep.
        depth_flat = depth.reshape(depth.shape[0], history_length, -1)
        if normals_xyz.shape[0] != depth_flat.shape[-1]:
            raise RuntimeError(
                "agg_force probe normal count must match contact depth count: "
                f"{normals_xyz.shape[0]} normals vs {depth_flat.shape[-1]} depths."
            )
        force_xyz_per_step = (depth_flat[..., None] * normals_xyz[None, None, :, :]).sum(dim=2)
        # Sim probe configs use local XYZ; real XHand calc_pressure reports the same patch as ZYX.
        force_xyz_per_step = force_xyz_per_step[:, :, [2, 1, 0]] * 1e4  # (N, H, 3)
        # Reduce the substep axis exactly like the per-probe sensors.
        return _apply_temporal_reduction(force_xyz_per_step, history_length, temporal_reduction)

    if depth.ndim > 2:
        depth = depth.flatten(start_dim=1)
    if normals_xyz.shape[0] != depth.shape[-1]:
        raise RuntimeError(
            "agg_force probe normal count must match contact depth count: "
            f"{normals_xyz.shape[0]} normals vs {depth.shape[-1]} depths."
        )
    force_xyz = (depth[..., None] * normals_xyz[None, :, :]).sum(dim=1)
    return force_xyz[:, [2, 1, 0]] * 1e4


# ============================== xhand1 funcs ===============================
# Convert one fingertip's xhand1 hardware reading into the same sim format the
# matching postprocess func produces. The ``reading`` dict carries:
#   - ``calc_pressure``: aggregate fingertip force [fx, fy, fz], shape (3,)
#   - ``raw_pressure``: per-taxel raw force [fx, fy, fz], shape (n_taxels, 3)
#   - ``sensor_temperature``: scalar
# Signature: (reading, *, device, threshold) -> (1, N) tensor.


def _calc_pressure_tensor(reading: dict[str, Any], device: Any) -> torch.Tensor:
    """Extract the aggregate fingertip force [fx, fy, fz] as a ``(1, 3)`` tensor."""
    if "calc_pressure" not in reading:
        raise RuntimeError("xhand1 fingertip reading is missing 'calc_pressure'.")
    pressure = torch.as_tensor(reading["calc_pressure"], device=device, dtype=torch.float32).reshape(1, -1)
    if pressure.shape[-1] != 3:
        raise RuntimeError(f"Expected calc_pressure with 3 values [fx, fy, fz], got shape {tuple(pressure.shape)}.")
    return pressure


def xhand1_agg_force(reading: dict[str, Any], *, device: Any, threshold: float) -> torch.Tensor:
    """Real XHand calc_pressure is already in the sim agg_force convention."""
    return _calc_pressure_tensor(reading, device)


def xhand1_bool(reading: dict[str, Any], *, device: Any, threshold: float) -> torch.Tensor:
    """Thresholded contact-pressure magnitude -> per-finger boolean."""
    pressure = _calc_pressure_tensor(reading, device)
    magnitude = torch.linalg.norm(pressure, dim=-1, keepdim=True)
    return (magnitude > threshold).float()


# ============================== sensor spec ================================


@dataclass(frozen=True)
class TactileSensorSpec:
    """Describes one tactile sensor type end to end.

    Parameters
    ----------
    name : str
        Sensor type name, e.g. ``"agg_force"`` -- the key used in ``TACTILE_SENSORS``.
    placement : {"probes", "link"}
        ``"probes"`` for probe-config sensors, ``"link"`` for link-attached sensors.
    sensor_cls : type
        The ``gs.sensors`` class instantiated for this type.
    params : dict[str, Any]
        Base construction kwargs passed to ``sensor_cls``. Loaded per sensor
        type from ``conf/sensor/tactile_params.yaml``.
    needs_track_link : bool
        Point-cloud sensors require a tracked link to query geometry against.
    supports_2d : bool
        Whether a 2D (grid) probe layout is supported (probe sensors only).
    noise_params : dict[str, Any]
        Extra construction kwargs applied on top of ``params`` when the sensor
        type is requested with a trailing ``/noisy`` flag (see
        ``TactileSensorsMod``). Loaded per sensor type from
        ``conf/sensor/tactile_params.yaml``. Empty (the default) means no noise
        model, so ``/noisy`` is a no-op for that sensor type.
    features_per_probe : int
        Dim of the per-probe (or per-link, for ``link_*``/``agg_force``) feature
        vector produced by ``postprocess``. Used by ``derive_tactile_layout`` to
        size grid-aware encoders.
    postprocess : Callable
        Maps raw ``sensor.read()`` output to a flat ``(num_envs, N)`` tensor.
    xhand1_func : Callable | None
        Converts one fingertip's xhand1 hardware reading dict (``calc_pressure``,
        ``raw_pressure``, ``sensor_temperature``) to sim format; ``None`` if unsupported.

    The within-step substep reduction mode (median/none/last) is set on
    ``TactileSensorRead`` (and surfaced via the ``--temporal_reduction`` CLI arg
    on ``TactileSensorsMod``), not on the spec, since every postprocess function
    supports every mode.
    """

    name: str
    placement: Literal["probes", "link"]
    sensor_cls: type
    params: dict[str, Any]
    needs_track_link: bool = False
    noise_params: dict[str, Any] = field(default_factory=dict)
    features_per_probe: int = 1
    postprocess: Callable[..., torch.Tensor] = postprocess_generic
    xhand1_func: Callable[..., torch.Tensor] | None = None


# ============================= params loader ==============================
# Per-sensor-type construction kwargs live in
# ``conf/sensor/tactile_params.yaml`` (one block per sensor type, each with
# ``params`` and optional ``noise_params``). ``noise_params`` are layered on
# top of ``params`` when a sensor type is requested with a trailing ``/noisy``
# flag. Editing magnitudes is a YAML-only change -- no code edits needed.

_PARAMS_YAML_PATH = Path(__file__).resolve().parents[1] / "conf" / "sensor" / "tactile_params.yaml"


def _load_sensor_params() -> dict[str, dict[str, dict[str, Any]]]:
    """Load ``{sensor_type: {"params": {...}, "noise_params": {...}}}`` from yaml.

    Returns an empty mapping if the file is missing, so missing config simply
    turns the ``/noisy`` flag into a no-op and leaves base ``params`` empty
    rather than breaking config builds. Top-level keys starting with ``_`` are
    treated as anchor holders and skipped.
    """
    if not _PARAMS_YAML_PATH.is_file():
        return {}
    with open(_PARAMS_YAML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # dict(v) detaches each block (YAML aliases share one object across keys).
    return {
        str(k): {
            "params": dict(v.get("params") or {}),
            "noise_params": dict(v.get("noise_params") or {}),
        }
        for k, v in data.items()
        if isinstance(v, dict) and not str(k).startswith("_")
    }


_SENSOR_PARAMS: dict[str, dict[str, dict[str, Any]]] = _load_sensor_params()


def _params(name: str) -> dict[str, Any]:
    return dict(_SENSOR_PARAMS.get(name, {}).get("params", {}))


def _noise(name: str) -> dict[str, Any]:
    return dict(_SENSOR_PARAMS.get(name, {}).get("noise_params", {}))


TACTILE_SENSORS: dict[str, TactileSensorSpec] = {
    "link_bool": TactileSensorSpec(
        name="link_bool",
        placement="link",
        sensor_cls=gs.sensors.Contact,
        params=_params("link_bool"),
        noise_params=_noise("link_bool"),
        features_per_probe=1,
    ),
    "link_force": TactileSensorSpec(
        name="link_force",
        placement="link",
        sensor_cls=gs.sensors.ContactForce,
        params=_params("link_force"),
        noise_params=_noise("link_force"),
        features_per_probe=3,
    ),
    "bool": TactileSensorSpec(
        name="bool",
        placement="probes",
        sensor_cls=gs.sensors.ContactProbe,
        params=_params("bool"),
        noise_params=_noise("bool"),
        features_per_probe=1,
        xhand1_func=xhand1_bool,
    ),
    "depth": TactileSensorSpec(
        name="depth",
        placement="probes",
        sensor_cls=gs.sensors.ContactDepthProbe,
        params=_params("depth"),
        noise_params=_noise("depth"),
        features_per_probe=1,
    ),
    "agg_bool": TactileSensorSpec(
        name="agg_bool",
        placement="probes",
        sensor_cls=gs.sensors.ContactProbe,
        params=_params("agg_bool"),
        noise_params=_noise("agg_bool"),
        # postprocess collapses every probe into a single per-link bit; with
        # temporal_reduction='none' the H substep bits are kept and concatenated,
        # producing H features per link.
        features_per_probe=1,
        postprocess=postprocess_agg_bool,
        xhand1_func=xhand1_bool,
    ),
    "agg_force": TactileSensorSpec(
        name="agg_force",
        placement="probes",
        sensor_cls=gs.sensors.ContactDepthProbe,
        params=_params("agg_force"),
        noise_params=_noise("agg_force"),
        # postprocess collapses every probe into a single per-link [fx, fy, fz];
        # with temporal_reduction='none' the H substep [fx, fy, fz]s are kept and
        # concatenated, producing 3*H features per link.
        features_per_probe=3,
        postprocess=postprocess_agg_force,
        xhand1_func=xhand1_agg_force,
    ),
    "force": TactileSensorSpec(
        name="force",
        placement="probes",
        sensor_cls=gs.sensors.KinematicTaxel,
        params=_params("force"),
        noise_params=_noise("force"),
        features_per_probe=3,
        postprocess=postprocess_force,
    ),
    "force_torque": TactileSensorSpec(
        name="force_torque",
        placement="probes",
        sensor_cls=gs.sensors.KinematicTaxel,
        params=_params("force_torque"),
        noise_params=_noise("force_torque"),
        # Raw KinematicTaxel read flattens to [fx, fy, fz, tx, ty, tz] per probe.
        features_per_probe=6,
    ),
    "proximity": TactileSensorSpec(
        name="proximity",
        placement="probes",
        sensor_cls=gs.sensors.ProximityTaxel,
        params=_params("proximity"),
        needs_track_link=True,
        noise_params=_noise("proximity"),
        # ProximityTaxelData is (force, torque); postprocess_generic concats both
        # -> 6 features per probe, matching force_torque.
        features_per_probe=6,
    ),
    "elastomer": TactileSensorSpec(
        name="elastomer",
        placement="probes",
        sensor_cls=gs.sensors.ElastomerTaxel,
        params=_params("elastomer"),
        needs_track_link=True,
        noise_params=_noise("elastomer"),
        features_per_probe=3,
    ),
}


def spec_for_sensor_name(name: str) -> TactileSensorSpec | None:
    """Resolve the spec for a built sensor named ``tactile_<type>_<link>``.

    Sensor types can contain underscores (``agg_force``, ``force_torque``,
    ``link_bool``), so match by the longest ``tactile_<type>_`` prefix. Returns
    ``None`` for non-tactile sensors (e.g. ``priv_contact_*``).
    """
    best: tuple[str, TactileSensorSpec] | None = None
    for key, spec in TACTILE_SENSORS.items():
        if name.startswith(f"tactile_{key}_") and (best is None or len(key) > len(best[0])):
            best = (key, spec)
    return best[1] if best else None
