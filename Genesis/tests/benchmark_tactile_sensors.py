import math
import time
from typing import Any

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu

from .test_rigid_benchmarks import (
    STEP_DT,
    SceneMeta,
    get_rigid_solver_options,
    run_benchmark,
)

# Tactile benchmarks use shorter timing windows than the rigid benchmarks' 45s/15s
# defaults. The per-step tactile cost is stable once warmed, so 10s warmup + 10s record
# is plenty and keeps the full env sweep tractable. Passed to SceneMeta in the builders below.
TACTILE_DURATION_WARMUP = 10.0
TACTILE_DURATION_RECORD = 10.0


pytestmark = [
    pytest.mark.benchmarks,
    pytest.mark.cache(True),
    # Benchmark numbers are only meaningful in Genesis performance mode (static/FIELD array
    # backend, use_ndarray=False). The conftest skips these tests unless GS_ENABLE_NDARRAY=0
    # is also set in the environment (see babel/benchmark_sensors.sh), so a forgotten env var
    # fails loudly instead of silently recording slow dynamic-ndarray numbers.
    pytest.mark.performance_mode(True),
]


TACTILE_SENSOR_RUNNABLES = (
    "surface_distance_probe",
    "contact_depth_probe",
    "kinematic_taxel",
    "proximity_taxel",
    "elastomer_taxel",
)

PROBE_SENSOR_RUNNABLES = (
    "surface_distance_probe",
    "contact_depth_probe",
    "kinematic_taxel",
    "proximity_taxel",
    "elastomer_taxel",
)
POINTCLOUD_SENSOR_RUNNABLES = (
    "proximity_taxel",
    "elastomer_taxel",
)

N_ENVS_VARIANTS = (512, 1024, 2048, 4096, 8192, 16384)
N_SENSORS_VARIANTS = (1, 5)
PROBE_COUNTS = (10, 100, 1000, 10_000)
SAMPLE_POINT_COUNTS = (60, 600, 6000, 60_000, 600_000)

# TacSL (https://arxiv.org/pdf/2408.06506, Table II) reports force-field generation
# speed as a sweep over the number of parallel environments at two taxel grids:
# 10x10 (100 taxels) and 100x100 (10000 taxels). To overlay our kinematic taxel on the
# same axes (see scripts/plot/plot_tacsl_comparison.py and data/sensor_benchmarks/tacsl.csv),
# we fix n_probes at those two taxel counts and sweep batch_size over the same env counts
# TacSL reported. TacSL's table stops at 4096 envs for 100x100, but we push both
# resolutions to 32768 to show how far our taxel scales past where they ran out of memory.
#
# We also run a "no_sensors" pass (physics only, same scene) over the env sweep so the
# plot can separate tactile-computation cost from the underlying physics step. Resolution
# is meaningless without a sensor, so no_sensors runs once per env count with n_probes=0.
#
# TacSL's benchmark scene is minimal (their ball-rolling setup: one tactile sensor
# pressing a single object against a ground plane), so this comparison runs a matching
# box-on-sphere-on-plane scene with one sensor (n_sensors=1) rather than the multi-box
# pyramid the other tactile benchmarks use, to keep the per-step physics cost comparable
# and isolate the tactile computation. See make_box_on_sphere_with_sensors.
TACSL_COMPARISON_N_ENVS = (64, 256, 1024, 4096, 8192, 16384, 32768)
TACSL_COMPARISON_RESOLUTIONS = (100, 10_000)
# (runnable, n_probes) pairs for the parametrize grid below. The kinematic taxel runs at
# both taxel resolutions; no_sensors runs once (n_probes=0) as the physics baseline.
TACSL_COMPARISON_CASES = (
    ("no_sensors", 0),
    *(("kinematic_taxel", res) for res in TACSL_COMPARISON_RESOLUTIONS),
)

DEFAULT_N_SENSORS = 1
DEFAULT_N_PROBES = 100
DEFAULT_N_SAMPLE_POINTS = 600
DEFAULT_N_ENVS = 1024

# Sensor imperfections applied when running the noised variant. Each is filtered through
# ``_sensor_has_field`` so it is only set on sensors that expose it (see examples/sensors/tactile_sandbox.py):
NOISE_KWARGS = {
    "hysteresis_strength": 0.5,
    "hysteresis_tau": 0.1,
    "probe_radius_noise": 0.001,
    "probe_gain": 1.5,
    "probe_gain_resample_range": (0.8, 1.2),
    "dead_taxel_probability": 0.02,
    "dead_taxel_value_range": (0.0, 0.0),
    "noise": 0.001,
    "random_walk": 0.0001,
    "bias": 0.0005,
    "resolution": 0.0005,
    "jitter": 0.001,
    "crosstalk_strength": 0.3,
    "crosstalk_sigma": 0.01,
}


def _sensor_has_field(sensor_cls: type[Any], field_name: str) -> bool:
    return field_name in sensor_cls.model_fields


def _make_probe_kwargs(
    sensor_cls: type[Any],
    n_probes: int,
    half_size: float,
    probe_plane_z: float | None = None,
) -> dict[str, Any]:
    nx = math.ceil(math.sqrt(n_probes))
    ny = math.ceil(n_probes / nx)
    n_total = nx * ny
    n_filler = n_total - n_probes

    # The probe grid spans ±half_size in x/y at local z=plane_z, with the normal facing -z.
    # Defaults to half_size (the cube-face convention) when probe_plane_z is not given.
    plane_z = half_size if probe_plane_z is None else probe_plane_z
    grid = gu.generate_grid_points_on_plane(
        lo=[-half_size, -half_size, plane_z],
        hi=[half_size, half_size, plane_z],
        normal=(0.0, 0.0, -1.0),
        nx=nx,
        ny=ny,
    )

    # KinematicTaxel and ElastomerTaxel accept probe_radius=0 filler entries, so they always take the 2D
    # (ny, nx, 3) grid and pad any leftover cells. ProximityTaxel cannot pad, so when n_probes does not tile a
    # grid exactly it falls back to a flat (N, 3) layout trimmed to n_probes. (ProximityTaxel/KinematicTaxel/
    # ElastomerTaxel all need the 2D grid for their grid-FFT/spatial-crosstalk paths; the rest only ever use flat.)
    supports_filler = sensor_cls in (gs.sensors.KinematicTaxel, gs.sensors.ElastomerTaxel)
    if n_filler > 0 and not supports_filler:
        return {"probe_local_pos": grid.reshape(-1, 3)[:n_probes]}

    grid_classes = (gs.sensors.KinematicTaxel, gs.sensors.ElastomerTaxel, gs.sensors.ProximityTaxel)
    keep_grid = sensor_cls in grid_classes
    probe_local_pos = grid if keep_grid else grid.reshape(-1, 3)
    kwargs: dict[str, Any] = {"probe_local_pos": probe_local_pos}
    if n_filler > 0:
        # probe_radius is validated by element count and flattened row-major internally, so a flat (n_total,)
        # array lines up with the row-major grid regardless of whether probe_local_pos is 2D or flat.
        probe_radius = np.full(n_total, 0.01, dtype=gs.np_float)
        probe_radius[-n_filler:] = 0.0
        kwargs["probe_radius"] = probe_radius
    return kwargs


def _make_tactile_sensor_options(
    sensor_cls: type[Any],
    *,
    box: Any,
    track_box: Any,
    n_probes: int,
    n_sample_points: int,
    half_size: float,
    grid_size: tuple[int, int, int] | None = None,
    noise: bool = False,
    probe_plane_z: float | None = None,
):
    sensor_kwargs = {"entity_idx": box.idx}

    if _sensor_has_field(sensor_cls, "track_link_idx"):
        sensor_kwargs["track_link_idx"] = (track_box.base_link_idx,)

    if _sensor_has_field(sensor_cls, "probe_local_pos"):
        sensor_kwargs.update(_make_probe_kwargs(sensor_cls, n_probes, half_size, probe_plane_z=probe_plane_z))

    if _sensor_has_field(sensor_cls, "n_sample_points"):
        sensor_kwargs["n_sample_points"] = n_sample_points

    if _sensor_has_field(sensor_cls, "properties_dict"):
        sensor_kwargs["properties_dict"] = {-1: gs.sensors.TemperatureProperties()}

    if _sensor_has_field(sensor_cls, "grid_size") and grid_size is not None:
        sensor_kwargs["grid_size"] = grid_size

    if noise:
        for field, value in NOISE_KWARGS.items():
            if _sensor_has_field(sensor_cls, field):
                sensor_kwargs[field] = value

    return sensor_cls(**sensor_kwargs)


def make_box_pyramid_with_sensors(
    n_envs,
    sensor_cls,
    n_sensors=DEFAULT_N_SENSORS,
    n_probes=DEFAULT_N_PROBES,
    n_sample_points=DEFAULT_N_SAMPLE_POINTS,
    n_cubes=4,
    grid_size=None,
    noise=False,
    **scene_kwargs,
):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                tolerance=1e-5,
            )
        ),
        **{
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(0.0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            "show_viewer": False,
            "show_FPS": False,
            **scene_kwargs,
        },
    )

    scene.add_entity(gs.morphs.Plane())

    box_size = 0.25
    box_spacing = (1.0 - 1e-3) * box_size
    box_pos_offset = (-0.5, 1.0, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    boxes = []
    for i in range(n_cubes):
        for j in range(n_cubes - i):
            box = scene.add_entity(
                gs.morphs.Box(
                    size=[box_size, box_size, box_size],
                    pos=box_pos_offset + box_spacing * np.array([i + 0.5 * j, 0.0, j]),
                ),
            )
            boxes.append(box)

    half_size = box_size / 2.0
    for sensor_idx in range(n_sensors):
        box = boxes[sensor_idx % len(boxes)]
        track_box = boxes[(sensor_idx + 1) % len(boxes)]
        scene.add_sensor(
            _make_tactile_sensor_options(
                sensor_cls,
                box=box,
                track_box=track_box,
                n_probes=n_probes,
                n_sample_points=n_sample_points,
                half_size=half_size,
                grid_size=grid_size,
                noise=noise,
            )
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        for box in boxes:
            box.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            duration_warmup=TACTILE_DURATION_WARMUP,
            duration_record=TACTILE_DURATION_RECORD,
        ),
    )


def make_box_on_sphere_with_sensors(
    n_envs,
    sensor_cls,
    n_sensors=DEFAULT_N_SENSORS,
    n_probes=DEFAULT_N_PROBES,
    n_sample_points=DEFAULT_N_SAMPLE_POINTS,
    noise=False,
    **scene_kwargs,
):
    """TacSL-style minimal contact scene: a flat box resting on a sphere on the ground plane.

    The sensor lives on the box's bottom face (the box-sphere contact patch), mirroring a flat
    tactile pad pressing a curved object. ``n_sensors=0`` yields the physics-only baseline.
    """
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            **get_rigid_solver_options(
                dt=STEP_DT,
                tolerance=1e-5,
            )
        ),
        **{
            "viewer_options": gs.options.ViewerOptions(
                camera_pos=(0.0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            "show_viewer": False,
            "show_FPS": False,
            **scene_kwargs,
        },
    )

    scene.add_entity(gs.morphs.Plane())

    sphere_radius = 0.2
    box_size = (0.5, 0.5, 0.1)
    box_half_height = box_size[2] / 2.0

    scene.add_entity(
        gs.morphs.Sphere(radius=sphere_radius, pos=(0.0, 0.0, sphere_radius)),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=list(box_size), pos=(0.0, 0.0, 2.0 * sphere_radius + box_half_height)),
    )

    # Probes tile the box's bottom (0.5 x 0.5) face, which is the surface contacting the sphere.
    probe_half_extent = box_size[0] / 2.0
    for _ in range(n_sensors):
        scene.add_sensor(
            _make_tactile_sensor_options(
                sensor_cls,
                box=box,
                track_box=box,
                n_probes=n_probes,
                n_sample_points=n_sample_points,
                half_size=probe_half_extent,
                probe_plane_z=-box_half_height,
                noise=noise,
            )
        )

    time_start = time.time()
    scene.build(n_envs=n_envs)
    compile_time = time.time() - time_start

    if n_envs > 0:
        box.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))

    def step():
        scene.step()

    return (
        scene,
        step,
        SceneMeta(
            compile_time=compile_time,
            duration_warmup=TACTILE_DURATION_WARMUP,
            duration_record=TACTILE_DURATION_RECORD,
        ),
    )


def _run_tactile_sensor_benchmark(
    n_envs, n_sensors, n_probes, n_sample_points, sensor_cls, noise=False, grid_size=None
):
    _, step_fn, meta = make_box_pyramid_with_sensors(
        n_envs,
        sensor_cls,
        n_sensors=n_sensors,
        n_probes=n_probes,
        n_sample_points=n_sample_points,
        grid_size=grid_size,
        noise=noise,
    )
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


def _run_box_on_sphere_benchmark(n_envs, n_sensors, n_probes, sensor_cls, noise=False):
    _, step_fn, meta = make_box_on_sphere_with_sensors(
        n_envs,
        sensor_cls,
        n_sensors=n_sensors,
        n_probes=n_probes,
        n_sample_points=DEFAULT_N_SAMPLE_POINTS,
        noise=noise,
    )
    return run_benchmark(step_fn, n_envs=n_envs, meta=meta)


# Maps the runnable name logged for each sensor to its sensor class, so benchmarks that
# need a non-default scene (e.g. the TacSL comparison's box-on-sphere) can build directly
# without routing through the pyramid fixtures used by the other tests.
_SENSOR_CLS_BY_RUNNABLE = {
    "surface_distance_probe": gs.sensors.SurfaceDistanceProbe,
    "contact_depth_probe": gs.sensors.ContactDepthProbe,
    "kinematic_taxel": gs.sensors.KinematicTaxel,
    "elastomer_taxel": gs.sensors.ElastomerTaxel,
    "proximity_taxel": gs.sensors.ProximityTaxel,
}


@pytest.fixture
def n_sensors():
    return DEFAULT_N_SENSORS


@pytest.fixture
def n_probes():
    return DEFAULT_N_PROBES


@pytest.fixture
def n_sample_points():
    return DEFAULT_N_SAMPLE_POINTS


@pytest.fixture
def noise(request):
    # Default off; the noised tests flip this to True via ``indirect=["noise"]`` parametrize.
    return getattr(request, "param", False)


@pytest.fixture
def no_sensors(n_envs):
    return _run_tactile_sensor_benchmark(n_envs, 0, 0, 0, None)


@pytest.fixture
def surface_distance_probe(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.SurfaceDistanceProbe, noise=noise
    )


@pytest.fixture
def contact_depth_probe(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ContactDepthProbe, noise=noise
    )


@pytest.fixture
def kinematic_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.KinematicTaxel, noise=noise
    )


@pytest.fixture
def elastomer_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ElastomerTaxel, noise=noise
    )


@pytest.fixture
def proximity_taxel(n_envs, n_sensors, n_probes, n_sample_points, noise):
    return _run_tactile_sensor_benchmark(
        n_envs, n_sensors, n_probes, n_sample_points, gs.sensors.ProximityTaxel, noise=noise
    )


@pytest.fixture
def grid_size(request):
    return getattr(request, "param", (1, 1, 1))


@pytest.fixture
def temperature_grid(n_envs, n_sensors, n_probes, n_sample_points, noise, grid_size):
    return _run_tactile_sensor_benchmark(
        n_envs,
        n_sensors,
        n_probes,
        n_sample_points,
        gs.sensors.TemperatureGrid,
        noise=noise,
        grid_size=grid_size,
    )


# ---------------------------------------------------------------------------
# Parametrized benchmark test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runnable, n_envs, backend",
    [("no_sensors", n_envs, gs.gpu) for n_envs in N_ENVS_VARIANTS],
)
def test_scene_speed(factory_logger, request, runnable, n_envs):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": 0,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, backend",
    [
        (runnable, n_envs, n_sensors, gs.gpu)
        for runnable in TACTILE_SENSOR_RUNNABLES
        for n_envs in N_ENVS_VARIANTS
        for n_sensors in N_SENSORS_VARIANTS
    ],
)
def test_tactile_sensor_speed(factory_logger, request, runnable, n_envs, n_sensors):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, grid_size, backend",
    [
        ("temperature_grid", n_envs, n_sensors, grid_size, gs.gpu)
        for n_envs in N_ENVS_VARIANTS
        for n_sensors in N_SENSORS_VARIANTS
        for grid_size in ((1, 1, 1), (2, 2, 2))
    ],
    indirect=["grid_size"],
)
def test_temperature_grid_speed(factory_logger, request, runnable, n_envs, n_sensors, grid_size):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "grid_size": grid_size,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, noise, backend",
    [
        (runnable, n_envs, n_sensors, True, gs.gpu)
        for runnable in TACTILE_SENSOR_RUNNABLES
        for n_envs in N_ENVS_VARIANTS
        for n_sensors in N_SENSORS_VARIANTS
    ],
    indirect=["noise"],
)
def test_noised_tactile_sensor_speed(factory_logger, request, runnable, n_envs, n_sensors, noise):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "noise": True,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_probes, backend",
    [(runnable, DEFAULT_N_ENVS, n_probes, gs.gpu) for runnable in PROBE_SENSOR_RUNNABLES for n_probes in PROBE_COUNTS],
)
def test_probe_sensor_speed_per_num_probe(factory_logger, request, runnable, n_envs, n_sensors, n_probes):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_probes, backend",
    [
        (runnable, n_envs, n_probes, gs.gpu)
        for runnable, n_probes in TACSL_COMPARISON_CASES
        for n_envs in TACSL_COMPARISON_N_ENVS
    ],
)
def test_tacsl_comparison_speed(factory_logger, request, runnable, n_envs, n_sensors, n_probes):
    # Box-on-sphere-on-plane with one sensor, matching TacSL's minimal ball-rolling scene
    # (see TACSL_COMPARISON_* above) rather than the shared box-pyramid fixtures. The
    # no_sensors case is the physics-only baseline (no sensor instantiated).
    if runnable == "no_sensors":
        n_sensors = 0
        result = _run_box_on_sphere_benchmark(n_envs, 0, 0, None)
    else:
        result = _run_box_on_sphere_benchmark(n_envs, n_sensors, n_probes, _SENSOR_CLS_BY_RUNNABLE[runnable])
    with factory_logger(
        {
            "env": "box_on_sphere_with_sensor",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(result)


@pytest.mark.parametrize(
    "runnable, n_envs, n_sensors, n_probes, backend",
    [("contact_depth_probe", n_envs, 5, 1024, gs.gpu) for n_envs in N_ENVS_VARIANTS],
)
def test_tacmap_comparsion(factory_logger, request, runnable, n_envs, n_sensors, n_probes):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))


@pytest.mark.parametrize(
    "runnable, n_envs, n_sample_points, backend",
    [
        (runnable, DEFAULT_N_ENVS, n_sample_points, gs.gpu)
        for runnable in POINTCLOUD_SENSOR_RUNNABLES
        for n_sample_points in SAMPLE_POINT_COUNTS
    ],
)
def test_pointcloud_sensor_speed_per_num_samples(
    factory_logger,
    request,
    runnable,
    n_envs,
    n_sensors,
    n_probes,
    n_sample_points,
):
    with factory_logger(
        {
            "env": "box_pyramid_with_sensors",
            "sensor": runnable,
            "batch_size": n_envs,
            "n_sensors": n_sensors,
            "n_probes": n_probes,
            "n_sample_points": n_sample_points,
            "use_contact_island": False,
        }
    ) as logger:
        logger.write(request.getfixturevalue(runnable))
