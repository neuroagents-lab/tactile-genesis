"""Minimal No-Sensor scaling repro for the 8k->16k throughput cliff.

The benchmark's "No Sensor" bar collapses at 16384 envs, so the regression is in the
core rigid step (Newton constraint solve / collider), not in any tactile sensor. This
script reproduces it with zero sensors so `git bisect` has a fast, sensor-independent
signal.

Run on the GPU box:
    python tests/scaling_repro.py
Exit code is 0 (PASS) when 16384 weak-scales within 2x of 8192, 1 (FAIL) on the cliff,
so it can drive `git bisect run python tests/scaling_repro.py`.
"""

import sys
import time

import numpy as np
import torch

import genesis as gs

WARMUP_STEPS = 30
RECORD_STEPS = 100
CLIFF_RATIO = 2.0  # FAIL if steps/s at 8192 is more than this multiple of steps/s at 16384.


def build_box_pyramid(n_envs):
    # Matches make_box_pyramid_with_sensors' physics scene (10 boxes on a plane), no sensors.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(dt=0.01, tolerance=1e-5),
        show_viewer=False,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box_size = 0.25
    spacing = (1.0 - 1e-3) * box_size
    offset = (-0.5, 1.0, 0.0) + 0.5 * np.array([box_size, box_size, box_size])
    boxes = []
    n_cubes = 4
    for i in range(n_cubes):
        for j in range(n_cubes - i):
            boxes.append(
                scene.add_entity(
                    gs.morphs.Box(size=[box_size] * 3, pos=offset + spacing * np.array([i + 0.5 * j, 0.0, j])),
                )
            )
    scene.build(n_envs=n_envs)
    for box in boxes:
        box.set_dofs_velocity(0.04 * torch.rand((n_envs, 6), dtype=gs.tc_float, device=gs.device))
    return scene


def measure(n_envs):
    scene = build_box_pyramid(n_envs)
    for _ in range(WARMUP_STEPS):
        scene.step()
    torch.cuda.synchronize()
    peak_before = torch.cuda.max_memory_reserved() / 1e9
    t = time.time()
    for _ in range(RECORD_STEPS):
        scene.step()
    torch.cuda.synchronize()
    dt = time.time() - t
    steps_per_s = RECORD_STEPS * n_envs / dt
    peak = torch.cuda.max_memory_reserved() / 1e9
    return steps_per_s, peak


def main():
    gs.init(backend=gs.gpu, logging_level="warning")
    results = {}
    for n_envs in (8192, 16384):
        torch.cuda.reset_peak_memory_stats()
        sps, peak = measure(n_envs)
        results[n_envs] = sps
        print(f"n_envs={n_envs:6d}  {sps/1e3:8.1f} k env-steps/s   peak_reserved={peak:6.1f} GB")
    ratio = results[8192] / max(results[16384], 1.0)
    print(f"ratio 8192/16384 = {ratio:.2f}  (FAIL if > {CLIFF_RATIO})")
    return 1 if ratio > CLIFF_RATIO else 0


if __name__ == "__main__":
    sys.exit(main())
