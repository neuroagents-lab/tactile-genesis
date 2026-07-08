# Tactile Genesis — Dexterous Hands

> **Dexterous Sensorized Hand Design through Task Learnability.** Reinforcement learning on
> dexterous robot hands with various tactile sensors in the Genesis simulator. The goal is to
> identify the sensor data type, fidelity, and placement that yield the largest task-performance
> gains when combined with robot learning — quantifying "task learnability" to guide hand design
> under spatial and monetary constraints.

This is the active project of the [Tactile Genesis monorepo](../README.md). It depends on the
sibling `Eden`, `Genesis`, and `rsl_rl` checkouts, resolved as local editable packages (see the
top-level README for how they are wired).

## Setup

```bash
# From this directory (dexterous-hands/), python 3.12
cp .env.example .env             # add your W&B API key (https://wandb.ai/settings#apikeys)
uv sync                          # installs eden / genesis-world / rsl-rl-lib from ../ (editable)
```

`uv sync` installs this project as an editable package, so `main.py` and the scripts can
import the project's own modules from `src/` directly — just run them with `uv run python ...`.

## Repository structure

```text
dexterous-hands/
├── main.py                 # Train / eval / deploy / optimize entrypoint
├── conf/
│   ├── experiments/        # Run configs (tiny.yaml, mid.yaml, ...)
│   ├── sample_grasps/      # Grasp-generation configs per task/robot
│   └── sensor/             # Tactile sensor parameter configs
├── scripts/                # Interactive viewers + tooling (see below)
├── src/
│   ├── tasks/              # Task configs + custom terms
│   │   ├── in_hand_repose/
│   │   ├── in_palm_rotate/
│   │   ├── rummage_hot/
│   │   └── screwdriver/
│   ├── entities/robots/    # Robot definitions (xhand1, sharpa)
│   ├── models/             # Policy/encoder model configs
│   ├── calibration/        # Sensor calibration
│   ├── assets/             # URDFs, meshes, grasps, objects, sensors
│   ├── registry.py         # Task/robot registry + argparser
│   └── ...                 # shared_terms, tactile_sensors, deploy, optimization, ...
└── pyproject.toml
```

## Tasks and robots

- **Tasks:** `in_hand_repose`, `in_palm_rotate` (default), `rummage_hot`, `screwdriver`
- **Robots:** `xhand1`, `sharpa`

## Run modes

`main.py --mode` accepts: `train` (default), `play`/`inference`, `deploy`, `optimize`,
`rollout-benchmark`.

```bash
# Local CPU smoke test (4 envs, 10 iters)
uv run python main.py \
  --task=in_hand_repose --robot=xhand1 --cpu --config=conf/experiments/tiny.yaml

# Visualize a trained policy
uv run python scripts/task_viewer.py \
  --task=in_hand_repose --robot=xhand1 --cpu --checkpoint <path>

# Slurm training
sbatch scripts/slurm/run.sh train --task=<task> --robot=<robot> --config=conf/experiments/<cfg>.yaml
```

## Tooling scripts

| Script | Purpose |
|---|---|
| `scripts/task_viewer.py` | Roll out / debug a policy in the viewer |
| `scripts/robot_dofs_viewer.py` | Interactively pose the robot / inspect DOFs |
| `scripts/grasps_generator.py` | Generate initial grasps/poses for a task (reads `conf/sample_grasps/<task>_<robot>.yaml`) |
| `scripts/sensors_viewer.py`, `scripts/sensor_probes_selector.py` | Visualize / place tactile sensors |
| `scripts/manual_calibration_gui.py` | Manual tactile calibration |
| `scripts/expand_distill_config.py`, `scripts/submit_distill.sh` | Teacher/student tactile distillation |

## Notes

- **W&B** is used for experiment tracking; the `tiny.yaml` config logs to TensorBoard instead so
  smoke tests need no W&B login.
- **Prefer Eden's built-in observation terms** (`eden.managers.terms.observations`, especially
  `proprio.py`) over custom ones — they already support a `history_length` parameter:

  ```python
  import eden as en

  dofs_pos = ObsTerm.configure(
      func=en.observations.dofs_pos,
      params={"entity_name": "robot"},
      noise=GaussianNoise.configure(std=0.005),
      history_length=3,
      flatten_history_dim=True,
  )
  ```
