# 🍎 Eden

**Eden: Embodied-AI Development Environments Nexus** — a vectorized RL environment framework
built on [Genesis](../Genesis) and [rsl_rl](../rsl_rl). It provides declarative Pydantic configs
and pluggable terms composed through manager/term registries.

> This is an unofficial development copy used as the environment backend for the
> [`dexterous-hands`](../dexterous-hands) project in the Tactile Genesis monorepo.
> The official, complete Eden is at https://github.com/embodied-ai-nexus/Eden

## Installation

Eden is installed automatically (editable) when you `uv sync` from
[`../dexterous-hands`](../dexterous-hands) — see the [monorepo README](../README.md). No separate
install is needed for normal use.

## What's here

- `eden/envs/` — `EnvBase` / `RLEnvBase` and the rsl_rl env wrapper.
- `eden/managers/` — observation/reward/action/command/event/termination/curriculum/metric/recorder managers and their built-in terms.
- `eden/options/` — the Pydantic config hierarchy (`EdenConfig`, `EdenRLConfig`) and manager/robot/scene/material options.
- `eden/tasks/` — the task **registry + parser**; consumers register their own tasks.
- `eden/extensions/` — `sysid` (system identification), `deployment` (real-robot), `visualization` (viser).
- `eden/entities/`, `eden/utils/`, `eden/constants.py` — entities, helpers, enums.

## Key conventions

- **Import `eden` before `genesis` and `torch`** — `eden/__init__.py` isolates GPUs per rank before CUDA init (multi-GPU DDP).
- **Quaternions are wxyz.**
- Prefer built-in observation terms in `eden.managers.terms.observations` (they support `history_length`).
- `ruff` line length 120; use `pre-commit install` when developing.

## Acknowledgments

Built on [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) and
[quadrants](https://github.com/Genesis-Embodied-AI/quadrants); the manager-based environment
design draws on [IsaacLab](https://github.com/isaac-sim/IsaacLab),
[mjlab](https://github.com/mujocolab/mjlab), and [ManiSkill](https://github.com/haosulab/ManiSkill).
