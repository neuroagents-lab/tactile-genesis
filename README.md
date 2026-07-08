# Tactile Genesis

This repository reflects the self-contained code snapshot to reproduce the paper, [Tactile Genesis](https://arxiv.org/abs/2606.22332).
**Please note**, if you wish to use tactile sensors for your own project, use the latest [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world) physics engine which includes our tactile sensors,
and optionally, the official [Eden](https://github.com/embodied-ai-nexus/Eden) if you require a managed learning framework.

## Structure

| Directory | Package | Role |
|---|---|---|
| [`dexterous-hands/`](dexterous-hands/) | `tactile_genesis` | Tactile dexterous-hand RL tasks, robots, sensors, training/eval entrypoints. |
| [`Eden/`](Eden/) | `eden` | RL environment framework built on Genesis + rsl_rl. Declarative Pydantic configs, manager/term registries. |
| [`Genesis/`](Genesis/) | `genesis-world` | The physics-simulation backend (rigid/MPM/SPH/FEM/PBD). |
| [`rsl_rl/`](rsl_rl/) | `rsl-rl-lib` | PPO + student/teacher distillation training loop. |

Dependency topology: `dexterous-hands` → `Eden` → `Genesis`, with `rsl_rl` as the training loop.

## How the packages are wired

`dexterous-hands/pyproject.toml` resolves the three dependencies from the sibling
checkouts via `[tool.uv.sources]` (editable path installs). These sources override the
package origin across the whole resolution graph, so Eden's own transitive
`genesis-world` / `rsl-rl-lib` requirements also point at the local checkouts:

```toml
[tool.uv.sources]
eden          = { path = "../Eden",    editable = true }
genesis-world = { path = "../Genesis", editable = true }
rsl-rl-lib    = { path = "../rsl_rl",  editable = true }
```

## Quickstart

Everything is run from `dexterous-hands/`:

```bash
cd dexterous-hands
uv sync  # installs eden, genesis-world, rsl-rl-lib from local
uv run python main.py --task=in_hand_repose --robot=xhand1 --cpu
```

See [`dexterous-hands/README.md`](dexterous-hands/README.md) for the full task/robot list, run modes, and tooling scripts.

## Citation

If you use or reference our simulated tactile sensors in any way, please cite:

```bibtex
@article{chung2026tactilegenesis,
  title   = {Tactile Genesis: Exploring Tactile Sensors at Scale for Learning Dexterous Tasks},
  author  = {Chung, Trinity and Yamazaki, Kashu and Patel, Dhruv and Duburcq, Alexis and Qiao, Yiling and Fragkiadaki, Katerina and Nayebi, Aran},
  journal = {arXiv preprint arXiv:2606.22332},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.22332}
}
```