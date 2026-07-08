# Genesis World

**Genesis World** (`genesis-world`) is a simulation platform for physical AI: a unified
multi-physics engine (Rigid, FEM, MPM, Particle PBD/SPH, coupling) behind a Pythonic interface,
with a cross-platform compiler ([Quadrants](https://github.com/Genesis-Embodied-AI/quadrants)) and
optional renderers. In this monorepo it is the **physics-simulation backend** that
[Eden](../Eden) and [`dexterous-hands`](../dexterous-hands) build on.

> This is a **vendored copy** for the Tactile Genesis monorepo. It is installed automatically
> (editable) when you `uv sync` from [`../dexterous-hands`](../dexterous-hands) — see the
> [monorepo README](../README.md). No separate install is needed.

## Full documentation

For the complete engine documentation, tutorials, examples, and API reference, see the upstream
project:

- Repo: <https://github.com/Genesis-Embodied-AI/Genesis>
- Docs: <https://genesis-world.readthedocs.io>

## Notes for this monorepo

- **Quaternions are wxyz** — the convention propagates through Eden and the tasks.
- Genesis is initialized via `eden.init(...)`; import `eden` before `genesis`/`torch`.
- CPU backend is available for local smoke tests (`--cpu` in the dexterous-hands entrypoints).

## License and Citation

Genesis is released under the Apache-2.0 License (see [`LICENSE`](LICENSE)). If you use it in your
research, please cite:

```bibtex
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```
