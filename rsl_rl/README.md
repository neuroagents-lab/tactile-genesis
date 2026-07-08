# RSL-RL

**RSL-RL** (`rsl-rl-lib`) is a GPU-accelerated, lightweight learning library for robotics
research. In this monorepo it provides the PPO and student/teacher distillation training loop
used by [`dexterous-hands`](../dexterous-hands) through [Eden](../Eden)'s rsl_rl env wrapper.

> This is a **vendored copy** for the Tactile Genesis monorepo. It is installed automatically
> (editable) when you `uv sync` from [`../dexterous-hands`](../dexterous-hands) — see the
> [monorepo README](../README.md). No separate install is needed.

## Key features

- Minimal, readable codebase with clear extension points.
- Robotics-first methods: PPO and Student-Teacher Distillation.
- Native multi-GPU training support.

Upstream project: <https://github.com/leggedrobotics/rsl_rl>

## Citation

If you use RSL-RL in your research, please cite the [paper](https://arxiv.org/abs/2509.10771):

```text
@article{schwarke2025rslrl,
  title={RSL-RL: A Learning Library for Robotics Research},
  author={Schwarke, Clemens and Mittal, Mayank and Rudin, Nikita and Hoeller, David and Hutter, Marco},
  journal={arXiv preprint arXiv:2509.10771},
  year={2025}
}
```
