"""CLI helpers for building task configs from argparse.

This module wires :class:`~eden.tasks.registry.TaskMod` parameters into a
standard ``argparse.ArgumentParser`` so that user-facing scripts can accept
flat flags (``--difficulty hard``, ``--robot g1``) that are translated into
typed mod kwargs and applied via :meth:`TaskRegistry.build`.
"""

from __future__ import annotations

import argparse
import re
import sys
import types
from typing import Any, Union, get_args, get_origin

from eden.tasks import TASK_REGISTRY, TaskMod
from eden.tasks.registry import cli_fields, literal_choices
from eden.utils import distributed as eden_dist
from eden.utils.configs import EdenConfig
from eden.utils.format import ColorHelpFormatter

# Flag names owned by :func:`get_task_argparser` itself. A mod field that
# resolves to one of these would otherwise collide with argparse's opaque
# "conflicting option string" error at parser-build time — turn it into a
# clear message pointing at the offending mod.
_RESERVED_CLI_FLAGS: frozenset[str] = frozenset({"task", "run_name", "config", "full_config", "checkpoint", "cpu"})


def _peek_task_name(argv: list[str], default: str) -> str:
    """Pre-parse ``--task`` so we know which task's mods to add to the full parser."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--task", default=default)
    known, _ = pre.parse_known_args(argv)
    return known.task


def _add_mod_field_to_parser(
    parser: argparse.ArgumentParser,
    mod: TaskMod,
    field_name: str,
    annotation: Any,
    help_text: str | None = None,
) -> None:
    """Translate a single CLI-tunable field on a TaskMod instance into an argparse arg.

    The flag name comes from ``mod.flag_for(field_name)`` (prefix-aware) and
    the default is read off the instance — so per-registration defaults and
    per-instance prefixes both flow through. ``help_text`` (from
    ``Annotated[T, "..."]`` metadata) populates argparse ``--help``.
    """
    default = getattr(mod, field_name)
    flag = mod.flag_for(field_name)

    if flag in _RESERVED_CLI_FLAGS:
        raise ValueError(
            f"TaskMod {type(mod).__name__}.{field_name} resolves to CLI flag "
            f"'--{flag}', which is reserved by get_task_argparser. Rename the "
            f"field or set a distinct prefix via .with_prefix(...). "
            f"Reserved flags: {sorted(_RESERVED_CLI_FLAGS)}."
        )

    add_kwargs: dict[str, Any] = {"default": default}
    if help_text is not None:
        add_kwargs["help"] = help_text

    # Unwrap Optional[T] / T | None to T so path-like fields with default=None
    # work — argparse handles None defaults natively, we just need a concrete
    # type= callable for parsing the supplied value.
    if get_origin(annotation) in (Union, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            annotation = non_none[0]

    choices = literal_choices(annotation)
    if choices is not None:
        item_type = type(choices[0])
        parser.add_argument(f"--{flag}", type=item_type, choices=list(choices), **add_kwargs)
        return

    if annotation is bool:
        # BooleanOptionalAction supports both --flag and --no-flag with explicit default.
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction, **add_kwargs)
        return

    if annotation in (int, float, str):
        parser.add_argument(f"--{flag}", type=annotation, **add_kwargs)
        return

    raise TypeError(
        f"TaskMod {type(mod).__name__} field '{field_name}' has unsupported annotation {annotation!r} for CLI; "
        "supported forms: str, int, float, bool, typing.Literal[...]."
    )


def get_task_argparser(
    description: str,
    default_task_name: str = "",
    argv: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Build an ``ArgumentParser`` with standard flags plus the selected task's mod flags.

    The selected task is determined by pre-parsing ``--task`` from ``argv``
    (defaults to ``sys.argv[1:]``). Modifier flags are added only for the
    selected task, so ``--help`` shows only the relevant options. Pass
    ``argv`` explicitly when invoking this from a wrapper that already has
    its own arg slice (e.g. the ``eden train`` / ``eden inference``
    subcommands forward ``extra_args``) so the peek and the final parse
    agree on input.
    """
    task_name = _peek_task_name(sys.argv[1:] if argv is None else argv, default_task_name)

    parser = argparse.ArgumentParser(description=description, formatter_class=ColorHelpFormatter)
    parser.add_argument("--run_name", type=str, default="", help="Run name prefix for wandb.")
    parser.add_argument(
        "--task",
        type=str,
        default=default_task_name,
        help="Task name. May expose additional mod flags depending on the task.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to a partial config override (.json/.yaml).")
    parser.add_argument("--full_config", type=str, default=None, help="Path to a full config file (.json/.yaml).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend.")

    if task_name and task_name in TASK_REGISTRY:
        for mod in TASK_REGISTRY.get_modifiers(task_name):
            for field_name, annotation, _default, help_text in cli_fields(type(mod)):
                _add_mod_field_to_parser(parser, mod, field_name, annotation, help_text)

    return parser


def _collect_modifier_kwargs(args: argparse.Namespace, task_name: str) -> dict[str, Any]:
    """Pull modifier kwargs for ``task_name`` out of an argparse Namespace.

    Returns keys matching each mod's prefixed flag names, ready to pass to
    :meth:`TaskRegistry.build` without translation.
    """
    if not task_name or task_name not in TASK_REGISTRY:
        return {}
    out: dict[str, Any] = {}
    for mod in TASK_REGISTRY.get_modifiers(task_name):
        for field_name, _annotation, _default, _help in cli_fields(type(mod)):
            flag = mod.flag_for(field_name)
            if hasattr(args, flag):
                out[flag] = getattr(args, flag)
    return out


def _apply_run_name(config: Any, run_name: str) -> None:
    """Write a filesystem-safe, length-capped run_name onto ``config.runner_options`` if present."""
    if run_name and getattr(config, "runner_options", None) is not None:
        config.runner_options.run_name = re.sub(r"[/\\#?%:]", "-", run_name)[:64]


def get_task_config(
    task_name: str,
    *,
    run_name: str = "",
    modifier_kwargs: dict[str, Any] | None = None,
    config_override_path: str | None = None,
) -> Any:
    """Build a task config programmatically.

    Equivalent to :meth:`TaskRegistry.build` plus optional file overrides and
    a sanitized ``run_name`` written to ``runner_options`` if present.
    """
    config = TASK_REGISTRY.build(task_name, **(modifier_kwargs or {}))
    if config_override_path:
        config = config.with_overrides_from_file(config_override_path)
    _apply_run_name(config, run_name)
    return config


def get_task_config_from_args(args: argparse.Namespace, run_name: str = "") -> Any:
    """Build a task config from an argparse Namespace produced by :func:`get_task_argparser`."""
    if args.full_config:
        config = EdenConfig.load_from_file(args.full_config)
        if args.config:
            config = config.with_overrides_from_file(args.config)
        _apply_run_name(config, run_name)
    else:
        mod_kwargs = _collect_modifier_kwargs(args, args.task)
        config = get_task_config(
            args.task,
            run_name=run_name,
            modifier_kwargs=mod_kwargs,
            config_override_path=args.config,
        )

    runner_options = getattr(config, "runner_options", None)
    if runner_options is not None:
        if eden_dist.is_distributed():
            # Each rank sees one isolated GPU as cuda:0 via CUDA_VISIBLE_DEVICES;
            # "auto" resolves to that through gs.device, keeping device selection
            # uniform with the non-DDP path.
            runner_options.device = "auto"
        if args.cpu:
            runner_options.device = "cpu"
    return config
