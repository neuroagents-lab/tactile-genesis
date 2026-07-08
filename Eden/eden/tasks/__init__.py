"""Eden task configurations.

Tasks self-register via ``@TASK_REGISTRY.register()`` on the config class.
The registry AST-scans ``eden/tasks/**/*.py`` on first lookup and finds the
decorator without importing the module — so writing a new task is a single
file change with no central index to update. See
``eden/tasks/benchmark/reacher/config.py`` for the canonical example.

Use ``TASK_REGISTRY.get("task_name")`` to load a config class on demand,
or ``TASK_REGISTRY.build("task_name", **modifier_kwargs)`` to construct an
instance with applied modifiers.
"""

from eden.tasks.registry import TASK_REGISTRY, TaskMod, TaskRegistry  # noqa: F401
