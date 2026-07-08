"""Task registry: lazy task lookup and CLI field derivation.

:data:`TASK_REGISTRY` resolves task config classes by name, importing the owning module
lazily on first access. Built-in tasks are registered in :mod:`eden.tasks`
(``eden/tasks/__init__.py``); register your own with ``@TASK_REGISTRY.register()`` (or
``register(name=..., override=True)`` to shadow a built-in). This module also derives
argparse CLI fields from a task config's keyword-only fields.
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
import textwrap
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from eden.utils.misc import to_snake_case


def _default_task_name(class_name: str) -> str:
    """Canonical task name from a config class name.

    Applies ``to_snake_case`` and strips a trailing ``_config`` suffix so that
    ``ReacherConfig`` → ``reacher`` (not ``reacher_config``), matching the
    convention used across ``eden/tasks/**/config.py``.
    """
    snake = to_snake_case(class_name)
    return snake.removesuffix("_config") or snake


class TaskMod:
    """Tool that modifies an EdenConfig instance.

    TaskMod is a plain Python class (not a Pydantic model) because it is
    *logic* with a few parameters, not user-facing data. ``EdenConfig`` is the
    Pydantic data model; ``TaskMod`` operates on it.

    To define one: subclass, declare CLI-tunable parameters as
    **positional-or-keyword** ``__init__`` arguments with type annotations,
    and implement :meth:`apply`. Non-CLI constructor state goes in
    **keyword-only** parameters (after ``*``) — the CLI introspection skips
    those.

    Quick reference (this is the convention agents miss most often)::

        def __init__(
            self,
            difficulty: Literal["easy", "hard"] = "easy",  # ← CLI flag (--difficulty)
            *,
            entity_name: str = "robot",                    # ← non-CLI ctor state
        ):
            self.difficulty = difficulty
            self.entity_name = entity_name

    Calling ``super().__init__()`` is **optional** — ``TaskMod`` keeps no
    instance state of its own; ``_prefix`` is a class-attribute default that
    :meth:`with_prefix` shadows on assignment. So your subclass ``__init__``
    can just store its parameters and return.

    For per-field CLI ``--help`` text, wrap the annotation in
    ``typing.Annotated[T, "help text"]``. The string metadata is forwarded
    to ``argparse``; the inner type is what drives validation::

        difficulty: Annotated[Literal["easy", "hard"], "Task difficulty preset."] = "easy"

    Implement :meth:`apply` to return the modified config; in-place mutation
    is fine because :meth:`TaskRegistry.build` calls ``cls()`` fresh on every
    build (and Pydantic deep-copies the config's BaseModel defaults per
    instance), so mutations do not leak across builds.

    Modifiers attach to a task by passing **instances** to
    ``@TASK_REGISTRY.register(modifiers=(...))`` (a bare subclass is accepted
    and instantiated with no args). Modifiers are stored on the registry,
    not on the config class.

    Per-instance scoping: chain :meth:`with_prefix` on a constructed instance
    so its CLI flags rename from ``<field>`` to ``<prefix>_<field>``. The same
    subclass can be registered multiple times with distinct prefixes.

    Example
    -------
    ::

        from typing import Annotated, Literal

        class RobotMod(TaskMod):
            def __init__(
                self,
                robot: Annotated[
                    Literal["allegro", "shadow", "xhand1"],
                    "Hand model to load into the scene.",
                ] = "xhand1",
                *,
                entity_name: str = "robot",  # keyword-only → not CLI-exposed
            ):
                self.robot = robot
                self.entity_name = entity_name

            def apply(self, config):
                setattr(config.scene_options, self.entity_name, ROBOT_REGISTRY.get(self.robot)())
                return config

        @TASK_REGISTRY.register(modifiers=(
            RobotMod(robot="shadow", entity_name="robot_left").with_prefix("left"),
            RobotMod(entity_name="robot_right").with_prefix("right"),
        ))
        class TwoArmTask(EdenRLConfig):
            ...

        # CLI: --left_robot shadow --right_robot xhand1
        TASK_REGISTRY.build("two_arm_task", left_robot="allegro", right_robot="shadow")
    """

    # Per-instance CLI/kwarg namespace; class-attribute default lets subclasses
    # skip `super().__init__()`. with_prefix() shadows this with an instance attr.
    _prefix: str = ""

    @property
    def prefix(self) -> str:
        """Per-instance CLI/kwarg namespace; empty string means unprefixed."""
        return self._prefix

    def with_prefix(self, prefix: str) -> "TaskMod":
        """Set this instance's CLI/kwarg prefix and return self for chaining.

        Chain at registration time::

            RobotMod(entity_name="robot_left").with_prefix("left")
        """
        self._prefix = prefix
        return self

    def flag_for(self, field_name: str) -> str:
        """Return the CLI/build-kwarg name for a field on this instance.

        With a non-empty :attr:`prefix`, returns ``f"{prefix}_{field_name}"``;
        otherwise returns ``field_name`` unchanged.
        """
        return f"{self._prefix}_{field_name}" if self._prefix else field_name

    def apply(self, config: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement apply().")


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    """Strip ``typing.Annotated[T, ...]`` and return ``(T, first_string_meta)``.

    Annotation pass-through: non-Annotated annotations come back unchanged
    with ``help=None``. The first string in the metadata tuple is treated as
    argparse help text; non-string metadata is ignored so other tools can
    layer their own markers without conflict.
    """
    if get_origin(annotation) is Annotated:
        inner, *metadata = get_args(annotation)
        help_text = next((m for m in metadata if isinstance(m, str)), None)
        return inner, help_text
    return annotation, None


def cli_fields(mod_cls: type[TaskMod]) -> list[tuple[str, Any, Any, str | None]]:
    """Return ``[(name, annotation, default, help), ...]`` for a TaskMod subclass's CLI fields.

    CLI fields are the **positional-or-keyword** parameters of ``mod_cls.__init__``
    (excluding ``self``). Keyword-only parameters (declared after ``*`` in the
    signature) are non-CLI constructor state and skipped here. Annotations are
    resolved with :func:`typing.get_type_hints` (``include_extras=True``) so
    postponed annotations (``from __future__ import annotations``) and
    ``Annotated[...]`` wrappers are returned as real types. ``help`` is the
    first string inside ``Annotated[T, ...]`` metadata, or ``None``.
    """
    if mod_cls is TaskMod:
        return []
    sig = inspect.signature(mod_cls.__init__)
    # Narrow the resolution failure to the cases we expect — typo'd
    # forward references (NameError) and malformed Annotated args
    # (TypeError). Other exceptions are real bugs and should propagate.
    # We still warn on the survivable cases so the user sees what failed
    # instead of getting a misleading "unsupported annotation" downstream.
    try:
        hints = get_type_hints(mod_cls.__init__, include_extras=True)
    except (NameError, TypeError) as exc:
        warnings.warn(
            f"cli_fields: could not resolve type hints for "
            f"{mod_cls.__module__}.{mod_cls.__name__}.__init__ "
            f"({type(exc).__name__}: {exc}). Falling back to raw annotations; "
            f"CLI flag generation may misclassify fields on this mod.",
            stacklevel=2,
        )
        hints = {}
    out: list[tuple[str, Any, Any, str | None]] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
            continue
        raw = hints.get(name, param.annotation)
        annotation, help_text = _unwrap_annotated(raw)
        default = param.default if param.default is not inspect.Parameter.empty else None
        out.append((name, annotation, default, help_text))
    return out


def _keyword_only_fields(mod_cls: type[TaskMod]) -> set[str]:
    """Return the names of keyword-only ``__init__`` parameters (the non-CLI bucket).

    Used by :meth:`TaskRegistry.build` to convert a confusing
    ``Unknown modifier kwargs`` error into a targeted hint when the unknown
    kwarg matches a parameter that the author placed after ``*`` (the
    convention for non-CLI constructor state).
    """
    if mod_cls is TaskMod:
        return set()
    sig = inspect.signature(mod_cls.__init__)
    return {
        name
        for name, param in sig.parameters.items()
        if name != "self" and param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def _check_cli_value(field_name: str, annotation: Any, value: Any) -> None:
    """Validate ``value`` against a CLI field's annotation. Raises ValueError on mismatch.

    Replaces what Pydantic's construction-time validation gave us: Literal
    membership and primitive type coercion. Called from
    :meth:`TaskRegistry.build` before applying overrides — argparse already
    enforces these for the CLI path, so this only fires for programmatic
    ``build("task", field=value)`` calls.
    """
    choices = literal_choices(annotation)
    if choices is not None:
        if value not in choices:
            raise ValueError(f"Invalid value for '{field_name}': {value!r} is not one of {list(choices)}.")
        return
    if annotation in (int, float, str, bool) and not isinstance(value, annotation):
        raise ValueError(
            f"Invalid value for '{field_name}': expected {annotation.__name__}, got {type(value).__name__}."
        )


# A modifier entry at registration time is one of:
#   * a TaskMod instance (preferred; supports prefix + non-CLI ctor state),
#   * a TaskMod subclass (auto-instantiated with no args — convenience).
ModifierEntry = TaskMod | type[TaskMod]


class TaskRegistry:
    """Task registry with two lookup paths, tried in order.

    1. Eager registry — populated by ``@TASK_REGISTRY.register()`` decorators
       or explicit ``register(cls, ...)`` calls.
    2. Auto-discovery — an AST scan of the configured search packages (default
       ``["eden.tasks"]``) finds ``@TASK_REGISTRY.register()`` decorators in
       every ``*.py`` file without importing them. On first :meth:`get` miss
       we import only the one file that matches the requested name; that
       triggers the decorator and populates the eager registry.

    Tasks self-register at the definition site — no central index in
    ``eden/tasks/__init__.py`` to keep in sync. Out-of-tree tasks can opt
    into discovery by calling :meth:`add_search_path` from the consuming
    package.

    Auto-discovery rules (files that are NOT scanned):

    * ``__init__.py`` is skipped — put decorated classes in a sibling module
      such as ``config.py``.
    * files whose name starts with ``_`` are skipped (e.g. ``_meta.py``) —
      private modules are treated as implementation details.

    Auto-discovery rules (decorator recognition):

    * The scan recognizes ``@<name>.register`` where ``<name>`` is bound, in
      the same file, to ``TASK_REGISTRY`` via an absolute import from
      ``eden.tasks`` or ``eden.tasks.registry`` (``from ... import
      TASK_REGISTRY`` or ``... as X``). Both forms work.
    * Using the literal name ``TASK_REGISTRY`` without importing it from a
      recognized location emits a warning — the scan will not silently miss
      the registration.

    Usage::

        # Preferred: decorator at definition site; modifiers passed to the
        # decorator so they live on the registry (not on the config class,
        # which is a Pydantic model and should only hold serializable schema).
        @TASK_REGISTRY.register(modifiers=(ReacherDifficultyMod,))
        class ReacherConfig(EdenRLConfig):
            ...

        cls = TASK_REGISTRY.get("reacher")                   # returns the class
        cfg = TASK_REGISTRY.build("reacher", difficulty="hard")  # returns an instance
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}
        # Modifiers explicitly attached at registration time (override class attr).
        self._modifiers: dict[str, tuple[TaskMod, ...]] = {}
        # Packages to AST-scan for @TASK_REGISTRY.register decorators.
        self._search_packages: list[str] = ["eden.tasks"]
        # Cached discovery map: name -> "module.path:ClassName". None means "not yet scanned".
        self._discovered: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        obj: Any = None,
        *,
        name: str | None = None,
        override: bool = False,
        modifiers: tuple[ModifierEntry, ...] = (),
    ) -> Any:
        """Register a config class eagerly. Can be used as a decorator or function call.

        Parameters
        ----------
        obj: Any
            The config class to register. When ``None``, returns a decorator.
        name: str | None
            Lookup name. Defaults to ``_default_task_name(cls.__name__)``
            (``to_snake_case`` with any trailing ``_config`` stripped).
        override: bool
            If ``True``, silently replace an existing registration with the
            same name.
        modifiers: tuple[ModifierEntry, ...]
            Optional sequence of TaskMod **instances** (preferred) or bare
            TaskMod subclasses (auto-instantiated with no args). Applied in
            order by :meth:`build`.
        """

        def deco(cls: Any) -> Any:
            self._do_register(name or _default_task_name(cls.__name__), cls, override=override, modifiers=modifiers)
            return cls

        return deco if obj is None else deco(obj)

    def add_search_path(self, package: str) -> None:
        """Register an additional package root to AST-scan for decorated tasks.

        Use this from a user project to make auto-discovery pick up
        out-of-tree tasks::

            from eden.tasks import TASK_REGISTRY
            TASK_REGISTRY.add_search_path("my_project.tasks")
        """
        if package in self._search_packages:
            return
        self._search_packages.append(package)
        # Invalidate the discovery cache so the new package is scanned on the next miss.
        self._discovered = None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Get a config CLASS by name.

        Resolution order: eager registry → auto-discovery. The returned
        object is the config CLASS. Use :meth:`build` to obtain a configured
        instance.
        """
        if name in self._registry:
            return self._registry[name]

        discovered = self._ensure_discovered()
        if name in discovered:
            qualname = discovered[name]
            module_path = qualname.rsplit(":", 1)[0]
            # Importing the module triggers @TASK_REGISTRY.register(), which
            # populates self._registry[name].
            importlib.import_module(module_path)
            if name in self._registry:
                return self._registry[name]
            # Import succeeded but the decorator didn't fire (e.g. the class
            # is gated behind `if __name__ == "__main__":`). Report clearly.
            raise RuntimeError(
                f"Auto-discovered task '{name}' at '{qualname}' but importing its module "
                "did not register it. Check that @TASK_REGISTRY.register() runs at import time."
            )

        available = sorted(set(self._registry) | set(discovered))
        available_str = ", ".join(available) or "none"
        raise KeyError(f"No task named '{name}' found in task registry. Available: {available_str}")

    def get_modifiers(self, name: str) -> tuple[TaskMod, ...]:
        """Resolve and return the TaskMod **instances** associated with ``name``.

        Modifiers live on the registry itself (not on the config class),
        attached via ``register(modifiers=...)``. For decorator-registered
        tasks the decorator's ``modifiers=`` tuple is populated when the
        module is imported, so we call :meth:`get` first to ensure that
        import has happened.
        """
        if name not in self:
            raise KeyError(f"No task named '{name}' found in task registry.")
        # Ensure the module is loaded so the decorator has had a chance to
        # attach modifiers. No-op if already loaded.
        self.get(name)
        return self._modifiers.get(name, ())

    def build(self, name: str, **modifier_kwargs: Any) -> Any:
        """Instantiate the config and apply registered TaskMods.

        Modifier kwargs are flat and use each mod's **prefixed** field names
        (see :meth:`TaskMod.flag_for`). For an unprefixed mod that exposes
        ``difficulty``, pass ``difficulty=...``; for an instance registered
        with ``prefix="left"`` that exposes ``robot``, pass ``left_robot=...``.

        Collisions — two mods that would claim the same flag — raise
        ``ValueError`` at build time with both owners listed. Unknown kwargs
        raise ``ValueError`` with the list of accepted flags.
        """
        cls = self.get(name)
        config = cls()
        mods = self._modifiers.get(name, ())

        # Map each prefixed flag to (mod index, field_name, annotation).
        flag_map: dict[str, tuple[int, str, Any]] = {}
        for i, mod in enumerate(mods):
            for field_name, annotation, _default, _help in cli_fields(type(mod)):
                flag = mod.flag_for(field_name)
                if flag in flag_map:
                    j, other_field, _ = flag_map[flag]
                    raise ValueError(
                        f"Modifier flag collision on task '{name}': '{flag}' is claimed by "
                        f"both mod[{j}] ({type(mods[j]).__name__}.{other_field}) and "
                        f"mod[{i}] ({type(mod).__name__}.{field_name}). "
                        "Give the TaskMod instances distinct prefix values via .with_prefix(...)."
                    )
                flag_map[flag] = (i, field_name, annotation)

        unknown = set(modifier_kwargs) - set(flag_map)
        if unknown:
            # Surface a targeted hint when the unknown kwarg matches a
            # keyword-only init param on a registered mod — that's the most
            # common adoption mistake (declaring a CLI knob after `*` so it
            # never reaches the CLI surface). Match on the prefixed flag form
            # so the hint also fires for prefixed mods.
            kwonly_index: dict[str, tuple[TaskMod, str]] = {}
            for mod in mods:
                for field in _keyword_only_fields(type(mod)):
                    kwonly_index[mod.flag_for(field)] = (mod, field)
            kwonly_hits: list[str] = []
            for u in sorted(unknown):
                if u in kwonly_index:
                    mod, field = kwonly_index[u]
                    kwonly_hits.append(
                        f"'{u}' is keyword-only on {type(mod).__name__} (as '{field}') — "
                        f"keyword-only params are non-CLI ctor state. Set it at registration "
                        f"with modifiers=({type(mod).__name__}({field}=...),); if you own "
                        f"{type(mod).__name__} and want a CLI flag, move '{field}' before "
                        f"`*` in __init__."
                    )
            hint = ("\n" + textwrap.indent("Hint:\n" + "\n".join(kwonly_hits), "  ")) if kwonly_hits else ""
            raise ValueError(
                f"Unknown modifier kwargs for task '{name}': {sorted(unknown)}. "
                f"Known: {sorted(flag_map) or 'none'}{hint}"
            )

        # Group kwargs per mod, stripping the prefix and validating each value
        # against its declared annotation (Literal membership + primitive type).
        updates_per_mod: list[dict[str, Any]] = [{} for _ in mods]
        for flag, value in modifier_kwargs.items():
            i, field_name, annotation = flag_map[flag]
            _check_cli_value(field_name, annotation, value)
            updates_per_mod[i][field_name] = value

        for mod, updates in zip(mods, updates_per_mod):
            if updates:
                # Deep copy so per-build overrides don't mutate the registered
                # instance; deep so mutable instance state (lists/dicts a
                # subclass might keep) is not aliased across builds.
                applied = copy.deepcopy(mod)
                for field_name, value in updates.items():
                    setattr(applied, field_name, value)
            else:
                applied = mod
            config = applied.apply(config)
        return config

    def list_tasks(self) -> list[str]:
        """List all registered task names (eager + discovered)."""
        discovered = self._ensure_discovered()
        return sorted(set(self._registry) | set(discovered))

    def __contains__(self, name: str) -> bool:
        if name in self._registry:
            return True
        return name in self._ensure_discovered()

    def __repr__(self) -> str:
        names = self.list_tasks()
        if not names:
            return "TaskRegistry: (empty)"
        return "TaskRegistry:\n" + "\n".join(f"  {n}" for n in names)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _do_register(
        self,
        name: str,
        obj: Any,
        *,
        override: bool = False,
        modifiers: tuple[ModifierEntry, ...] = (),
    ) -> None:
        if not override and name in self._registry:
            raise ValueError(f"A task named '{name}' is already registered in the task registry")
        self._registry[name] = obj
        self._modifiers[name] = _normalize_modifier_entries(modifiers)

    def _ensure_discovered(self) -> dict[str, str]:
        if self._discovered is None:
            self._discovered = _scan_packages_for_tasks(self._search_packages)
        return self._discovered

    def _invalidate_discovery_cache(self) -> None:
        """For tests: force the next lookup to re-scan the filesystem."""
        self._discovered = None


# ---------------------------------------------------------------------------
# AST-based auto-discovery
# ---------------------------------------------------------------------------


# Modules that legitimately export ``TASK_REGISTRY``. A decorator matches only
# when the identifier it uses is bound (directly or via ``as``) to one of these.
_TASK_REGISTRY_IMPORT_SOURCES = frozenset({"eden.tasks", "eden.tasks.registry"})


def _task_registry_aliases(tree: ast.AST) -> set[str]:
    """Return the set of local names in ``tree`` bound to ``TASK_REGISTRY``.

    Only ``from eden.tasks[.registry] import TASK_REGISTRY [as X]`` forms are
    recognized (absolute imports; matches CLAUDE.md's import convention). Names
    introduced by other routes (``import eden.tasks``, relative imports,
    re-export through a third module) are not tracked — decorators relying on
    those will either (a) match the bare ``TASK_REGISTRY`` fallback in
    :func:`_task_name_from_decorators` and emit a warning, or (b) not match at
    all if the identifier is unrecognized.
    """
    aliases: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _TASK_REGISTRY_IMPORT_SOURCES:
            for alias in node.names:
                if alias.name == "TASK_REGISTRY":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _scan_packages_for_tasks(packages: list[str]) -> dict[str, str]:
    """Return ``{task_name: "module.path:ClassName"}`` for every registered task class in ``packages``.

    Scans for classes decorated with ``@TASK_REGISTRY.register`` (with or without args).

    The scan does NOT import the target modules — it parses their source with
    :mod:`ast` so auto-discovery preserves the lazy-import contract.

    Matching is alias-aware: a decorator is recognized whenever its attribute
    base is a local name bound to ``TASK_REGISTRY`` via ``from eden.tasks[.registry]
    import TASK_REGISTRY [as X]``. When a file uses the literal name
    ``TASK_REGISTRY`` but hasn't imported it from a recognized location, the
    scan emits a warning so the miss is loud rather than silent.

    Collisions between files raise ``ValueError`` with both locations listed;
    collisions with explicitly-registered (eager/lazy) names are resolved
    later at :meth:`TaskRegistry.get` time (explicit wins).
    """
    out: dict[str, str] = {}
    locations: dict[str, str] = {}  # name -> source file path, for diagnostics
    for package in packages:
        for module_path, file_path in _iter_package_py_files(package):
            try:
                source = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                warnings.warn(
                    f"TaskRegistry auto-discovery: could not read '{file_path}': {exc}. "
                    "Tasks defined in this file will not be discoverable.",
                    stacklevel=2,
                )
                continue
            try:
                tree = ast.parse(source, filename=str(file_path))
            except SyntaxError as exc:
                warnings.warn(
                    f"TaskRegistry auto-discovery: syntax error in '{file_path}': {exc}. "
                    "Tasks defined in this file will not be discoverable.",
                    stacklevel=2,
                )
                continue
            aliases = _task_registry_aliases(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                registered_name = _task_name_from_decorators(node, aliases=aliases, file_path=file_path)
                if registered_name is None:
                    continue
                qualname = f"{module_path}:{node.name}"
                if registered_name in out and out[registered_name] != qualname:
                    prev_loc = locations.get(registered_name, out[registered_name])
                    raise ValueError(
                        f"Task name '{registered_name}' is registered in two locations:\n"
                        f"  - {prev_loc}\n"
                        f"  - {file_path}\n"
                        "Pass name='...' to @TASK_REGISTRY.register() to disambiguate."
                    )
                out[registered_name] = qualname
                locations[registered_name] = str(file_path)
    return out


def _iter_package_py_files(package: str):
    """Yield ``(dotted.module.path, Path)`` for every scannable ``*.py`` file under ``package``.

    Two kinds of files are skipped:

    * ``__init__.py`` — typically re-exports; tasks live in siblings.
    * files whose name starts with ``_`` (e.g. ``_meta.py``) — private modules.

    Writing ``@TASK_REGISTRY.register()`` inside a skipped file will NOT be
    discovered; put the decorated class in a regular module (e.g. ``config.py``).

    Namespace packages are handled via the importlib spec.
    """
    try:
        spec = find_spec(package)
    except (ImportError, ValueError):
        # ImportError if a parent package is missing; ValueError if the module
        # was imported but has a mangled spec. Either way: nothing to discover.
        return
    if spec is None:
        return
    search_locations = list(spec.submodule_search_locations or [])
    for root_dir in search_locations:
        root = Path(root_dir)
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            name = py_file.name
            if name == "__init__.py":
                continue
            if name.startswith("_"):
                continue
            relative = py_file.relative_to(root).with_suffix("")
            module_path = package + "." + ".".join(relative.parts)
            yield module_path, py_file


def _task_name_from_decorators(
    cls_def: ast.ClassDef,
    *,
    aliases: set[str] | frozenset[str] = frozenset({"TASK_REGISTRY"}),
    file_path: Path | None = None,
) -> str | None:
    """Return the registered task name if a decorator is ``@<alias>.register``, else None.

    Matches ``@<alias>.register`` with or without call args.

    ``aliases`` is the set of local names the scanned module has bound to
    ``TASK_REGISTRY`` (see :func:`_task_registry_aliases`). The default
    ``{"TASK_REGISTRY"}`` lets this helper be called in isolation (e.g. from
    unit tests) without import-tracking.

    When a decorator literally uses the name ``TASK_REGISTRY`` but it's not in
    ``aliases`` (i.e. the module didn't import it from a recognized location),
    a warning is emitted so the miss is loud rather than silent. Decorators
    that use an unrelated ``*_REGISTRY`` name are ignored quietly — Eden has
    several term registries that share the ``.register()`` shape.
    """
    for dec in cls_def.decorator_list:
        call_args = None
        if isinstance(dec, ast.Call):
            func = dec.func
            call_args = dec
        else:
            func = dec
        # Must look like <name>.register
        if not (isinstance(func, ast.Attribute) and func.attr == "register" and isinstance(func.value, ast.Name)):
            continue
        reg_name = func.value.id
        if reg_name not in aliases:
            # Loud miss: the file used the canonical name but didn't import it.
            if reg_name == "TASK_REGISTRY":
                where = f" in '{file_path}'" if file_path is not None else ""
                warnings.warn(
                    f"TaskRegistry auto-discovery: @TASK_REGISTRY.register on "
                    f"'{cls_def.name}'{where} is not recognized — the module "
                    f"does not import TASK_REGISTRY from 'eden.tasks' or "
                    f"'eden.tasks.registry'. This task will not be discovered. "
                    f"Add `from eden.tasks import TASK_REGISTRY` at the top of "
                    f"the file.",
                    stacklevel=2,
                )
            continue
        # If called with explicit name=, use that.
        if call_args is not None:
            for kw in call_args.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    return kw.value.value
        return _default_task_name(cls_def.name)
    return None


def _normalize_modifier_entries(entries: tuple[ModifierEntry, ...]) -> tuple[TaskMod, ...]:
    """Normalize registration-time entries to a tuple of TaskMod instances.

    Bare TaskMod subclasses are instantiated with no args eagerly so that
    registration-time errors surface immediately; instances pass through.
    """
    out: list[TaskMod] = []
    for e in entries:
        if isinstance(e, TaskMod):
            out.append(e)
        elif isinstance(e, type) and issubclass(e, TaskMod):
            out.append(e())
        else:
            raise TypeError(f"Modifier entries must be TaskMod instances or TaskMod subclasses; got {e!r}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Helpers used by the CLI parser (eden/tasks/parser.py)
# ---------------------------------------------------------------------------


def literal_choices(annotation: Any) -> tuple[Any, ...] | None:
    """Return the choices for a Literal annotation, or None if not a Literal."""
    return get_args(annotation) if get_origin(annotation) is Literal else None


# Module-level singleton; populated by discovery + eden.tasks.__init__.
TASK_REGISTRY = TaskRegistry()
