"""Terminal styling and formatting helpers (ANSI colors, argparse help formatter).

Color output is auto-disabled when stdout is not a TTY or when ``NO_COLOR`` is set.
Set ``FORCE_COLOR`` to override and force color output (e.g. for piped output that
will be re-rendered in a terminal).
"""

from __future__ import annotations

import os
import re
import sys


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_ITALIC = "\033[3m"
ANSI_UNDERLINE = "\033[4m"

ANSI_BLACK = "\033[30m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"


def color_enabled() -> bool:
    """Return whether ANSI color output is enabled.

    Honors the de-facto standards:

    - ``NO_COLOR`` (any value) disables color.
    - ``FORCE_COLOR`` (any value) enables color regardless of TTY.
    - Otherwise, color is enabled only when ``sys.stdout`` is a TTY.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def colorize(text: str, *codes: str) -> str:
    """Wrap ``text`` with the given ANSI escape ``codes`` and a reset suffix.

    Returns ``text`` unchanged when :func:`color_enabled` is False, so it's safe to
    use unconditionally. Multiple codes are concatenated (e.g. ``ANSI_BOLD + ANSI_CYAN``).
    """
    if not codes or not color_enabled():
        return text
    return f"{''.join(codes)}{text}{ANSI_RESET}"


class ColorHelpFormatter:
    """Argparse help formatter that ANSI-colorizes section headings and ``--flag`` tokens.

    Suppresses the auto-generated ``usage:`` block in full ``--help`` output (where it
    duplicates the ``options:`` section) but **keeps** it on parse errors so
    ``parser.error(...)`` still produces a useful diagnostic. Wraps
    :class:`argparse.RawDescriptionHelpFormatter` so descriptions and epilogs preserve
    newlines.

    Subclassing is deferred to ``__new__`` so ``argparse`` is not imported at module load.
    """

    def __new__(cls, prog, *args, **kwargs):
        from argparse import RawDescriptionHelpFormatter

        class _Impl(RawDescriptionHelpFormatter):
            def start_section(self, heading):
                return super().start_section(colorize(heading, ANSI_BOLD, ANSI_YELLOW))

            def _format_action_invocation(self, action):
                text = super()._format_action_invocation(action)
                return re.sub(
                    r"(-{1,2}[A-Za-z][\w-]*)",
                    lambda m: colorize(m.group(1), ANSI_GREEN),
                    text,
                )

            def add_usage(self, usage, actions, groups, prefix=None):
                # `ArgumentParser.format_help` and `ArgumentParser.format_usage` both call
                # `formatter.add_usage(...)`. We only want to suppress the usage line in
                # the former (full --help), not the latter (used by `print_usage` /
                # `parser.error`). Distinguish by inspecting the caller frame.
                caller = sys._getframe(1).f_code.co_name
                if caller == "format_help":
                    return None
                return super().add_usage(usage, actions, groups, prefix)

        return _Impl(prog, *args, **kwargs)
