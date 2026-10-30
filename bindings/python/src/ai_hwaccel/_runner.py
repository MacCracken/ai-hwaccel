"""Locate and invoke the ``ai-hwaccel`` binary, returning parsed JSON.

There is no FFI into the cyrius core — the toolchain emits executables
only — so the bindings drive the compiled binary as a subprocess and
parse its JSON stdout. Binary discovery order:

1. an explicit path passed to the call
2. the ``AI_HWACCEL_BIN`` environment variable
3. a binary bundled in the wheel at ``ai_hwaccel/_bin/`` (shipped from
   2.3.3 onward; absent in the pure-source install)
4. ``ai-hwaccel`` on ``PATH``

An explicit path or ``AI_HWACCEL_BIN`` names the build the caller chose, so
when it is not an executable file :class:`BinaryNotFoundError` is raised
rather than another binary being run (2.5.5; it fell through to 3 and 4
before).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class BinaryNotFoundError(RuntimeError):
    """Raised when no ``ai-hwaccel`` binary can be located."""


class CommandError(RuntimeError):
    """Raised when the binary exits non-zero, runs past ``timeout``, or
    prints output that is not valid JSON."""


class CommandTimeoutError(CommandError, subprocess.TimeoutExpired):
    """The binary ran past ``timeout``. It is also the
    ``subprocess.TimeoutExpired`` these calls raised before 2.5.5, with what
    the binary printed before it was killed in ``.stdout`` / ``.output`` and
    ``.stderr``, or None, as subprocess gives them: bytes on POSIX, str on
    Windows."""

    def __init__(
        self, message: str, cmd: list, timeout: float, output=None, stderr=None
    ):
        subprocess.TimeoutExpired.__init__(self, cmd, timeout, output, stderr)
        self.args = (message,)

    def __str__(self) -> str:
        return self.args[0]

    def __reduce__(self):  # picklable, e.g. out of a process pool
        state = (self.args[0], self.cmd, self.timeout, self.output, self.stderr)
        return (type(self), state)


class InvalidOutputError(CommandError, json.JSONDecodeError):
    """The binary printed output that is not valid JSON. It is also the
    ``json.JSONDecodeError`` these calls raised before 2.5.5; ``doc`` holds
    the output."""


def _bundled_path() -> Path:
    name = "ai-hwaccel.exe" if os.name == "nt" else "ai-hwaccel"
    return Path(__file__).resolve().parent / "_bin" / name


def _runnable(path: str) -> bool:
    return Path(path).is_file() and os.access(path, os.X_OK)


def find_binary(explicit: Optional[str] = None) -> str:
    """Return the path to a runnable ai-hwaccel binary, or raise.

    An explicit path, else ``AI_HWACCEL_BIN`` when it is set, is the only
    candidate: if it is not an executable file this raises instead of
    falling back to the bundled binary or ``PATH``.
    """
    if explicit:
        if _runnable(explicit):
            return explicit
        raise BinaryNotFoundError(f"binary={explicit!r} is not an executable file")
    env = os.environ.get("AI_HWACCEL_BIN")
    if env:
        if _runnable(env):
            return env
        raise BinaryNotFoundError(f"AI_HWACCEL_BIN={env!r} is not an executable file")

    candidates = [str(_bundled_path())]
    on_path = shutil.which("ai-hwaccel")
    if on_path:
        candidates.append(on_path)
    for c in candidates:
        if _runnable(c):
            return c

    raise BinaryNotFoundError(
        "could not locate the 'ai-hwaccel' binary. Tried: "
        + ", ".join(candidates)
        + ". Set AI_HWACCEL_BIN, put it on PATH, or pass binary=..."
    )


def _data_dir_for(exe: str) -> Optional[str]:
    """If ``exe`` is the wheel-bundled binary, return the dir holding its
    bundled ``data/`` (so the binary's --cost finds data/cloud_pricing.json
    regardless of cwd). Otherwise None. ``--version`` is built into the
    binary (2.5.5) and reads no file."""
    try:
        if Path(exe).resolve() == _bundled_path().resolve():
            return str(_bundled_path().parent)
    except OSError:
        pass
    return None


def _run(args: list, binary: Optional[str], timeout: float) -> str:
    exe = find_binary(binary)
    env = os.environ.copy()
    cmd = [exe]
    # Point the bundled binary at its bundled data files. Never override a
    # value the caller already set, and leave PATH/explicit binaries to
    # the caller's environment (their data files, if any, are their own).
    if "AI_HWACCEL_DATA_DIR" not in env:
        data_dir = _data_dir_for(exe)
        if data_dir is not None:
            # Pass it as a CLI flag, which every binary honours: before 2.3.21
            # the binary read environment variables only from
            # /proc/self/environ, so on macOS and Windows --cost ignored
            # AI_HWACCEL_DATA_DIR. Still export the env var for consumers
            # that read it.
            cmd += ["--data-dir", data_dir]
            env["AI_HWACCEL_DATA_DIR"] = data_dir
    cmd += args
    try:
        # The binary writes UTF-8 on every platform (Windows argv and DXGI
        # adapter names are converted to it). Text mode alone decodes with
        # the locale's code page, which on Windows is cp1252 / cp936 / ...:
        # mojibake, or a UnicodeDecodeError on a CJK locale.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        # Keep the partial output: the stderr log names the detector that hung.
        raise CommandTimeoutError(
            f"ai-hwaccel {' '.join(args)} timed out after {timeout} s",
            cmd,
            timeout,
            e.output,
            e.stderr,
        ) from e
    if proc.returncode != 0:
        raise CommandError(
            f"ai-hwaccel {' '.join(args)} exited {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def run_json(args: list, *, binary: Optional[str] = None, timeout: float = 60.0):
    """Run the binary with ``args`` and parse stdout as JSON."""
    out = _run(args, binary, timeout)
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        raise InvalidOutputError(
            f"ai-hwaccel {' '.join(args)} printed invalid JSON ({e.msg})", e.doc, e.pos
        ) from e


def run_text(args: list, *, binary: Optional[str] = None, timeout: float = 60.0) -> str:
    """Run the binary with ``args`` and return raw stdout text."""
    return _run(args, binary, timeout)
