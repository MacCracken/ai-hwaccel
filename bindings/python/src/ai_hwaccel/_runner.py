"""Locate and invoke the ``ai-hwaccel`` binary, returning parsed JSON.

There is no FFI into the cyrius core — the toolchain emits executables
only — so the bindings drive the compiled binary as a subprocess and
parse its JSON stdout. Binary discovery order:

1. an explicit path passed to the call
2. the ``AI_HWACCEL_BIN`` environment variable
3. a binary bundled in the wheel at ``ai_hwaccel/_bin/`` (shipped from
   2.3.3 onward; absent in the pure-source install)
4. ``ai-hwaccel`` on ``PATH``
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
    """Raised when the binary exits non-zero."""


def _bundled_path() -> Path:
    name = "ai-hwaccel.exe" if os.name == "nt" else "ai-hwaccel"
    return Path(__file__).resolve().parent / "_bin" / name


def find_binary(explicit: Optional[str] = None) -> str:
    """Return the path to a runnable ai-hwaccel binary, or raise."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("AI_HWACCEL_BIN")
    if env:
        candidates.append(env)
    candidates.append(str(_bundled_path()))
    on_path = shutil.which("ai-hwaccel")
    if on_path:
        candidates.append(on_path)

    for c in candidates:
        p = Path(c)
        if p.is_file() and os.access(c, os.X_OK):
            return c

    raise BinaryNotFoundError(
        "could not locate the 'ai-hwaccel' binary. Tried: "
        + ", ".join(candidates or ["<none>"])
        + ". Set AI_HWACCEL_BIN, put it on PATH, or pass binary=..."
    )


def _data_dir_for(exe: str) -> Optional[str]:
    """If ``exe`` is the wheel-bundled binary, return the dir holding its
    bundled ``VERSION`` + ``data/`` (so the binary's --version / --cost
    work regardless of cwd via AI_HWACCEL_DATA_DIR). Otherwise None."""
    try:
        if Path(exe).resolve() == _bundled_path().resolve():
            return str(_bundled_path().parent)
    except OSError:
        pass
    return None


def _run(args: list, binary: Optional[str], timeout: float) -> str:
    exe = find_binary(binary)
    env = os.environ.copy()
    # Point the bundled binary at its bundled data files. Never override a
    # value the caller already set, and leave PATH/explicit binaries to
    # the caller's environment (their data files, if any, are their own).
    if "AI_HWACCEL_DATA_DIR" not in env:
        data_dir = _data_dir_for(exe)
        if data_dir is not None:
            env["AI_HWACCEL_DATA_DIR"] = data_dir
    proc = subprocess.run(
        [exe, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise CommandError(
            f"ai-hwaccel {' '.join(args)} exited {proc.returncode}: "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout


def run_json(args: list, *, binary: Optional[str] = None, timeout: float = 30.0):
    """Run the binary with ``args`` and parse stdout as JSON."""
    return json.loads(_run(args, binary, timeout))


def run_text(args: list, *, binary: Optional[str] = None, timeout: float = 30.0) -> str:
    """Run the binary with ``args`` and return raw stdout text."""
    return _run(args, binary, timeout)
