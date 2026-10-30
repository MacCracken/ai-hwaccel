"""End-to-end tests that drive the real ai-hwaccel binary.

Skipped when no binary is locatable (set AI_HWACCEL_BIN, or put ai-hwaccel
on PATH). When AI_HWACCEL_BIN is set and is not an executable file, the
module fails to import instead of skipping. CI's "Python bindings" step in
.github/workflows/ci.yml builds the binary, exports AI_HWACCEL_BIN and runs
these with ``-W error::UserWarning``, so a schema mismatch fails it.
"""

import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import ai_hwaccel  # noqa: E402
from ai_hwaccel import Registry, ShardingPlan, TrainingMemory, CostReport  # noqa: E402

if os.environ.get("AI_HWACCEL_BIN"):
    # A binary was named: one that is not there is an error, not a skip.
    _BINARY = ai_hwaccel.find_binary()
else:
    try:
        _BINARY = ai_hwaccel.find_binary()
    except ai_hwaccel.BinaryNotFoundError:
        _BINARY = None

requires_binary = unittest.skipIf(_BINARY is None, "ai-hwaccel binary not found")


@requires_binary
class TestEndToEnd(unittest.TestCase):
    def test_detect(self):
        reg = ai_hwaccel.detect()
        self.assertIsInstance(reg, Registry)
        self.assertGreaterEqual(len(reg.profiles), 1)
        # The binary and these bindings must agree on the format.
        self.assertEqual(reg.schema_version, ai_hwaccel.SCHEMA_VERSION)

    def test_summary(self):
        s = ai_hwaccel.summary()
        self.assertIn("device_count", s)
        self.assertIn("has_accelerator", s)

    def test_plan(self):
        p = ai_hwaccel.plan("70B")
        self.assertIsInstance(p, ShardingPlan)
        self.assertGreater(p.total_memory_bytes, 0)

    def test_training_memory(self):
        t = ai_hwaccel.training_memory("70B", method="lora")
        self.assertIsInstance(t, TrainingMemory)
        self.assertGreater(t.total_bytes, 0)

    def test_cost(self):
        c = ai_hwaccel.cost("70B")
        self.assertIsInstance(c, CostReport)
        self.assertEqual(c.model, "70B")

    def test_version(self):
        # Built into the binary from VERSION (2.5.5), so it is the repo's
        # from any cwd: through 2.5.4 the binary read ./VERSION at run time
        # and printed "unknown" elsewhere.
        version_file = pathlib.Path(__file__).resolve().parents[3] / "VERSION"
        self.assertEqual(ai_hwaccel.version(), version_file.read_text().strip())

    def _stderr(self, args, log_env):
        import subprocess

        env = dict(os.environ, AI_HWACCEL_LOG=log_env)
        proc = subprocess.run([_BINARY] + args, capture_output=True, text=True, env=env)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc.stderr

    def test_quiet_silences_unknown_log_env(self):
        # The unknown-name warning comes after the level flags (2.5.5), so
        # -q, documented as silent, prints nothing on stderr.
        warned = self._stderr(["--version"], "verbose")
        self.assertIn("unknown log level 'verbose'", warned)
        self.assertEqual(self._stderr(["-q", "--version"], "verbose"), "")
        quiet = self._stderr(["--log-level", "error", "--version"], "verbose")
        self.assertEqual(quiet, "")


class TestVersion(unittest.TestCase):
    """`__version__` tracks the single-source VERSION file (no binary needed).

    The package version is derived, never hardcoded: installed metadata
    when present, else the repo-root VERSION file. Both come from VERSION
    (pyproject is propagated from it by scripts/version-bump.sh), so a
    source checkout must report exactly VERSION via the fallback branch.
    """

    def test_version_is_nonempty_string(self):
        self.assertIsInstance(ai_hwaccel.__version__, str)
        self.assertTrue(ai_hwaccel.__version__)

    def test_fallback_reads_repo_version(self):
        import importlib.metadata as md

        version_file = pathlib.Path(__file__).resolve().parents[3] / "VERSION"
        expected = version_file.read_text().strip()

        # Force the "not installed" branch so the assertion is deterministic
        # regardless of whatever wheel may be pip-installed in the test env.
        original = md.version

        def _not_found(name):
            raise md.PackageNotFoundError(name)

        md.version = _not_found
        try:
            self.assertEqual(ai_hwaccel._resolve_version(), expected)
        finally:
            md.version = original


if __name__ == "__main__":
    unittest.main()
