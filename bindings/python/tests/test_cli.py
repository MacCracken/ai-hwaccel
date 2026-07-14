"""End-to-end tests that drive the real ai-hwaccel binary.

Skipped automatically when no binary is locatable (set AI_HWACCEL_BIN,
or put ai-hwaccel on PATH). CI builds the binary and exports
AI_HWACCEL_BIN before running these.
"""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import ai_hwaccel  # noqa: E402
from ai_hwaccel import Registry, ShardingPlan, TrainingMemory, CostReport  # noqa: E402

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
        # The binary reads ./VERSION at runtime (cwd-relative), so it may
        # report "unknown" when run from outside the repo root. Assert the
        # wrapper returns a non-empty string either way; the cwd/data-file
        # dependence is documented and tracked for the 2.3.3 packaging work.
        v = ai_hwaccel.version()
        self.assertIsInstance(v, str)
        self.assertTrue(v)


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
