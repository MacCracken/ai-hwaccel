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


if __name__ == "__main__":
    unittest.main()
