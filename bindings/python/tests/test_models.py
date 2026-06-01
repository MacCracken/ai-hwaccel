"""Model-parsing tests against captured binary fixtures (no binary run)."""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import ai_hwaccel  # noqa: E402
from ai_hwaccel.models import (  # noqa: E402
    CostReport,
    Registry,
    ShardingPlan,
    TrainingMemory,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text())


class TestRegistry(unittest.TestCase):
    def test_parse(self):
        reg = Registry.from_dict(_load("registry.json"))
        self.assertEqual(reg.schema_version, ai_hwaccel.SCHEMA_VERSION)
        self.assertTrue(len(reg.profiles) >= 1)
        cpu = reg.profiles[0]
        self.assertEqual(cpu.family, "CPU")
        self.assertIsInstance(cpu.memory_bytes, int)
        self.assertIsInstance(cpu.available, bool)

    def test_system_io(self):
        reg = Registry.from_dict(_load("registry.json"))
        self.assertIsNotNone(reg.system_io)
        # Storage was detected on the capture host.
        for sd in reg.system_io.storage:
            self.assertIsInstance(sd.bandwidth_bytes_per_sec, int)
            self.assertTrue(sd.name)

    def test_optional_fields_default_none(self):
        # CPU profile has no compute_capability / temperature.
        reg = Registry.from_dict(_load("registry.json"))
        cpu = reg.profiles[0]
        self.assertIsNone(cpu.compute_capability)
        self.assertIsNone(cpu.temperature_c)

    def test_unknown_keys_ignored(self):
        reg = Registry.from_dict(
            {"schema_version": 4, "profiles": [], "future_key": 123}
        )
        self.assertEqual(reg.schema_version, 4)


class TestPlan(unittest.TestCase):
    def test_parse(self):
        plan = ShardingPlan.from_dict(_load("plan.json"))
        self.assertTrue(plan.strategy)
        self.assertIsInstance(plan.total_memory_bytes, int)
        self.assertTrue(len(plan.shards) >= 1)
        self.assertTrue(plan.shards[0].device)

    def test_est_tps_property(self):
        plan = ShardingPlan.from_dict(
            {"strategy": "None", "strategy_count": 1,
             "total_memory_bytes": 0, "est_tokens_per_sec_x1000": 28, "shards": []}
        )
        self.assertAlmostEqual(plan.est_tokens_per_sec, 0.028)

    def test_est_tps_absent(self):
        plan = ShardingPlan.from_dict(
            {"strategy": "None", "strategy_count": 1, "total_memory_bytes": 0, "shards": []}
        )
        self.assertIsNone(plan.est_tokens_per_sec)


class TestTraining(unittest.TestCase):
    def test_parse(self):
        t = TrainingMemory.from_dict(_load("training.json"))
        self.assertGreater(t.total_bytes, 0)
        self.assertEqual(
            t.total_gib_x1000,
            t.model_gib_x1000 + t.optimizer_gib_x1000 + t.activation_gib_x1000,
        )
        self.assertAlmostEqual(t.total_gib, t.total_gib_x1000 / 1000.0)


class TestCost(unittest.TestCase):
    def test_parse(self):
        c = CostReport.from_dict(_load("cost.json"))
        self.assertEqual(c.model, "70B")
        self.assertTrue(c.quantization)
        self.assertGreater(c.memory_required_bytes, 0)
        if c.recommendations:
            r = c.recommendations[0]
            self.assertTrue(r.instance)
            self.assertAlmostEqual(
                r.price_per_hour_usd, r.price_per_hour_usd_x100 / 100.0
            )


if __name__ == "__main__":
    unittest.main()
