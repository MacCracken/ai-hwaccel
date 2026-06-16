"""Bundled-binary tests: --version / --cost must work from any cwd.

These exercise the 2.3.3 fix — the runner sets AI_HWACCEL_DATA_DIR to the
bundled _bin/ dir so the binary finds VERSION + data/cloud_pricing.json
regardless of the caller's working directory. Skipped unless the binary
has been staged (run scripts/stage_binary.sh).
"""

import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import ai_hwaccel  # noqa: E402
from ai_hwaccel._runner import _bundled_path  # noqa: E402

_BUNDLED = _bundled_path()
requires_bundled = unittest.skipUnless(
    _BUNDLED.is_file(), "bundled binary not staged (run scripts/stage_binary.sh)"
)


@requires_bundled
class TestBundledFromForeignCwd(unittest.TestCase):
    def setUp(self):
        # Force the bundled binary and a foreign cwd; ensure no ambient
        # AI_HWACCEL_DATA_DIR masks the runner's auto-set behavior.
        self._prev_env = os.environ.pop("AI_HWACCEL_DATA_DIR", None)
        self._prev_cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)
        self._bin = str(_BUNDLED)

    def tearDown(self):
        os.chdir(self._prev_cwd)
        if self._prev_env is not None:
            os.environ["AI_HWACCEL_DATA_DIR"] = self._prev_env
        os.rmdir(self._tmp)

    def test_version_resolves_off_cwd(self):
        # Without the fix this returns "unknown" from a foreign cwd.
        v = ai_hwaccel.version(binary=self._bin)
        self.assertRegex(v, r"^\d+\.\d+\.\d+")

    def test_cost_resolves_off_cwd(self):
        # Without the fix the pricing file isn't found -> no recommendations.
        rep = ai_hwaccel.cost("70B", binary=self._bin)
        self.assertEqual(rep.model, "70B")
        self.assertTrue(rep.recommendations, "expected cloud recommendations")

    def test_detect_still_works(self):
        reg = ai_hwaccel.detect(binary=self._bin)
        self.assertGreaterEqual(len(reg.profiles), 1)

    def test_version_resolves_via_data_dir_flag(self):
        # PE-faithful path (2.3.12): on Windows the binary cannot read env
        # vars, so the runner passes --data-dir. Prove the flag ALONE
        # resolves VERSION — env var stripped, foreign cwd, flag only.
        import subprocess

        data_dir = str(_BUNDLED.parent)
        env = {k: v for k, v in os.environ.items() if k != "AI_HWACCEL_DATA_DIR"}
        proc = subprocess.run(
            [self._bin, "--data-dir", data_dir, "--version"],
            capture_output=True,
            text=True,
            cwd=self._tmp,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertRegex(proc.stdout.strip(), r"^\d+\.\d+\.\d+")

    def test_runner_passes_data_dir_flag(self):
        # The runner must put --data-dir on the argv for the bundled binary
        # (not rely on the env var, which is a no-op on PE).
        from ai_hwaccel import _runner

        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd

            class _P:
                returncode = 0
                stdout = "2.0.0\n"
                stderr = ""

            return _P()

        orig = _runner.subprocess.run
        _runner.subprocess.run = fake_run
        try:
            _runner.run_text(["--version"], binary=self._bin)
        finally:
            _runner.subprocess.run = orig
        self.assertIn("--data-dir", captured["cmd"])
        i = captured["cmd"].index("--data-dir")
        self.assertEqual(captured["cmd"][i + 1], str(_BUNDLED.parent))


if __name__ == "__main__":
    unittest.main()
