"""Bundled-binary tests: --cost must find its data from any cwd.

These exercise the 2.3.3 fix: the runner points the bundled binary at the
bundled _bin/ dir (--data-dir, and AI_HWACCEL_DATA_DIR) so it finds
data/cloud_pricing.json regardless of the caller's working directory. Each
channel is also run alone, since the runner's two would hide a break in
either. --version is built into the binary (2.5.5) and reads no file, so
only --cost proves a data dir resolves. Skipped unless the binary has been
staged (run scripts/stage_binary.sh).
"""

import json
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

    def _direct(self, args, env_dir=None):
        # The binary itself, from the foreign cwd, with AI_HWACCEL_DATA_DIR
        # only when `env_dir` is given.
        import subprocess

        env = {k: v for k, v in os.environ.items() if k != "AI_HWACCEL_DATA_DIR"}
        if env_dir is not None:
            env["AI_HWACCEL_DATA_DIR"] = env_dir
        return subprocess.run(
            [self._bin] + args, capture_output=True, text=True, cwd=self._tmp, env=env
        )

    def _cost_recommendations(self, args, env_dir=None):
        proc = self._direct(args + ["--cost", "70B", "--json"], env_dir)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)["recommendations"], proc.stderr

    def test_version_is_the_bundled_version(self):
        # Built in (2.5.5): the VERSION the wheel ships beside it, from any
        # cwd and whatever --data-dir names.
        want = (_BUNDLED.parent / "VERSION").read_text().strip()
        self.assertEqual(ai_hwaccel.version(binary=self._bin), want)
        nowhere = os.path.join(self._tmp, "none")
        proc = self._direct(["--data-dir", nowhere, "--version"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), want)

    def test_cost_resolves_off_cwd(self):
        # Without the fix the pricing file isn't found -> no recommendations.
        rep = ai_hwaccel.cost("70B", binary=self._bin)
        self.assertEqual(rep.model, "70B")
        self.assertTrue(rep.recommendations, "expected cloud recommendations")

    def test_detect_still_works(self):
        reg = ai_hwaccel.detect(binary=self._bin)
        self.assertGreaterEqual(len(reg.profiles), 1)

    def test_cost_resolves_via_data_dir_flag(self):
        # PE-faithful path (2.3.12): the runner passes --data-dir, which
        # every binary honours. The flag ALONE: env var stripped, foreign
        # cwd. This test ran --version until 2.5.5 built the version in,
        # which left it passing whatever the flag did.
        recs, err = self._cost_recommendations(["--data-dir", str(_BUNDLED.parent)])
        self.assertTrue(recs, err)

    def test_cost_resolves_via_env(self):
        recs, err = self._cost_recommendations([], env_dir=str(_BUNDLED.parent))
        self.assertTrue(recs, err)

    def test_cost_needs_a_data_dir_off_cwd(self):
        # The control: with neither, the foreign cwd has no data/, so the two
        # tests above pass because of the data dir alone.
        recs, err = self._cost_recommendations([])
        self.assertEqual(recs, [])
        self.assertIn("no cloud pricing rows", err)

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
