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


if __name__ == "__main__":
    unittest.main()
