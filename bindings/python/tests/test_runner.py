"""Runner and API tests that need no ai-hwaccel binary.

``subprocess.run`` is replaced by a fake, and ``sys.executable`` stands in
for an executable file wherever ``find_binary`` must accept one.
"""

import json
import os
import pathlib
import pickle
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import ai_hwaccel  # noqa: E402
from ai_hwaccel import _runner  # noqa: E402

_TRAINING = {
    "model_bytes": 1, "model_gib_x1000": 0, "optimizer_bytes": 1,
    "optimizer_gib_x1000": 0, "activation_bytes": 1,
    "activation_gib_x1000": 0, "total_bytes": 3, "total_gib_x1000": 0,
}


class _Proc:
    def __init__(self, stdout, returncode=0, stderr=""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


class TestRunnerErrors(unittest.TestCase):
    def _run_json(self, fake):
        with mock.patch.object(_runner.subprocess, "run", fake):
            return _runner.run_json(["--cost", "x"], binary=sys.executable, timeout=5)

    def test_decodes_utf8(self):
        # Text mode alone decodes with the locale code page (cp1252 / cp936 on
        # Windows); the binary writes UTF-8 everywhere.
        seen = {}

        def fake(cmd, **kw):
            seen.update(kw)
            return _Proc('{"device_name": "Microsoft 基本显示适配器"}')

        self.assertEqual(self._run_json(fake)["device_name"], "Microsoft 基本显示适配器")
        self.assertEqual(seen.get("encoding"), "utf-8")
        self.assertEqual(seen.get("errors"), "replace")
        self.assertNotIn("text", seen)

    def test_bad_json_is_command_error(self):
        def fake(cmd, **kw):
            return _Proc('{"model":"Llama "3" 8B"}')

        with self.assertRaises(ai_hwaccel.CommandError) as cm:
            self._run_json(fake)
        e = cm.exception
        self.assertIsInstance(e, ai_hwaccel.InvalidOutputError)
        # Still what callers caught before 2.5.5.
        self.assertIsInstance(e, json.JSONDecodeError)
        self.assertEqual(e.doc, '{"model":"Llama "3" 8B"}')
        self.assertIn("--cost x", str(e))
        self.assertEqual(str(pickle.loads(pickle.dumps(e))), str(e))

    def test_timeout_is_command_error(self):
        def fake(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])

        with self.assertRaises(ai_hwaccel.CommandError) as cm:
            self._run_json(fake)
        e = cm.exception
        self.assertIsInstance(e, ai_hwaccel.CommandTimeoutError)
        self.assertIsInstance(e, subprocess.TimeoutExpired)
        self.assertEqual(e.timeout, 5)
        self.assertIn("timed out after 5 s", str(e))
        self.assertEqual(str(pickle.loads(pickle.dumps(e))), str(e))

    def test_nonzero_exit_is_command_error(self):
        def fake(cmd, **kw):
            return _Proc("", returncode=2, stderr="boom")

        with self.assertRaisesRegex(ai_hwaccel.CommandError, "exited 2: boom"):
            self._run_json(fake)


class TestFindBinary(unittest.TestCase):
    """An explicit binary= or AI_HWACCEL_BIN is never skipped for another."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.missing = os.path.join(self._tmp, "no-such-ai-hwaccel")
        self.not_exec = os.path.join(self._tmp, "ai-hwaccel")
        with open(self.not_exec, "w") as f:
            f.write("#!/bin/sh\n")
        os.chmod(self.not_exec, 0o644)

    def tearDown(self):
        os.remove(self.not_exec)
        os.rmdir(self._tmp)

    def test_explicit_missing_raises_even_with_valid_env(self):
        with mock.patch.dict(os.environ, {"AI_HWACCEL_BIN": sys.executable}):
            with self.assertRaisesRegex(ai_hwaccel.BinaryNotFoundError, "binary="):
                ai_hwaccel.find_binary(self.missing)

    @unittest.skipIf(os.name == "nt", "no execute bit on Windows")
    def test_explicit_not_executable_raises(self):
        with mock.patch.dict(os.environ, {"AI_HWACCEL_BIN": sys.executable}):
            with self.assertRaises(ai_hwaccel.BinaryNotFoundError):
                ai_hwaccel.find_binary(self.not_exec)

    def test_env_missing_raises_even_with_valid_fallbacks(self):
        with mock.patch.dict(os.environ, {"AI_HWACCEL_BIN": self.missing}), \
                mock.patch.object(_runner, "_bundled_path", return_value=pathlib.Path(sys.executable)), \
                mock.patch.object(_runner.shutil, "which", return_value=sys.executable):
            with self.assertRaisesRegex(ai_hwaccel.BinaryNotFoundError, "AI_HWACCEL_BIN="):
                ai_hwaccel.find_binary()

    def test_explicit_wins_over_env(self):
        with mock.patch.dict(os.environ, {"AI_HWACCEL_BIN": self.missing}):
            self.assertEqual(ai_hwaccel.find_binary(sys.executable), sys.executable)

    def test_falls_back_to_path_without_explicit_or_env(self):
        env = {k: v for k, v in os.environ.items() if k != "AI_HWACCEL_BIN"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(_runner, "_bundled_path", return_value=pathlib.Path(self.missing)), \
                mock.patch.object(_runner.shutil, "which", return_value=sys.executable):
            self.assertEqual(ai_hwaccel.find_binary(), sys.executable)


class TestTrainingMemoryArgv(unittest.TestCase):
    """--train never read --quant, so quant= is deprecated and not sent."""

    def test_no_quant_flag_sent(self):
        with mock.patch.object(_runner, "run_json", return_value=_TRAINING) as rj:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                ai_hwaccel.training_memory("70B", method="lora")
        self.assertEqual(rj.call_args[0][0], ["--train", "70B", "--method", "lora"])

    def test_quant_kwarg_warns_and_is_not_sent(self):
        with mock.patch.object(_runner, "run_json", return_value=_TRAINING) as rj:
            with self.assertWarns(DeprecationWarning):
                ai_hwaccel.training_memory("70B", method="lora", quant="int4")
        self.assertNotIn("--quant", rj.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
