"""Packaging metadata checks (no binary needed)."""

import pathlib
import re
import unittest

_PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"

# The first setuptools release to need each Python, from PyPI's
# Requires-Python, newest first: 83.0 needs 3.10, 75.4 needs 3.9.
_SETUPTOOLS_NEEDS = [((83, 0), (3, 10)), ((75, 4), (3, 9))]


class TestBuildRequirement(unittest.TestCase):
    def test_oldest_declared_python_can_install_the_build_backend(self):
        # A source install (pip install ./bindings/python) needs setuptools
        # from [build-system], 77 or later since 2.5.5, which no Python
        # before 3.9 can run: requires-python must not claim 3.8.
        text = _PYPROJECT.read_text()
        st = re.search(r'^requires = \["setuptools>=(\d+)(?:\.(\d+))?', text, re.M)
        py = re.search(r'^requires-python = ">=(\d+)\.(\d+)"', text, re.M)
        self.assertIsNotNone(st, "no setuptools build requirement")
        self.assertIsNotNone(py, "no requires-python")
        st_floor = (int(st.group(1)), int(st.group(2) or 0))
        py_floor = (int(py.group(1)), int(py.group(2)))
        for first, needs in _SETUPTOOLS_NEEDS:
            if st_floor >= first:
                self.assertGreaterEqual(
                    py_floor,
                    needs,
                    f"setuptools>={st_floor[0]}.{st_floor[1]} needs Python "
                    f"{needs[0]}.{needs[1]}; requires-python says "
                    f"{py_floor[0]}.{py_floor[1]}",
                )
                break


if __name__ == "__main__":
    unittest.main()
