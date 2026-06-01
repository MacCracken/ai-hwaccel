"""Platform-specific wheel build.

The wheel bundles a prebuilt `ai-hwaccel` binary (+ VERSION + data/) under
`ai_hwaccel/_bin/`, so it is NOT a pure-python wheel — it must be tagged
for the target platform. There is no compiled Python extension, so the
wheel is python-agnostic: `py3-none-<platform>`.

The platform tag defaults to the build host's, but is overridable via
`AIH_WHEEL_PLAT` so cross-staged binaries (e.g. a macOS or Windows binary
built on its own machine, copied into _bin/) can be packaged with the
correct tag. See scripts/build_wheel.sh.

Metadata lives in pyproject.toml; this file only customizes the wheel tag.
"""

import os

from setuptools import setup
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 vendors bdist_wheel
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools: from the wheel package
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    """Mark the distribution as non-pure so a platform wheel is built."""

    def has_ext_modules(self):  # noqa: D401
        return True


class bdist_wheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False  # platform wheel, not py3-none-any

    def get_tag(self):
        _python, _abi, plat = super().get_tag()
        plat = os.environ.get("AIH_WHEEL_PLAT", plat)
        # python-agnostic (no extension), platform-specific (bundled binary)
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
