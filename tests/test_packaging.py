"""
Packaging tests.

These tests exercise SynHydro the way a ``pip`` user sees it: they build a
wheel with hatchling, install it into an empty target directory, and import
it from a subprocess whose working directory is outside the repository. The
rest of the suite runs against the editable checkout, which cannot catch a
data file that is missing from the wheel.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import synhydro

REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_DATA_FILES = [
    "usgs_daily_streamflow_cms.csv",
    "usgs_monthly_streamflow_cms.csv",
]

# Executed in a subprocess against the installed wheel. It prints one JSON
# line so the parent can assert on the result without importing synhydro
# from the repository.
CHILD_SCRIPT = """
import importlib.metadata
import json
import logging

logging.disable(logging.CRITICAL)

import synhydro
from synhydro.utils import get_example_data_path, list_example_datasets

daily = synhydro.load_example_data()
monthly = synhydro.load_example_data("usgs_monthly_streamflow_cms")

print(json.dumps({
    "package_file": synhydro.__file__,
    "attr_version": synhydro.__version__,
    "metadata_version": importlib.metadata.version("synhydro"),
    "datasets": list_example_datasets(),
    "daily_path": str(get_example_data_path("usgs_daily_streamflow_cms.csv")),
    "daily_shape": list(daily.shape),
    "daily_columns": list(daily.columns),
    "daily_index_is_datetime": type(daily.index).__name__ == "DatetimeIndex",
    "monthly_shape": list(monthly.shape),
}))
"""


def _run(cmd, **kwargs):
    """Run a command and fail with its output if it exits non-zero."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, **kwargs
    )
    assert result.returncode == 0, (
        f"command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    """Build the wheel and pip-install it into a clean target directory.

    Returns a dict with the wheel path, the install target, and an empty
    working directory outside the repository. Skips when hatchling or pip
    is not importable, or when the tests are not running from a checkout.
    """
    pytest.importorskip("hatchling")
    pytest.importorskip("pip")
    if not (REPO_ROOT / "pyproject.toml").exists():
        pytest.skip("pyproject.toml not found; not running from a source checkout")

    root = tmp_path_factory.mktemp("wheel")
    dist_dir = root / "dist"
    target_dir = root / "site"
    workdir = root / "elsewhere"
    workdir.mkdir()

    _run(
        [
            sys.executable, "-m", "hatchling", "build",
            "-t", "wheel", "-d", str(dist_dir),
        ],
        cwd=REPO_ROOT,
    )
    wheels = sorted(dist_dir.glob("synhydro-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"

    _run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-deps", "--no-index", "--target", str(target_dir), str(wheels[0]),
        ],
        cwd=workdir,
    )
    return {"wheel": wheels[0], "target": target_dir, "workdir": workdir}


class TestWheelContents:
    """The built wheel carries the example data."""

    def test_wheel_contains_example_data(self, installed_wheel):
        with zipfile.ZipFile(installed_wheel["wheel"]) as whl:
            names = set(whl.namelist())
        assert "synhydro/data/__init__.py" in names
        for filename in EXAMPLE_DATA_FILES:
            assert f"synhydro/data/{filename}" in names

    def test_installed_target_has_example_data(self, installed_wheel):
        data_dir = installed_wheel["target"] / "synhydro" / "data"
        for filename in EXAMPLE_DATA_FILES:
            assert (data_dir / filename).is_file()


class TestInstalledPackage:
    """The installed package works from a directory outside the repository."""

    def test_load_example_data_from_installed_wheel(self, installed_wheel):
        target_dir = installed_wheel["target"].resolve()
        env = dict(os.environ)
        env["PYTHONPATH"] = str(target_dir)

        result = _run(
            [sys.executable, "-c", CHILD_SCRIPT],
            cwd=installed_wheel["workdir"],
            env=env,
        )
        info = json.loads(result.stdout.strip().splitlines()[-1])

        # The subprocess imported the installed copy, not the checkout
        package_file = Path(info["package_file"]).resolve()
        assert package_file.is_relative_to(target_dir), package_file
        daily_path = Path(info["daily_path"]).resolve()
        assert daily_path.is_relative_to(target_dir), daily_path

        assert info["datasets"] == EXAMPLE_DATA_FILES
        assert info["daily_shape"] == [29340, 4]
        assert info["monthly_shape"] == [965, 4]
        assert info["daily_index_is_datetime"]
        assert info["daily_columns"] == [
            "USGS-01434000",
            "USGS-01438500",
            "USGS-01440000",
            "USGS-01463500",
        ]
        assert info["metadata_version"] == info["attr_version"]
        assert info["attr_version"] == synhydro.__version__


class TestVersion:
    """The version is single-sourced from synhydro.__version__."""

    def test_version_matches_installed_metadata(self):
        installed = importlib.metadata.version("synhydro")
        assert synhydro.__version__ == installed, (
            f"synhydro.__version__ is {synhydro.__version__} but the installed "
            f"metadata says {installed}; reinstall with 'pip install -e .'"
        )
