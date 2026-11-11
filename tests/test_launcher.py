"""The build launcher: `import mojopymodes` with no compiled library present.

`mojopymodes._lib` compiles `src/capi.mojo` on first import when `dist/` is
empty. That path is invisible to the rest of the suite, which runs with a
library already built, so this file drives the launcher itself: it must find
a toolchain that can see `std`, and it must build and load a working library
from a clean output directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

from mojopymodes import _lib


def _mojo() -> str | None:
    return _lib.mojo_command()[0] if _lib.mojo_command() else None


def test_mojo_command_is_available():
    """Whatever launcher is chosen, the command has to exist."""
    cmd = _lib.mojo_command()
    assert cmd
    assert shutil.which(cmd[0]) or os.path.exists(cmd[0]), cmd


def test_toolchain_env_names_a_package_root_when_mojo_is_a_conda_binary():
    """A `mojo` from a conda prefix gets that prefix's MODULAR_HOME.

    The compiler only finds `std` and `max` through MODULAR_HOME, so a
    launcher that starts `mojo` without it fails on every file; that is the
    bug this guards. `mojo` itself must live in `<prefix>/bin`.
    """
    mojo = _mojo()
    if mojo is None or os.path.basename(mojo) != "mojo":
        pytest.skip("launcher is pixi, not a mojo binary")
    home = _lib._toolchain_home(mojo)
    if home is None:
        pytest.skip("mojo is not from a conda prefix")
    assert os.path.isdir(os.path.join(os.path.dirname(home), "max"))


def test_a_clean_build_produces_a_loadable_library():
    """Build into an empty directory and call one export from it."""
    pytest.importorskip("numpy")
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "libmojo-pymodes.so")
        old = _lib.LIB
        try:
            _lib.LIB = target
            _lib.build(force=True)
            assert os.path.isfile(target)
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import ctypes,sys;"
                    "k=ctypes.CDLL(sys.argv[1]);"
                    "k.mp_gray2int.restype=ctypes.c_int64;"
                    "print(k.mp_gray2int(ctypes.c_int64(0b1010)))",
                    target,
                ],
                capture_output=True,
                text=True,
                env={k: v for k, v in os.environ.items() if k != "MOJOPYMODES_LIB"},
                timeout=600,
            )
        finally:
            _lib.LIB = old
        assert probe.returncode == 0, probe.stderr
        # Gray 1010 -> 1100 (12).
        assert probe.stdout.strip() == "12", probe.stdout


def test_mojopymodes_lib_is_used_as_given(monkeypatch, tmp_path):
    """An explicit library path is loaded, not rebuilt, and read at call time.

    A packaged install has no `src/` next to the package; the launcher must not
    try to walk it, and an override set after import must still take effect.
    """
    target = tmp_path / "lib.so"
    shutil.copyfile(_lib.build(), target)
    monkeypatch.setenv("MOJOPYMODES_LIB", str(target))
    assert _lib.library_path() == str(target)
    assert _lib.build() == str(target)


def test_missing_library_and_no_sources_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("MOJOPYMODES_LIB", raising=False)
    monkeypatch.setattr(_lib, "SRC", str(tmp_path / "nowhere"))
    monkeypatch.setattr(_lib, "LIB", str(tmp_path / "dist" / "lib.so"))
    with pytest.raises(_lib.BuildError, match="no Mojo sources"):
        _lib.build()
    monkeypatch.setenv("MOJOPYMODES_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(_lib.BuildError, match="does not exist"):
        _lib.build()
