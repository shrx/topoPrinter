"""Shared test setup: import path, and a hard memory cap on the test process.

The cap is applied at conftest IMPORT time -- before pytest imports any test module,
and therefore before numpy/rasterio/trimesh are pulled in -- so it bounds the entire
run rather than just the test bodies. A runaway allocation then dies with a
MemoryError naming the test, instead of driving the machine into the OOM killer.

This matters here because the pipeline sizes its arrays from DATA: a DEM read at the
wrong downsample, a resampled ring whose perimeter blew up, or a mesh built at full
grid resolution can each ask for tens of gigabytes from a fixture that looks small.
Those are exactly the bugs the suite exists to catch, so the suite must survive
finding one.

``RLIMIT_AS`` caps the virtual address space, which is the only rlimit Linux still
enforces for ordinary allocations (``RLIMIT_RSS`` has been a no-op for years). It
counts file-backed mmaps too, so GDAL reading a raster shows up against the cap --
the default below is set with that included.

Override for a genuinely large run, or disable it:

    TOPOPRINTER_TEST_MEM_MB=8192 pytest
    TOPOPRINTER_TEST_MEM_MB=0 pytest        # no cap
"""

import os
import resource
import sys


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


DEFAULT_MEM_MB = 4096
_ENV_VAR = "TOPOPRINTER_TEST_MEM_MB"

# Set by _apply_memory_cap so the header can report what is actually in force.
_cap_mb = None
_cap_note = "not supported on this platform"


def _apply_memory_cap():
    """Cap the test process's address space. Returns (mb, note)."""
    if not sys.platform.startswith("linux"):
        return None, "not supported on this platform"

    raw = os.environ.get(_ENV_VAR)
    try:
        mb = DEFAULT_MEM_MB if raw is None else int(raw)
    except ValueError:
        return None, f"ignored invalid {_ENV_VAR}={raw!r}"

    if mb <= 0:
        return None, f"disabled via {_ENV_VAR}={mb}"

    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = mb * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)        # never try to raise our own hard limit
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ValueError, OSError) as exc:
        return None, f"could not apply: {exc}"

    source = "default" if raw is None else _ENV_VAR
    return limit // (1024 * 1024), source


_cap_mb, _cap_note = _apply_memory_cap()


def pytest_report_header(config):
    """Make the cap visible in every run's header, capped or not."""
    if _cap_mb is None:
        return f"memory cap: NONE ({_cap_note})"
    return (f"memory cap: {_cap_mb} MB address space ({_cap_note}); "
            f"override with {_ENV_VAR}=<mb>, 0 to disable")
