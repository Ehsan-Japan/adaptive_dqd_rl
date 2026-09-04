"""
metrics.py — the baseline's metric code, imported, not reimplemented.

The comparison in the paper is only as trustworthy as the claim that both
methods were scored by the same ruler.  The cheapest way to make that claim
false is to reimplement F1@tau here, get the distance-transform convention or
the per-device averaging subtly different, and report a 0.02 difference that
is the metric and not the method.

So this module does not define the metric.  It imports

    dqd.ml.grid_metrics

from the baseline repository — checked out as the `baseline/` submodule, or
pointed at by $DQD_BASELINE — and re-exports it.  If the import fails the
error says how to fix it rather than silently falling back to a lookalike.

Run `python scripts/run_1_import_devices.py --check` to verify the import
resolves and to print the resolved path; that path belongs in the paper's
reproducibility statement.
"""
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]


def _baseline_src() -> Path:
    """Where the baseline repo's src/ lives.  Submodule first, env var second."""
    env = os.environ.get("DQD_BASELINE")
    candidates = [Path(env) / "src"] if env else []
    candidates += [_REPO / "baseline" / "src", _REPO.parent /
                   "static_RBC_test_noise_8_15_ML_edited" / "src"]
    for c in candidates:
        if (c / "dqd" / "ml" / "grid_metrics.py").is_file():
            return c
    raise ImportError(
        "The baseline repository was not found, and this project deliberately "
        "refuses to reimplement its metrics.\n"
        "Fix it with either:\n"
        "  git submodule update --init          # uses baseline/\n"
        "  export DQD_BASELINE=/path/to/static_RBC_test_noise_8_15_ML_edited\n"
        f"Looked in: {', '.join(str(c) for c in candidates)}")


BASELINE_SRC = _baseline_src()
if str(BASELINE_SRC) not in sys.path:
    sys.path.insert(0, str(BASELINE_SRC))

# Deliberately torch-free.  test_geometry.py asserts the load-bearing claim
# of the whole comparison — that the action space contains the fan — and it
# should be runnable on a laptop with numpy and scipy, in CI, before anyone
# waits on a GPU.  The torch-side re-exports live in decoder/baseline_net.py.
from dqd.ml.grid_metrics import evaluate, iou, tolerant_f1   # noqa: E402,F401
from dqd.ml.ray_peaks import load_grid, load_ground_truth    # noqa: E402,F401

__all__ = ["evaluate", "iou", "tolerant_f1",
           "load_grid", "load_ground_truth", "BASELINE_SRC"]
