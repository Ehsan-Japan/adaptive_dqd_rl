"""
baseline_net.py — the reconstruction network and its trainer, imported.

Same reasoning as eval/metrics.py: the decoder in this repository must be
byte-for-byte the baseline's `RayToLinesNet` and its training routine must be
byte-for-byte `grid_train`, or the comparison acquires a second uncontrolled
variable and stops being a comparison of measurement policies.

Nothing here is redefined.  If you find yourself wanting to change the
architecture, the loss, the optimiser or the threshold rule, change it in the
baseline repository — then both papers move together and the numbers in each
stay comparable with the numbers in the other.
"""
from ..eval.metrics import BASELINE_SRC     # noqa: F401  (sets sys.path)

from dqd.ml.grid_model import RayToLinesNet, WIDTH   # noqa: E402,F401
from dqd.ml import grid_train                        # noqa: E402,F401

__all__ = ["RayToLinesNet", "WIDTH", "grid_train", "BASELINE_SRC"]
