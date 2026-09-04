"""
devices.py — the SAME devices, and the SAME split, as the baseline paper.

Not "devices generated the same way".  The same ones.  The baseline
simulates a pool once, splits it at the device level with a fixed seed and
stores the split with the pool (`study/dataset.py`, `study/device_split.py`);
this module reads that pool and that split off disk.

Regenerating a pool here — even with the same code and the same seed — would
mean the comparison rests on an argument that two random draws are
equivalent.  Reading the same folders means it rests on nothing.  If the two
papers ever report a different F1 for the fan at 8 x 60, that is now a bug
with one cause, not a distribution to argue about.

The test devices are never touched by anything in this repository except the
final evaluation: no decoder is fitted on them, no threshold is chosen on
them, and no PPO episode is ever rolled out on them.  `load_split` returns
them in a separate object for that reason — it is harder to pass the wrong
list when the two are not interchangeable.
"""
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..eval.metrics import load_grid, load_ground_truth


@dataclass
class Split:
    train_dirs: List[str]
    test_dirs: List[str]

    def summary(self) -> str:
        return (f"{len(self.train_dirs)} train / {len(self.test_dirs)} test "
                f"devices, split at device level by the baseline repository")


def load_split(n_train: int = 500, n_test: int = 50,
               n_rays: int = 8, n_points: int = 60) -> Split:
    """
    Ask the baseline for its device pool and its stored split.

    The (n_rays, n_points) arguments do not change which devices come back —
    the baseline's own guarantee is that changing the budget changes how a
    device is measured, never which device it is.  They are passed because
    StudyConfig wants them.
    """
    from dqd.study import dataset
    from dqd.study.config import StudyConfig

    cfg = StudyConfig(n_rays=n_rays, n_points=n_points,
                      n_train=n_train, n_test=n_test)
    pool, _ = dataset.make_devices(cfg)
    train_ids, test_ids, _ = dataset.split_devices(cfg, pool)
    return Split(train_dirs=dataset.sample_dirs_for(pool, train_ids),
                 test_dirs=dataset.sample_dirs_for(pool, test_ids))


def load_arrays(sample_dirs: Sequence[str]
                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    (sensor grids, ground-truth line maps) for a list of device folders.

    Grids come back min-max normalised to [0, 1] by the baseline's load_grid,
    which is the same normalisation RayProcessor applies before firing a ray
    — so a signal value written into channel 0 here is the number the real
    pipeline would have recorded.  Ground truth is the simulator's exact
    charge-state boundary, not an edge-detected image.
    """
    grids, truths = [], []
    for d in sample_dirs:
        _, _, Z = load_grid(d)
        grids.append(Z.astype(np.float32))
        truths.append(load_ground_truth(d).astype(np.float32))
    return grids, truths


RESULTS = os.environ.get("ADQ_RESULTS", "results")
CHECKPOINTS = os.environ.get("ADQ_CHECKPOINTS", "checkpoints")
