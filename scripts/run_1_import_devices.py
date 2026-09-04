"""
run_1_import_devices.py — check the link to the baseline, print the split.

    python scripts/run_1_import_devices.py

Does no work.  It exists because every number this repository produces rests
on the claim that the devices, the split, the metric and the network came
from the baseline repository unchanged, and that claim should be checkable in
two seconds rather than inferred from an import that happened to succeed.

The paths it prints belong verbatim in the paper's reproducibility statement.
"""
from _common import banner, settings

from adaptive_dqd.config import devices as dv
from adaptive_dqd.eval.metrics import BASELINE_SRC

N_TRAIN, N_TEST = 500, 50
N_LINES, N_POINTS = 8, 60

if __name__ == "__main__":
    banner("baseline link check")
    settings(baseline_src=BASELINE_SRC, n_train=N_TRAIN, n_test=N_TEST,
             budget=f"{N_LINES} x {N_POINTS} = {N_LINES * N_POINTS} ops")

    split = dv.load_split(N_TRAIN, N_TEST, N_LINES, N_POINTS)
    print(f"\n  {split.summary()}")
    print(f"  first train device: {split.train_dirs[0]}")
    print(f"  first test  device: {split.test_dirs[0]}")

    overlap = set(split.train_dirs) & set(split.test_dirs)
    assert not overlap, f"device leak: {len(overlap)} devices on both sides"
    print("  no device appears on both sides of the split.")

    grids, truths = dv.load_arrays(split.test_dirs[:3])
    print(f"  grid shape {grids[0].shape}, "
          f"{100 * truths[0].mean():.2f}% of pixels are transition lines")
