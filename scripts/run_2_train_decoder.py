"""
run_2_train_decoder.py — the SHARED, geometry-agnostic decoder D_agn.

    python scripts/run_2_train_decoder.py

Trained once, on chords drawn at random from the full action space and at
every partial budget from 1 to n_lines, then frozen.  Every arm — the fan,
the greedy heuristics, the oracle and the agent — is scored through these
same weights, so a difference between arms is a difference in where the
sweeps went and cannot be a difference in the reconstruction network.

See src/adaptive_dqd/decoder/agnostic.py for why this, and why the MATCHED
column exists alongside it.  Training constants are the baseline's and are
deliberately not settable here.
"""
import os

from _common import banner, env_int, settings

from adaptive_dqd.config import devices as dv
from adaptive_dqd.decoder import agnostic
from adaptive_dqd.decoder.baseline_net import grid_train

N_TRAIN, N_TEST = 500, 50
# Subsample the TRAIN side only, after the split, so a reduced pass never
# moves a device across the split.  0 = use all of them.
TRAIN_SUBSET = env_int("ADQ_TRAIN_SUBSET", 0)
N_LINES, N_POINTS = 8, 60
EPOCHS = env_int("ADQ_EPOCHS", 50)
REPEATS = env_int("ADQ_REPEATS", 4)           # random geometries per device
SEED = 0

OUT = os.path.join(dv.CHECKPOINTS, f"d_agn_{N_LINES}x{N_POINTS}.pt")

if __name__ == "__main__":
    banner("shared geometry-agnostic decoder")
    settings(budget=f"{N_LINES} x {N_POINTS}", epochs=EPOCHS,
             train_subset=TRAIN_SUBSET or "all",
             geometries_per_device=REPEATS, out=OUT)

    split = dv.load_split(N_TRAIN, N_TEST, N_LINES, N_POINTS)
    train_dirs = split.train_dirs
    if TRAIN_SUBSET:
        train_dirs = train_dirs[:TRAIN_SUBSET]
        print(f"  REDUCED PASS: {len(train_dirs)} of "
              f"{len(split.train_dirs)} training devices")
    grids, truths = dv.load_arrays(train_dirs)            # TRAIN ONLY
    print(f"  {len(grids)} training devices -> "
          f"{len(grids) * REPEATS} random-geometry examples")

    net, threshold, _ = agnostic.train_agnostic(
        grids, truths, N_LINES, N_POINTS, epochs=EPOCHS, repeats=REPEATS,
        seed=SEED)

    os.makedirs(dv.CHECKPOINTS, exist_ok=True)
    grid_train.save(net, threshold, OUT, N_LINES, N_POINTS)
    print(f"\n  threshold {threshold:.3f} (chosen on a validation split of "
          f"the TRAINING devices)\n  saved -> {OUT}")
