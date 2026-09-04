"""
run_5_policy_anatomy.py — what did the policy actually learn, and is it physics?

    python scripts/run_5_policy_anatomy.py

A table saying the agent scored higher is a leaderboard entry.  What makes it
a physics paper is showing that the learned policy does something a physicist
would recognise, and testing that as a prediction rather than asserting it
from a pretty figure.

THE PREDICTION, STATED BEFORE LOOKING

A charge stability diagram is a honeycomb whose lattice vectors are set by
C1, C2 and Cm, which vary device to device.  Two things follow.

  (a) Early in an episode the lattice is unknown, and the informative
      measurement is a LONG OBLIQUE chord across the middle of the window: it
      crosses many edges of both honeycomb families at once, which is exactly
      why the baseline's oblique arms beat its axis-aligned ones.  So early
      sweeps should sit at small |rho| (near the centre) and at angles whose
      chords are long.

  (b) Once the lattice is pinned down, the reconstruction of a long straight
      edge is nearly free — the decoder can extrapolate it.  What it CANNOT
      extrapolate is the neighbourhood of the triple points, where three
      charge states meet, the interdot segments are short, and the sensor
      contrast is weakest.  So late sweeps should move OFF centre (|rho|
      grows) and toward the triple points.

Both are falsifiable, and this script measures them:

  1  |rho| against step index                    -> prediction (a) and (b)
  2  chord length against step index             -> prediction (a)
  3  distance from the chord to the nearest triple point, against step
                                                 -> prediction (b)
  4  reward per step                             -> diminishing returns; if
                                                    the last sweeps earn ~0,
                                                    the budget is too large
                                                    and the interesting
                                                    experiment is a smaller
                                                    one
  5  angular spread across devices, per step     -> is the policy adapting to
                                                    the device at all, or has
                                                    it collapsed to one fixed
                                                    geometry?  If step-0
                                                    angle has near-zero
                                                    variance across devices,
                                                    the first sweep is a
                                                    LEARNED FIXED geometry —
                                                    itself a nice result, and
                                                    the honest way to describe
                                                    it.

Writes results/anatomy/ -> anatomy.csv, anatomy.png, and overlay figures for
a handful of devices showing the chosen chords on the true honeycomb.

If none of the five come out as predicted but the F1 is still higher, say so.
An unexplained gain is a weaker paper than an explained one, and pretending
otherwise is how a reviewer finds the hole instead of you.
"""
import csv
import os

import numpy as np

from _common import banner, settings

from adaptive_dqd import policies
from adaptive_dqd.agents.ppo import PPOAgent, PPOConfig
from adaptive_dqd.config import devices as dv
from adaptive_dqd.decoder import agnostic
from adaptive_dqd.decoder.baseline_net import grid_train
from adaptive_dqd.envs import SweepEnv
from adaptive_dqd.geometry import lines

N_TRAIN, N_TEST = 500, 50
N_LINES, N_POINTS = 8, 60   # anatomy is read at the largest cell
DEVICE = "cuda"
N_OVERLAYS = 6

DECODER = os.path.join(dv.CHECKPOINTS, f"d_agn_{N_LINES}x{N_POINTS}.pt")
AGENT = os.path.join(dv.CHECKPOINTS, f"ppo_{N_LINES}x{N_POINTS}.pt")
OUT = os.path.join(dv.RESULTS, "anatomy")


def triple_points(truth: np.ndarray) -> np.ndarray:
    """
    (M, 2) pixels where three or more line branches meet.

    A junction pixel of a one-pixel-wide line map has more line neighbours
    than a pixel in the middle of a straight run.  Counting 8-neighbours and
    keeping >= 3 is the standard crossing-number proxy; it over-detects a
    little on diagonal staircases, which biases the measured distance UP and
    therefore makes prediction (b) harder to confirm, not easier.
    """
    from scipy.ndimage import convolve
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    line = (truth > 0.5).astype(np.uint8)
    deg = convolve(line, k, mode="constant")
    rc = np.argwhere((line == 1) & (deg >= 3))
    return rc


def chord_length(rho: float, theta: float) -> float:
    pts = lines.chord(rho, theta, 2)
    return 0.0 if not len(pts) else float(np.linalg.norm(pts[-1] - pts[0]))


def dist_to_triple(rho, theta, tps, shape) -> float:
    """Median distance, in pixels, from the sweep to the nearest triple point."""
    if not len(tps):
        return float("nan")
    rc = lines.sweep(rho, theta, 200, shape)
    d = np.linalg.norm(rc[:, None, :] - tps[None, :, :], axis=-1)
    return float(np.median(d.min(axis=1)))


if __name__ == "__main__":
    banner("policy anatomy — testing the physics prediction")
    settings(agent=AGENT, n_test=N_TEST, out=OUT)
    os.makedirs(OUT, exist_ok=True)

    split = dv.load_split(N_TRAIN, N_TEST, N_LINES, N_POINTS)
    grids, truths = dv.load_arrays(split.test_dirs)
    net, meta = grid_train.load(DECODER)
    env = SweepEnv(grids, truths, agnostic.as_callable(net, DEVICE),
                   threshold=meta["threshold"], n_lines=N_LINES,
                   n_points=N_POINTS)
    agent = PPOAgent(PPOConfig(), device=DEVICE).load(AGENT)
    policy = policies.from_agent(agent, deterministic=True)

    rows, records = [], []
    for i in range(len(grids)):
        rec = env.rollout(policy, i)
        records.append(rec)
        tps = triple_points(truths[i])
        for k, (rho, theta) in enumerate(rec.actions):
            rows.append({
                "device": i, "step": k, "rho": rho, "abs_rho": abs(rho),
                "theta_deg": np.degrees(theta),
                "chord_length": chord_length(rho, theta),
                "dist_to_triple_px": dist_to_triple(rho, theta, tps,
                                                    env.shape),
                "reward": rec.rewards[k], "f1_after": rec.f1[k + 1],
                "coverage_after": rec.coverage[k]})

    with open(os.path.join(OUT, "anatomy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    banner("per-step means over held-out devices")
    print(f"  {'step':>4} {'|rho|':>8} {'length':>8} {'d_triple':>9} "
          f"{'reward':>8} {'F1@1':>7} {'sd(theta)':>10}")
    for k in range(N_LINES):
        s = [r for r in rows if r["step"] == k]
        theta_sd = float(np.std([r["theta_deg"] for r in s]))
        print(f"  {k:>4} {np.mean([r['abs_rho'] for r in s]):>8.3f} "
              f"{np.mean([r['chord_length'] for r in s]):>8.3f} "
              f"{np.nanmean([r['dist_to_triple_px'] for r in s]):>9.2f} "
              f"{np.mean([r['reward'] for r in s]):>8.4f} "
              f"{np.mean([r['f1_after'] for r in s]):>7.4f} "
              f"{theta_sd:>10.2f}")
    print("\n  sd(theta) near 0 at a step means that sweep is a LEARNED FIXED "
          "geometry,\n  not an adaptive one — report it that way if so.")
