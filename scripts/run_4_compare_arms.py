"""
run_4_compare_arms.py — rays vs deep RL, at every cell of the budget sweep.

    python scripts/run_4_compare_arms.py

Not one number at one budget.  Fifteen cells — 4 to 8 sweeps of 40 to 60
points, 160 to 480 measurement operations — exactly the grid the companion
paper reports, on exactly the same held-out devices, through exactly the same
metric code.

WHY THE SWEEP AND NOT A SINGLE COMPARISON

A single cell answers "is the learned policy better at 8 x 60".  The sweep
answers the question an experimentalist actually has, which is about time on
the fridge:

    how few measurements does the learned policy need to reach the
    reconstruction the ray method gets from its full budget?

If the RL curve at 6 x 50 sits at the ray curve's 8 x 60 value, the claim is
"the same reconstruction from 62% of the acquisition" — a claim about cost,
in the units the experiment is actually paid for.  It is also far more robust
than a difference in the third decimal at one budget: a curve that sits above
another curve at fifteen cells is not a seed.

The companion paper's own numbers are the reference points:

    4 x 40   1.57% coverage   F1@1 0.672
    8 x 40   3.11%            0.810
    8 x 50   3.88%            0.835
    8 x 60   4.66%            0.849   <- the target to match from fewer ops

ARMS

    rays                the ray-based method, rebuilt per cell (its angles
                        depend on n_rays) — the thing being compared against
    rl_ppo              the learned policy.  ONE agent, all fifteen cells.
    random_lines        control: the score an untrained policy gets
    uncertainty_greedy  control: non-learned adaptive.  If it matches rl_ppo,
                        the paper is about adaptive measurement and not about
                        deep RL, and it should say so.
    oracle_greedy       upper bound, ground truth in hand.  Supplies the
                        headroom denominator.  Not a method — label it as a
                        bound in the figure itself, not only in the caption.

Writes results/sweep/ -> sweep.csv (one row per arm per cell), curves.csv,
table_<cell>.csv and per_device_<cell>.csv.  The oracle is the expensive arm;
set RUN_ORACLE = False for a quick pass while iterating, and never report a
headroom number from a pass where it was off.
"""
import csv
import os

import numpy as np

from _common import banner, settings, torch_device

from adaptive_dqd import policies
from adaptive_dqd.agents.ppo import PPOAgent, PPOConfig
from adaptive_dqd.config import devices as dv
from adaptive_dqd.decoder import agnostic
from adaptive_dqd.decoder.baseline_net import grid_train
from adaptive_dqd.envs import SweepEnv
from adaptive_dqd.eval import compare

N_TRAIN, N_TEST = 500, 50
MAX_LINES, MAX_POINTS = 8, 60
DEVICE = torch_device()
RUN_ORACLE = True
TARGET_CELL = (8, 60)          # the ray method's best cell — the bar to clear

DECODER = os.path.join(dv.CHECKPOINTS, f"d_agn_{MAX_LINES}x{MAX_POINTS}.pt")
AGENT = os.path.join(dv.CHECKPOINTS, f"ppo_{MAX_LINES}x{MAX_POINTS}.pt")
OUT = os.path.join(dv.RESULTS, "sweep")


def main():
    banner("rays vs deep RL — 15 budget cells, same devices, same metric")
    settings(cells=len(compare.BUDGET_GRID), n_test=N_TEST, decoder=DECODER,
             agent=AGENT, oracle="on" if RUN_ORACLE else "OFF", out=OUT)
    os.makedirs(OUT, exist_ok=True)

    split = dv.load_split(N_TRAIN, N_TEST, MAX_LINES, MAX_POINTS)
    grids, truths = dv.load_arrays(split.test_dirs)        # TEST ONLY, once
    net, meta = grid_train.load(DECODER)
    env = SweepEnv(grids, truths, agnostic.as_callable(net, DEVICE),
                   threshold=meta["threshold"], n_lines=MAX_LINES,
                   n_points=MAX_POINTS)
    agent = PPOAgent(PPOConfig(), device=DEVICE).load(AGENT)

    arms = {
        "rays": policies.rays(),                    # budget read per episode
        "rl_ppo": policies.from_agent(agent, deterministic=True),
        "random_lines": policies.random_lines(seed=0),
        "uncertainty_greedy": policies.uncertainty_greedy(),
    }
    if RUN_ORACLE:
        arms["oracle_greedy"] = policies.oracle_greedy()

    idx = list(range(len(grids)))
    all_rows, curves = [], {a: {} for a in arms}

    for n_lines, n_points in compare.BUDGET_GRID:
        ops = n_lines * n_points
        env.n_lines, env.n_points = n_lines, n_points
        print(f"\n--- {n_lines} x {n_points} = {ops} operations " + "-" * 26)

        results = {}
        for name, policy in arms.items():
            results[name] = compare.run_arm(env, policy, idx)
            m = results[name]["metrics"]
            curves[name][ops] = m["f1@1"]
            print(f"    {name:<20} F1@1 {m['f1@1']:.4f}  sd {m['sd_f1@1']:.4f}"
                  f"  coverage {100 * m['coverage']:.2f}%")

        cell = f"{n_lines}x{n_points}"
        rows = compare.write_table(results,
                                   os.path.join(OUT, f"table_{cell}.csv"))
        for r in rows:
            r.update(n_lines=n_lines, n_points=n_points, operations=ops)
        all_rows += rows

        with open(os.path.join(OUT, f"per_device_{cell}.csv"), "w",
                  newline="") as f:
            w = csv.writer(f)
            w.writerow(["device_index"] + list(results))
            for i in idx:
                w.writerow([i] + [results[a]["per_device"][i] for a in results])

    with open(os.path.join(OUT, "sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f,
                           fieldnames=sorted({k for r in all_rows for k in r}))
        w.writeheader()
        w.writerows(all_rows)

    # ── the sentence the abstract quotes ─────────────────────────────────
    banner("budget saving")
    target_ops = TARGET_CELL[0] * TARGET_CELL[1]
    target_f1 = curves["rays"][target_ops]
    print(f"  the ray method at {TARGET_CELL[0]} x {TARGET_CELL[1]} "
          f"({target_ops} ops): F1@1 = {target_f1:.4f}")
    for name in ("rl_ppo", "uncertainty_greedy"):
        need, frac = compare.budget_saving(curves["rays"], curves[name],
                                           target_ops)
        if np.isnan(need):
            print(f"  {name:<20} never reaches it on this grid "
                  f"(best {max(curves[name].values()):.4f}) — report that; "
                  f"do not extrapolate past the last cell")
        else:
            print(f"  {name:<20} reaches it at ~{need:.0f} operations = "
                  f"{100 * frac:.0f}% of the budget (interpolated between "
                  f"measured cells — say so in the caption)")

    with open(os.path.join(OUT, "curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["operations"] + list(curves))
        for ops in sorted(curves["rays"]):
            w.writerow([ops] + [round(curves[a][ops], 4) for a in curves])

    banner("F1@1 against budget")
    print(f"  {'ops':>5} " + " ".join(f"{a:>19}" for a in curves))
    for ops in sorted(curves["rays"]):
        print(f"  {ops:>5} " + " ".join(f"{curves[a][ops]:>19.4f}"
                                        for a in curves))


if __name__ == "__main__":
    main()
