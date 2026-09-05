"""
compare.py — every arm, same devices, same budget, same ruler, one table.

WHAT GETS REPORTED FOR EACH ARM, AND WHY EACH COLUMN HAS TO BE THERE

  budget       n_lines x n_points measurement operations.  Identical for all
               arms by construction — it is printed anyway, because a table
               where the reader has to trust that it is identical is a table
               the reader does not trust.

  coverage     the fraction of the 100x100 grid actually visited by at least
               one sweep.  NOT the same as the budget: nearest-cell sampling
               makes points collide, badly so near the fan's origin.  If the
               learned policy wins with equal budget and LOWER coverage, that
               is the strongest form of the result and it must be visible.
               If it wins with higher coverage, the honest reading is that
               part of the gain is simply seeing more pixels, and the paper
               says so and adds a coverage-matched arm.

  f1@0..3      the baseline's reporting convention, unchanged.  f1@1 is the
               headline; the tau sweep shows how much of the residual error
               is sub-pixel placement rather than a missed line.

  iou          strict, and low for one-pixel lines.  Reported because the
               baseline reports it.  Hiding a metric that got worse is how a
               paper stops being a measurement.

  sd           per-device standard deviation of F1@1.  Adaptive measurement
               should reduce it — the point of adapting is to handle the
               devices a fixed geometry happens to suit badly — so a gain in
               the mean with no drop in the spread is a weaker result than
               the mean alone suggests.

  headroom     (arm - rays) / (oracle - rays), at that budget.  The fraction
               of the achievable adaptive gain that this arm captured.  nan
               when the oracle does not beat `rays`, which is the honest
               output when adaptivity buys nothing at that cell rather than a
               ratio with a vanishing denominator.

The sweep adds one more, and it is the one an experimentalist reads first:

  budget       how few operations the learned policy needs to reach the ray
  saving       method's score at its best cell (8 x 60 = 480).  Reported as a
               percentage of 480 and read off the F1-vs-budget curves by
               linear interpolation, with the caveat that interpolating
               between fifteen measured cells is interpolation and is labelled
               as such in the figure.

PAIRING

All arms run on the SAME held-out devices in the SAME order, so the arms are
paired and the comparison should be tested as paired: a Wilcoxon signed-rank
over per-device F1@1 differences, not a two-sample t-test on the means.  With
50 test devices and per-device sd around 0.08, an unpaired test will call a
real 0.02 improvement insignificant.  `paired_test` does this.
"""
import csv
import os
from typing import Callable, Dict, List, Sequence

import numpy as np

from .metrics import evaluate, iou, tolerant_f1

TAUS = (0, 1, 2, 3)


def run_arm(env, policy: Callable, device_indices: Sequence[int]) -> Dict:
    """One arm over the held-out devices.  Returns metrics + per-device F1@1."""
    preds, truths, coverages, per_device = [], [], [], []
    for idx in device_indices:
        rec = env.rollout(policy, idx)
        pred = env.obs[2] > env.threshold          # CH_PROB after the episode
        preds.append(pred)
        truths.append(env.Y[idx])
        coverages.append(rec.coverage[-1])
        per_device.append(tolerant_f1(pred, env.Y[idx], 1.0)["f1"])
    m = evaluate(np.stack(preds), np.stack(truths), taus=TAUS)
    m["coverage"] = float(np.mean(coverages))
    m["budget"] = env.n_lines * env.n_points
    m["sd_f1@1"] = float(np.std(per_device))
    m["n_devices"] = len(device_indices)
    return {"metrics": m, "per_device": per_device}


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """
    Wilcoxon signed-rank on per-device F1@1, a vs b.

    Paired and non-parametric: the per-device F1 distribution is bounded and
    skewed, and the same devices are used for both arms, so this is the test
    that matches the design.  Also returns the median paired difference,
    because a p-value alone says nothing about size.
    """
    from scipy.stats import wilcoxon
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return {"p": 1.0, "median_diff": 0.0, "n_better": 0}
    stat, p = wilcoxon(d)
    return {"p": float(p), "median_diff": float(np.median(d)),
            "n_better": int((d > 0).sum())}


def headroom(arm_f1: float, rays_f1: float, oracle_f1: float) -> float:
    """
    Fraction of the adaptive headroom captured.  Undefined (nan) if the
    oracle does not beat the best fixed arm — in which case the honest
    conclusion is that adaptivity buys nothing here at this budget, and the
    paper reports that instead of a ratio with a tiny denominator.
    """
    gap = oracle_f1 - rays_f1
    return float("nan") if gap <= 1e-6 else (arm_f1 - rays_f1) / gap


def write_table(results: Dict[str, Dict], path: str,
                reference_arm: str = "rays",
                oracle_arm: str = "oracle_greedy") -> List[Dict]:
    """results: arm name -> run_arm() output.  Writes table.csv, returns rows."""
    ref = results.get(reference_arm, {}).get("metrics", {}).get("f1@1", float("nan"))
    oracle = results.get(oracle_arm, {}).get("metrics", {}).get("f1@1", float("nan"))

    rows = []
    for name, r in results.items():
        m = r["metrics"]
        row = {"arm": name, "budget": m["budget"],
               "coverage": round(m["coverage"], 5),
               **{f"f1@{t}": round(m[f"f1@{t}"], 4) for t in TAUS},
               "precision@1": round(m["precision@1"], 4),
               "recall@1": round(m["recall@1"], 4),
               "iou": round(m["iou"], 4),
               "sd_f1@1": round(m["sd_f1@1"], 4),
               "headroom": round(headroom(m["f1@1"], ref, oracle), 4)}
        if reference_arm in results and name != reference_arm:
            row.update({f"vs_rays_{k}": round(v, 5) for k, v in
                        paired_test(r["per_device"],
                                    results[reference_arm]["per_device"]).items()})
        rows.append(row)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)
    return rows


# ──────────────────────────────────────────────────────────────────────────
#  The sweep
# ──────────────────────────────────────────────────────────────────────────

# The baseline paper's grid, verbatim: 15 cells, 160 to 480 operations.
BUDGET_GRID = [(r, p) for r in (4, 5, 6, 7, 8) for p in (40, 50, 60)]

# A reduced pass may evaluate a SUBSET of those cells — never a different
# grid.  $ADQ_CELLS is "4x40,6x50,8x60"; every cell named must already be in
# BUDGET_GRID, so a short run stays a subset of the paper's sweep and the
# curve it draws is the same curve with fewer points on it.
_cells = os.environ.get("ADQ_CELLS")
if _cells:
    _want = [tuple(int(v) for v in c.strip().split("x"))
             for c in _cells.split(",") if c.strip()]
    _bad = [c for c in _want if c not in BUDGET_GRID]
    if _bad:
        raise ValueError(f"$ADQ_CELLS names cells outside the paper's grid: "
                         f"{_bad}; BUDGET_GRID is {BUDGET_GRID}")
    BUDGET_GRID = _want


def budget_saving(rays_curve, rl_curve, target_ops: int = 8 * 60):
    """
    How few operations the learned policy needs to match the ray method's
    score at `target_ops`.

    Both arguments are {operations: f1@1}.  The RL curve is interpolated
    linearly in operations, which is interpolation between fifteen measured
    cells and must be labelled as interpolation in the figure — with 40-point
    granularity between adjacent cells, the honest reading of "RL matches the
    fan from 62% of the budget" is "between 6 x 50 and 6 x 60".

    Returns (operations_needed, fraction_of_target).  operations_needed is
    nan if the learned policy never reaches the target anywhere on the grid,
    which is a legitimate outcome and should be reported as one rather than
    extrapolated past the end of the curve.
    """
    import numpy as _np
    target = rays_curve.get(target_ops)
    if target is None:
        raise KeyError(f"the ray method was not evaluated at {target_ops} ops")
    ops = sorted(rl_curve)
    f1 = [rl_curve[o] for o in ops]
    for i in range(1, len(ops)):
        if f1[i] >= target:
            if f1[i] == f1[i - 1]:
                need = float(ops[i])
            else:
                t = (target - f1[i - 1]) / (f1[i] - f1[i - 1])
                need = float(ops[i - 1] + t * (ops[i] - ops[i - 1]))
            return need, need / target_ops
    return float("nan"), float("nan")
