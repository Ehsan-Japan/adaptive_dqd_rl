"""
run_6_figures.py — the figures, drawn from the CSVs the earlier stages wrote.

    python scripts/run_6_figures.py

This stage computes nothing.  Every number it draws is read back off disk
from results/sweep/ and results/anatomy/, so a figure cannot disagree with
the table beside it, and a figure cannot exist for a run that was never made.
The one exception is the geometry overlay, which re-rolls two policies on a
handful of test devices in order to draw where their sweeps went; it re-uses
the same env, decoder and threshold as run_4.

Writes results/figures/.

The oracle is drawn dashed and labelled AS A BOUND inside the axes, not only
in the caption, because a reader who sees five solid curves will read it as a
sixth method.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402

from _common import banner, settings, torch_device     # noqa: E402

from adaptive_dqd import policies                      # noqa: E402
from adaptive_dqd.agents.ppo import PPOAgent, PPOConfig  # noqa: E402
from adaptive_dqd.config import devices as dv          # noqa: E402
from adaptive_dqd.decoder import agnostic              # noqa: E402
from adaptive_dqd.decoder.baseline_net import grid_train  # noqa: E402
from adaptive_dqd.envs import SweepEnv                 # noqa: E402
from adaptive_dqd.geometry import lines                # noqa: E402

N_TRAIN, N_TEST = 500, 50
MAX_LINES, MAX_POINTS = 8, 60
DEVICE = torch_device()

SWEEP = os.path.join(dv.RESULTS, "sweep")
ANATOMY = os.path.join(dv.RESULTS, "anatomy")
OUT = os.path.join(dv.RESULTS, "figures")
DECODER = os.path.join(dv.CHECKPOINTS, f"d_agn_{MAX_LINES}x{MAX_POINTS}.pt")
AGENT = os.path.join(dv.CHECKPOINTS, f"ppo_{MAX_LINES}x{MAX_POINTS}.pt")

# The companion paper's own reported cells, for the reference marks.  Quoted
# from its README, not recomputed here, and drawn as hollow stars so they can
# never be confused with this run's own points.
COMPANION = {160: 0.672, 320: 0.810, 400: 0.835, 480: 0.849}

STYLE = {
    "rays":               dict(color="#1f77b4", marker="o", ls="-"),
    "rl_ppo":             dict(color="#d62728", marker="s", ls="-"),
    "uncertainty_greedy": dict(color="#2ca02c", marker="^", ls="-"),
    "random_lines":       dict(color="#7f7f7f", marker="v", ls=":"),
    "oracle_greedy":      dict(color="#9467bd", marker="D", ls="--"),
}
LABEL = {
    "rays": "rays (baseline geometry)",
    "rl_ppo": "rl_ppo (learned)",
    "uncertainty_greedy": "uncertainty_greedy (heuristic control)",
    "random_lines": "random_lines (control)",
    "oracle_greedy": "oracle_greedy - BOUND, not a method",
}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def present_arms(rows):
    return [a for a in STYLE if any(r["arm"] == a for r in rows)]


def fig_budget_curve():
    rows = read_csv(os.path.join(SWEEP, "sweep.csv"))
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    for arm in present_arms(rows):
        pts = sorted((int(x["operations"]), float(x["f1@1"]))
                     for x in rows if x["arm"] == arm)
        ax.plot([o for o, _ in pts], [f for _, f in pts], label=LABEL[arm],
                ms=5, lw=1.6, **STYLE[arm])
    ax.plot(list(COMPANION), list(COMPANION.values()), "k*", ms=12,
            mfc="none", ls="none", label="companion paper, as published")
    bar = COMPANION[480]
    ax.axhline(bar, color="k", lw=0.9, ls="-.", alpha=0.65)
    ax.annotate(f"the bar: rays at 8x60 = {bar:.3f} (companion paper)",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5)
    ax.set_xlabel("measurement operations  (n_lines x n_points)")
    ax.set_ylabel("F1@1   (held-out devices, shared decoder)")
    ax.set_title("Reconstruction against acquisition cost", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_budget_curve.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_training():
    hist_path = AGENT.replace(".pt", "_history.json")
    if not os.path.isfile(hist_path):
        return None
    with open(hist_path) as f:
        h = json.load(f)
    it = [x["iteration"] for x in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.7))
    a1.plot(it, [x["mean_return"] for x in h], color="#d62728", lw=1.2)
    a1.set_xlabel("PPO iteration")
    a1.set_ylabel("mean episode return")
    a1.set_title("return = final F1@1 minus empty-measurement F1@1",
                 fontsize=9)
    a1.grid(alpha=0.25)
    a2.plot(it, [x["entropy"] for x in h], color="#1f77b4", lw=1.2)
    a2.axhline(0, color="k", lw=0.8, ls=":")
    a2.set_xlabel("PPO iteration")
    a2.set_ylabel("policy entropy")
    a2.set_title("entropy to 0 = collapsed to a LEARNED FIXED geometry;\n"
                 "entropy flat = nothing was learned", fontsize=9)
    a2.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_training.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_headroom_and_significance():
    rows = [r for r in read_csv(os.path.join(SWEEP, "sweep.csv"))
            if r["arm"] == "rl_ppo"]
    if not rows or "headroom" not in rows[0]:
        return None
    rows.sort(key=lambda r: int(r["operations"]))
    ops = [int(r["operations"]) for r in rows]
    x = np.arange(len(ops))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    head = [float(r["headroom"]) for r in rows]
    a1.bar(x, head, color="#d62728", alpha=0.85)
    a1.axhline(1.0, color="#9467bd", ls="--", lw=1,
               label="oracle_greedy bound (greedy, uses ground truth)")
    a1.axhline(0.0, color="#1f77b4", ls="-", lw=1, label="rays")
    a1.set_xticks(x)
    a1.set_xticklabels([str(o) for o in ops])
    a1.set_xlabel("operations")
    a1.set_ylabel("(rl - rays) / (oracle - rays)")
    a1.set_title("headroom captured by the learned policy", fontsize=9)
    a1.legend(fontsize=7)
    a1.grid(alpha=0.25, axis="y")

    if "vs_rays_median_diff" in rows[0]:
        med = [float(r["vs_rays_median_diff"]) for r in rows]
        pv = [float(r["vs_rays_p"]) for r in rows]
        a2.bar(x, med,
               color=["#d62728" if p < 0.05 else "#c9c9c9" for p in pv])
        for xi, (m, p) in enumerate(zip(med, pv)):
            a2.annotate(f"p={p:.1g}", (xi, m), ha="center", fontsize=6.5,
                        va="bottom" if m >= 0 else "top")
        a2.axhline(0, color="k", lw=0.8)
        a2.set_xticks(x)
        a2.set_xticklabels([str(o) for o in ops])
        a2.set_xlabel("operations")
        a2.set_ylabel("median paired F1@1 difference")
        a2.set_title("rl_ppo minus rays, Wilcoxon signed-rank\n"
                     "read as a family, not as N chances at p<0.05",
                     fontsize=9)
        a2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT, "fig3_headroom.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_coverage_and_spread():
    rows = read_csv(os.path.join(SWEEP, "sweep.csv"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for arm in present_arms(rows):
        pts = sorted((100 * float(x["coverage"]), float(x["f1@1"]))
                     for x in rows if x["arm"] == arm)
        a1.plot([c for c, _ in pts], [f for _, f in pts], label=LABEL[arm],
                ms=5, lw=1.6, **STYLE[arm])
        sd = sorted((int(x["operations"]), float(x["sd_f1@1"]))
                    for x in rows if x["arm"] == arm)
        a2.plot([o for o, _ in sd], [s for _, s in sd], ms=5, lw=1.6,
                **STYLE[arm])
    a1.set_xlabel("coverage (% of pixels actually visited)")
    a1.set_ylabel("F1@1")
    a1.set_title("equal budget is not equal coverage\n"
                 "(nearest-cell sampling makes ray points collide)",
                 fontsize=9.5)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=7, loc="lower right")
    a2.set_xlabel("operations")
    a2.set_ylabel("per-device sd of F1@1")
    a2.set_title("spread across devices\n"
                 "adaptivity should shrink this, not only raise the mean",
                 fontsize=9.5)
    a2.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT, "fig4_coverage_and_spread.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_anatomy():
    path = os.path.join(ANATOMY, "anatomy.csv")
    if not os.path.isfile(path):
        return None
    rows = read_csv(path)
    steps = sorted({int(r["step"]) for r in rows})

    def mean_by_step(col):
        return [float(np.nanmean([float(r[col]) for r in rows
                                  if int(r["step"]) == k])) for k in steps]

    theta_sd = [float(np.std([float(r["theta_deg"]) for r in rows
                              if int(r["step"]) == k])) for k in steps]

    panels = [("abs_rho", "mean |rho|\n(0 = chord through the centre)"),
              ("dist_to_triple_px",
               "median distance to the\nnearest triple point (px)"),
              ("reward", "mean reward  (delta F1@1)"),
              (None, "sd of theta across devices (deg)")]
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.5))
    for ax, (col, lab) in zip(axes, panels):
        y = theta_sd if col is None else mean_by_step(col)
        ax.plot(steps, y, "o-", color="#d62728", lw=1.4, ms=5)
        ax.set_xlabel("sweep index")
        ax.set_title(lab, fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[3].axhline(0, color="k", lw=0.8, ls=":")
    axes[3].annotate("sd near 0 at a step means that sweep is a\n"
                     "LEARNED FIXED geometry, not an adaptive one",
                     xy=(0.03, 0.85), xycoords="axes fraction", fontsize=6.5,
                     va="top")
    fig.suptitle("Policy anatomy on held-out devices. The physics prediction "
                 "is: long oblique chords early, close to the triple points "
                 "late.", fontsize=9.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig5_anatomy.png")
    fig.savefig(p, dpi=180)
    plt.close(fig)
    return p


def fig_overlays(n_devices=3, cell=(8, 60)):
    """Where the sweeps actually went, rays vs the learned policy."""
    split = dv.load_split(N_TRAIN, N_TEST, MAX_LINES, MAX_POINTS)
    grids, truths = dv.load_arrays(split.test_dirs)
    net, meta = grid_train.load(DECODER)
    env = SweepEnv(grids, truths, agnostic.as_callable(net, DEVICE),
                   threshold=meta["threshold"], n_lines=cell[0],
                   n_points=cell[1])
    agent = PPOAgent(PPOConfig(), device=DEVICE).load(AGENT)
    arms = {"rays": policies.rays(),
            "rl_ppo": policies.from_agent(agent, deterministic=True)}

    fig, axes = plt.subplots(n_devices, 4, figsize=(11.6, 3.0 * n_devices),
                             squeeze=False)
    for row in range(n_devices):
        i = row
        axes[row][0].imshow(env.Z[i], origin="lower", cmap="viridis")
        axes[row][0].set_ylabel(f"test device {i}", fontsize=8)
        axes[row][1].imshow(env.Y[i], origin="lower", cmap="gray_r")
        if row == 0:
            axes[row][0].set_title("sensor signal", fontsize=9)
            axes[row][1].set_title("ground-truth transition lines", fontsize=9)
        for col, (name, pol) in enumerate(arms.items(), start=2):
            rec = env.rollout(pol, i)
            axes[row][col].imshow(env.obs[2], origin="lower", cmap="magma",
                                  vmin=0, vmax=1)
            for rho, theta in rec.actions:
                rc = lines.sweep(rho, theta, env.n_points, env.shape)
                if len(rc):
                    axes[row][col].plot(rc[:, 1], rc[:, 0], ".", ms=1.2,
                                        color="#00e5ff")
            axes[row][col].set_xlabel(f"F1@1 = {rec.f1[-1]:.3f}", fontsize=8)
            if row == 0:
                axes[row][col].set_title(
                    f"{name}: decoder probability\nand where the sweeps went",
                    fontsize=9)
    for ax in np.ravel(axes):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Measurement geometry at {cell[0]} x {cell[1]} = "
                 f"{cell[0] * cell[1]} operations", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "fig6_overlays.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


if __name__ == "__main__":
    banner("figures")
    settings(sweep=SWEEP, anatomy=ANATOMY, out=OUT, device=DEVICE)
    os.makedirs(OUT, exist_ok=True)
    made = []
    for fn in (fig_budget_curve, fig_training, fig_headroom_and_significance,
               fig_coverage_and_spread, fig_anatomy, fig_overlays):
        try:
            p = fn()
        except FileNotFoundError as e:
            print(f"  SKIPPED {fn.__name__}: {e}")
            continue
        if p:
            made.append(p)
            print(f"  wrote {p}")
        else:
            print(f"  SKIPPED {fn.__name__}: its inputs are not on disk")
    print(f"\n  {len(made)} figures in {OUT}")
