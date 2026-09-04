# Learned adaptive measurement geometry for DQD stability diagrams

E. Alizadeh Kashtiban, T. Fujita, A. Oiwa — Osaka University (SANKEN).

Companion to `static_RBC_test_noise_8_15_ML_edited` (the *baseline*), which
reconstructs DQD charge-transition lines from a fixed corner fan of rays plus
a U-Net. This repository replaces the fixed geometry with a learned policy
and nothing else.

**The comparison is `rays` vs `rl_ppo` at matched budget, across the
companion paper's own 4-8 x 40-60 grid (15 cells).** `parallel_diag`, `hcuts`
and `vcuts` are NOT arms here — they belong to the companion paper's geometry
ablation. Their geometry stays in `geometry/lines.py` only so
`tests/test_geometry.py` can show the action space contains them, which is
what rules out "the agent won because it got a better primitive". Do not add
them back to the arm list. `README.md` is the human overview; this file is the
orientation for an agent.

## The one rule

**Import from the baseline; never reimplement.** The devices, the split, the
metric, the network and the training routine all come from `dqd.*` at run
time. If a number here disagrees with a number there, that must be a bug with
one cause, not two implementations to reconcile. `eval/metrics.py` raises
rather than falling back.

If the architecture, the loss, the optimiser or the threshold rule needs to
change, change it **in the baseline**, so both papers move together.

## Layout

```
scripts/            run_0 .. run_5, a settings block and a few lines each
src/adaptive_dqd/
    geometry/lines.py    THE ACTION SPACE — chords in normal form (rho,theta)
    envs/sweep_env.py    THE MDP — state, action, the telescoping reward
    policies.py          all seven arms behind one signature
    decoder/agnostic.py  the shared frozen decoder, and the confound it kills
    agents/ppo.py        PPO with a Beta head
    eval/metrics.py      the baseline's metric, imported
    eval/compare.py      the paper table, with paired tests and headroom
    config/devices.py    the baseline's device pool and split, read off disk
tests/test_geometry.py   asserts the action space contains the fan
```

## Claims this repo makes, and the file that makes each one true

- **The agent is not given a richer measurement primitive than the fan.**
  Every baseline geometry is a point in the (rho, theta) action space, and
  `tests/test_geometry.py` rebuilds each one and checks the visited pixels
  against `dqd.ml.ray_peaks` / `dqd.study.sampling` to within a Jaccard of
  0.98. Run the tests before quoting any number. The `rays` arm routes through
  `line_to_unit` -> `unit_to_line`, so if MARGIN ever clipped a fan ray the
  arm evaluated here would silently stop being the arm evaluated there;
  that is asserted too.
- **The reward is the headline metric.** `r_t = dF1@1`, so the return
  telescopes to final F1@1 minus a policy-independent constant. Dense credit,
  no proxy, no bias (potential-based shaping). Do not swap in "information
  gain" or "peaks found" without saying so in the paper. (`envs/sweep_env.py`)
- **Every action is a valid chord.** `|rho| <= (1-MARGIN) R(theta)` with
  `R = |cos|+|sin|` the support function of the square. No masking, no
  rejection, no early training spent learning the shape of the space.
- **One decoder for every arm.** `decoder/agnostic.py` trains D_agn on random
  chords at every partial budget 1..n_lines — partial budgets are not
  optional, since the agent queries the decoder after every single sweep.
  Read the module docstring before writing the methods section; the SHARED /
  MATCHED two-column protocol is the answer to the obvious referee question
  and it needs to be in the paper, not just in the code.
- **The test devices are untouched** until `run_4`. `config/devices.py`
  returns train and test as separate fields so they are hard to confuse.
- **Coverage is reported beside budget**, because nearest-cell sampling makes
  equal budget != equal coverage, worst near the fan's origin.

## Reporting conventions

- F1@tau, tau = 0..3, tau = 1 as headline. Never quote pixel accuracy.
- Report IoU even though it is low for one-pixel lines — the baseline does.
- Report per-device sd of F1@1. Adaptivity should shrink it; a gain in the
  mean with no change in the spread is a weaker result than it looks.
- Test paired (Wilcoxon signed-rank on per-device differences), not unpaired.
- Two headline numbers: the **budget saving** (operations the policy needs to
  match `rays` at 8 x 60, as a % of 480) and the **headroom ratio**
  (rl - rays)/(oracle - rays) per cell. The first is what an experimentalist
  reads; the second is what a referee reads.
- Fifteen cells means fifteen paired tests. Read them as a family; a curve
  that is above at all fifteen is the claim, not one cell at p < 0.05.
- Never call `oracle_greedy` optimal. It is greedy, so it is a lower bound on
  the adaptive optimum, and it uses ground truth, so it is not a method. Label
  it as a bound **in the figure**, not only in the caption.

## Failure modes to check for before believing a result

- PPO entropy collapses to ~0 -> the policy found no device-dependent
  structure and has learned a fixed geometry. That is publishable, but it is
  a different sentence.
- PPO entropy never falls -> nothing was learned; the return will sit at the
  `random_lines` control, which is where it starts by design.
- `uncertainty_greedy` matches `rl_ppo` -> adaptivity works, deep RL is not
  needed for it. Say that.
- The RL curve never crosses the `rays` 8 x 60 score anywhere on the grid ->
  report that plainly; do not extrapolate past the last measured cell.
- `oracle_greedy` matches `rays` at a cell -> adaptivity buys nothing at
  that budget. Report it; it is a real statement about the honeycomb's
  regularity, and it is why the oracle arm is not optional.
- Reward per step ~0 for the last sweeps -> the budget is too large and the
  interesting experiment is a smaller one.

## Conventions

- `results/`, `checkpoints/`, `*.pptx`, `*.pdf`, `*.docx` are gitignored.
- The baseline's folder name says `noise_8_15`; **the simulation carries no
  noise**. Inherited here. Do not infer otherwise from a path.
- The budget grid is the baseline's: (4,5,6,7,8) x (40,50,60), 15 cells,
  160-480 operations. 8 x 60 is the bar to clear (F1@1 0.849, 4.66% coverage).
- ONE agent covers all 15 cells; the budget is drawn per episode and enters
  the observation as two constant planes. Do not train fifteen agents — that
  would let a difference along the curve be a difference between training
  runs.
