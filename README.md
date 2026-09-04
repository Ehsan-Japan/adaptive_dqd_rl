# Learned adaptive measurement geometry for DQD charge stability diagrams

Where should the next voltage sweep go, given everything measured so far?

The companion repository,
[`static_RBC_test_noise_8_15_ML_edited`](https://github.com/Ehsan-Japan/static_RBC_test_noise_8_15_ML_edited),
recovers the charge-transition lines of a double quantum dot from a few
percent of the (V₁, V₂) plane: a fixed fan of rays from one corner, a U-Net
that turns the sparse traces into a dense transition-line map. Its own
limitations section names what is left: *fixed ray origin; rays are
non-adaptive.*

This repository removes that. The measurement becomes a **policy** — a
function from what has been measured to where to measure next — and the
policy is learned with PPO. The U-Net does not change. The metric does not
change. The devices do not change. Only the choice of where the rays go.

E. Alizadeh Kashtiban, T. Fujita, A. Oiwa — Osaka University (SANKEN).

## The physical argument, before the machine learning

Imagine standing in the (V₁, V₂) plane with a torch that lights one straight
line at a time, and a budget of eight lines.

A stability diagram is a honeycomb. Its edges come in three families —
add an electron to dot 1, add one to dot 2, move one between them — and the
lattice is set by C₁, C₂ and the interdot C_m, which differ from device to
device. Two consequences, and they pull in opposite directions.

**Early, you know nothing about the lattice, so you want a long oblique
sweep across the middle.** An oblique line crosses both honeycomb families;
an axis-aligned one runs nearly parallel to one of them and learns half as
much. This is not a hypothesis — it is already in the baseline's geometry
ablation, where both oblique arms beat both axis-aligned ones (0.879 and
0.849 against 0.841 and 0.839). Two or three such sweeps essentially fix the
lattice vectors.

**Late, the lattice is known, so the long straight edges are nearly free.**
A decoder that has seen two parallel edges can extrapolate the rest of the
family. What it cannot extrapolate is the neighbourhood of the **triple
points**, where three charge states meet: the interdot segments there are
short, they are not on the lattice of either family, and the charge sensor's
contrast is weakest. That is where the remaining uncertainty lives, and that
is where the last sweeps should go.

A fixed geometry cannot do both, because it is chosen before the device is
seen. That is the entire case for adaptivity, and it is a claim about the
physics of the honeycomb rather than about reinforcement learning. If it is
wrong — if the honeycomb is regular enough that one good ray fan is
already near-optimal — then this repository's `oracle_greedy` arm will say
so, and that is a result worth reporting too.

`scripts/run_5_policy_anatomy.py` tests the prediction: |ρ| and distance to
the nearest triple point, per step, over the held-out devices.

## The MDP

| | |
|---|---|
| **episode** | one device, `n_lines` sweeps |
| **action** | a chord of the window, (ρ, θ), continuous, always valid |
| **state** | measured signal, visited mask, decoder probability, decoder entropy, budget remaining |
| **reward** | `F1@1(after) − F1@1(before)` |
| **terminal** | budget spent |

The reward telescopes to `F1@1(final) − F1@1(nothing measured)`, and the
second term does not depend on the policy. So the agent optimises the paper's
headline metric itself, densely, with no proxy in between. That is
potential-based shaping, and it is the one design decision here worth
defending at length — see `envs/sweep_env.py`.

The action space is written in normal form, u·n(θ) = ρ, and it **contains
every geometry the baseline tested**: `rays` is the pencil through (1,1),
`parallel_diag` is θ = 3π/4, `hcuts` is θ = π/2. `tests/test_geometry.py`
asserts this against the baseline's own rasteriser, pixel for pixel. The
agent is therefore not given a richer measurement primitive than the ray
method — it is given the same primitive and allowed to place it.

## The comparison: `rays` vs `rl_ppo`, at matched budget

Two arms, fifteen budgets. The grid is the companion paper's own — 4–8 sweeps
of 40–60 points, 160 to 480 measurement operations — run on the same held-out
devices through the same metric code.

**One number at one budget is not the result.** The result is a curve, and
what an experimentalist reads off it is cost:

> how few measurements does the learned policy need to reach the
> reconstruction the ray method gets from its full budget?

The companion paper's reference points are `4×40 → 0.672`, `8×40 → 0.810`,
`8×50 → 0.835`, `8×60 → 0.849`. If the RL curve crosses 0.849 somewhere near
6×50, the sentence is *"the same reconstruction from 62 % of the acquisition"* —
in the units the fridge is actually paid for. And a curve that sits above
another curve at fifteen cells is not a seed.

Three controls ride along, because they cost almost nothing and each one
closes a hole:

| control | the objection it answers |
|---|---|
| `random_lines` | "any change of geometry would have helped" — also the score an *untrained* agent gets, so the training curve starts here by construction |
| `uncertainty_greedy` | **"you did not need deep RL for this"** — a one-line entropy heuristic. If it matches the agent, the paper is about adaptive measurement, not about deep RL, and it should say so |
| `oracle_greedy` | "adaptivity was worth nothing here" — ground truth in hand, an upper bound and not a method. It supplies the denominator in *(RL − rays)/(oracle − rays)* |

The geometry ablation arms — `parallel_diag`, `hcuts`, `vcuts` — belong to the
companion paper and are **not** arms here. Their geometry survives in
`geometry/lines.py` for one purpose: `tests/test_geometry.py` uses them to
show that the action space contains every geometry the baseline measured,
which is what rules out *"the agent won because it was handed a better
measurement primitive."*

## One agent, every budget

The policy is trained once, with a budget cell drawn uniformly per episode,
and evaluated at all fifteen. That mirrors the baseline's own guarantee —
one architecture, 1,949,409 parameters, every cell of the sweep — for the
same reason: fifteen separate training runs would let a difference along the
curve be a difference in how fifteen optimisations happened to go. The
observation carries sweeps-remaining and points-per-sweep as constant planes,
because the right first sweep genuinely differs: with eight sweeps in hand
you can afford a wide reconnaissance chord that only pays off later, and with
four you cannot.

The shared decoder is budget-agnostic for the same reason, and additionally
sees partial budgets of 1…n sweeps — not optional, since the agent queries it
after *every* sweep and would otherwise take its first two or three actions
blind.

## Getting started

```bash
git clone --recurse-submodules <this repo>
pip install -r requirements.txt
export DQD_BASELINE=/path/to/static_RBC_test_noise_8_15_ML_edited  # or use the submodule

python scripts/run_1_import_devices.py     # check the link to the baseline
pytest tests/ -v                           # assert the action space contains `rays`
python scripts/run_0_full_study.py         # everything
```

Stage by stage:

```
run_1_import_devices.py    verify the shared devices, split, metric, network
run_2_train_decoder.py     the shared geometry-agnostic decoder D_agn
run_3_train_agent.py       PPO, on training devices only
run_4_compare_arms.py      rays vs RL at 15 budgets -> results/sweep/
run_5_policy_anatomy.py    does the policy do what the physics predicts?
```

## What keeps the comparison honest

* **The same devices, not equivalent ones.** The pool and the device-level
  split are read off disk from the baseline repository, never regenerated.
* **The same metric code.** `eval/metrics.py` imports `dqd.ml.grid_metrics`
  and refuses to run if it cannot find it, rather than falling back to a
  lookalike.
* **The same network and the same training constants**, imported from
  `dqd.ml.grid_model` and `dqd.ml.grid_train` unchanged.
* **One frozen decoder for every arm and every budget** in the headline
  column, so a difference between arms cannot be a difference in the
  reconstruction network. A second column re-trains a decoder per arm; both are reported,
  and if they disagree, that disagreement is the finding.
* **Coverage is reported next to budget.** Nearest-cell sampling makes ray
  points collide, badly near the fan's origin, so equal budget is not equal
  coverage. If the agent wins with *lower* coverage, that is the strong form
  of the result and it should be visible.
* **Paired statistics.** All arms run on the same devices in the same order,
  so per-device differences are tested with a Wilcoxon signed-rank, not an
  unpaired t-test that would miss a real effect at n = 50. Fifteen cells means
  fifteen tests, so read them as a family, not as fifteen chances at p < 0.05.
* **No test device is loaded** by the decoder script or the agent script.

## Limitations, up front

Simulation only and noise-free, inherited from the baseline — and adaptive
measurement is precisely the thing most likely to look better in simulation
than on hardware, because a real sweep has settling time, drift and a sensor
that must be re-tuned. A learned policy that reorders sweeps arbitrarily may
be more expensive in wall-clock time than a fan even at equal point count;
this repository counts measurement operations, not seconds. θ lives on a
circle and the Beta head treats it as an interval, so the policy cannot place
a single mode across θ = 0 ≡ π. The oracle arm is greedy, so it is a lower
bound on the true adaptive optimum and must not be called "optimal" in the
text. Every device is 100 × 100 on a 2 × 2 mV window; nothing here has been
tested at another resolution.
