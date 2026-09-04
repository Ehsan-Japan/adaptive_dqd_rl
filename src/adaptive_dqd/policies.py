"""
policies.py — every arm of the comparison, behind one signature.

    policy(obs, env) -> (u_rho, u_theta) in [0, 1]^2

THE COMPARISON IS TWO ARMS.  The other three are controls, and controls are
not optional.

  rays               the ray-based method — the corner fan of the companion
                     paper, at the same budget.  This is what deep RL is
                     being measured against, at every cell of the 4-8 x 40-60
                     sweep.
  rl_ppo             the learned policy.

  random_lines       random (rho, theta).  The "any change of geometry would
                     have helped" control, and the score an UNTRAINED agent
                     gets, since the Beta head is initialised near-uniform.
                     The training curve starts here by construction.
  uncertainty_greedy NON-LEARNED adaptive: go where the decoder is least
                     sure.  A one-line heuristic that matches the agent makes
                     "deep RL" an unnecessary hypothesis; a referee will
                     raise this whether or not the arm is in the table, and
                     the only good answer is having run it.
  oracle_greedy      picks the sweep that maximises the ACTUAL gain in F1@1,
                     using the ground truth.  Not a method — an upper bound.
                     It is what turns "RL beat rays" into a statement with a
                     denominator.

THE TWO NUMBERS THE PAPER QUOTES

  budget saving    the operations the learned policy needs to match the ray
                   method's 8 x 60 score.  "Deep RL reaches the fan's best
                   reconstruction from NN% of the measurements" is the
                   sentence an experimentalist cares about, because
                   acquisition time is the thing being spent.

  headroom         (rl - rays) / (oracle - rays), per budget cell.  The
                   fraction of the achievable adaptive gain that was actually
                   captured.  A claim about a method; "0.849 -> 0.88" is a
                   claim about a seed.

The geometry ablation arms (parallel_diag, hcuts, vcuts) belong to the
companion paper and are NOT arms here.  Their geometry survives in
geometry/lines.py for one purpose only: tests/test_geometry.py uses them to
show the action space contains every geometry the baseline measured, which is
what rules out "the agent won because it was handed a better primitive".
"""
from typing import Callable, Optional, Tuple

import numpy as np

from .geometry import lines
from .envs.sweep_env import CH_ENTROPY, CH_VISITED, DECODER_CHANNELS


# ── the method under comparison ───────────────────────────────────────────

def fixed(family: str = "rays", n_lines: Optional[int] = None) -> Callable:
    """
    A fixed geometry as a policy: ignore the observation, emit the k-th line.

    `n_lines=None` reads the budget off the environment at reset time, which
    is what makes ONE `rays` policy valid at all fifteen cells of the sweep:
    the fan's angles are linspace(0, 90, n+2)[1:-1], so the geometry itself
    depends on the budget and must be rebuilt per cell, not fixed at import.

    Executed by the same env, the same rasteriser and the same decoder as the
    learned arm, so the two are directly comparable — and, by
    test_geometry.py, comparable with the number the baseline repo reports
    for the same family at the same budget.
    """
    cache = {}

    def policy(obs, env):
        n = n_lines if n_lines is not None else env.n_lines
        if n not in cache:
            cache[n] = [lines.line_to_unit(r, t)
                        for r, t in lines.FAMILIES[family](n)]
        units = cache[n]
        return units[min(env.step_k, len(units) - 1)]

    policy.name = family
    return policy


def rays(n_lines: Optional[int] = None) -> Callable:
    """The ray-based method.  `fixed("rays")`, named for what it is."""
    return fixed("rays", n_lines)


# ── 4: the control ────────────────────────────────────────────────────────

def random_lines(seed: int = 0) -> Callable:
    """
    Uniform on [0,1]^2, i.e. uniform in the policy's own coordinates.

    Note this is NOT uniform over chords in the kinematic-measure sense
    (the integral-geometry invariant measure is dp dtheta, and rho is scaled
    by R(theta) here).  It is the right control anyway, because it is the
    distribution an untrained policy head emits — so arm 4 minus arm 6 is
    exactly what training bought.
    """
    rng = np.random.default_rng(seed)

    def policy(obs, env):
        return float(rng.random()), float(rng.random())

    policy.name = "random_lines"
    return policy


# ── candidate set shared by the two search policies ───────────────────────

def candidate_grid(n_theta: int = 16, n_rho: int = 16) -> np.ndarray:
    """
    (n_theta * n_rho, 2) candidate actions in unit coordinates.

    Both search arms are only as good as this grid, so it is a declared
    parameter and not a hidden constant: a greedy baseline made artificially
    weak by a coarse search is a rigged comparison in the RL's favour, which
    is the single easiest way to lose a referee.  16 x 16 = 256 candidates
    resolves ~11 degrees in angle, comfortably finer than the ~30-45 degree
    scale on which the honeycomb families differ.
    """
    u = (np.arange(n_theta) + 0.5) / n_theta
    v = (np.arange(n_rho) + 0.5) / n_rho
    uu, vv = np.meshgrid(v, u, indexing="ij")
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


# ── 5: non-learned adaptive ───────────────────────────────────────────────

def uncertainty_greedy(n_theta: int = 16, n_rho: int = 16,
                       novelty_weight: float = 1.0) -> Callable:
    """
    Sweep the chord carrying the most decoder entropy per measurement.

    score(line) = mean over its pixels of  H(p) * (novelty on unvisited)

    The novelty factor down-weights pixels already measured, so the policy
    does not re-sweep the same high-entropy chord forever — the failure mode
    of naive uncertainty sampling, and one worth stating in the paper because
    it is the reason a *learned* policy can do better: entropy is a
    one-step-greedy signal, and the value function is not.

    Cheap: no decoder calls, one entropy lookup per candidate pixel.
    """
    cand = candidate_grid(n_theta, n_rho)

    def policy(obs, env):
        H = obs[CH_ENTROPY]
        seen = obs[CH_VISITED]
        best, best_s = cand[0], -np.inf
        for u in cand:
            rho, theta = lines.unit_to_line(u[0], u[1])
            rc = lines.sweep(rho, theta, env.n_points, env.shape)
            if not len(rc):
                continue
            r, c = rc[:, 0], rc[:, 1]
            w = 1.0 - novelty_weight * seen[r, c]
            s = float(np.mean(H[r, c] * w))
            if s > best_s:
                best_s, best = s, u
        return float(best[0]), float(best[1])

    policy.name = "uncertainty_greedy"
    return policy


# ── 6: the upper bound ────────────────────────────────────────────────────

def oracle_greedy(n_theta: int = 12, n_rho: int = 12) -> Callable:
    """
    One-step-optimal with the ground truth in hand.  An upper bound, not a
    method — it is not deployable and the paper must label it so in the
    figure itself, not only in the caption.

    For every candidate: apply the sweep to a copy of the state, decode,
    score F1@1, keep the best.  That is n_theta * n_rho decoder passes and
    the same number of distance transforms PER STEP, so the grid is coarser
    here than for the entropy arm; 12 x 12 with 8 steps on 50 test devices is
    ~58k evaluations, minutes on a GPU.

    It is greedy, so it is a LOWER bound on the true optimum of an adaptive
    policy: a learned agent that beats it has learned to sacrifice an early
    step for a later one, which is worth a sentence in the paper if it
    happens.  Do not describe it as "optimal" in the text.
    """
    cand = candidate_grid(n_theta, n_rho)

    def policy(obs, env):
        from .eval.metrics import tolerant_f1
        truth = env.Y[env.i]
        Z = env.Z[env.i]
        base = obs[list(DECODER_CHANNELS)]

        batch, keep = [], []
        for u in cand:
            rho, theta = lines.unit_to_line(u[0], u[1])
            rc = lines.sweep(rho, theta, env.n_points, env.shape)
            if not len(rc):
                continue
            x = base.copy()
            r, c = rc[:, 0], rc[:, 1]
            x[0, r, c] = Z[r, c]
            x[1, r, c] = 1.0
            batch.append(x)
            keep.append(u)
        if not batch:
            return 0.5, 0.5

        probs = np.asarray(env.decoder(np.stack(batch)))
        scores = [tolerant_f1(p > env.threshold, truth, env.tau)["f1"]
                  for p in probs]
        u = keep[int(np.argmax(scores))]
        return float(u[0]), float(u[1])

    policy.name = "oracle_greedy"
    return policy


# ── 7: the learned agent, wrapped to the same signature ───────────────────

def from_agent(agent, deterministic: bool = True,
               seed: Optional[int] = None) -> Callable:
    """
    Wrap a trained PPO actor as a policy.

    deterministic=True uses the mean of the Beta head.  Report the
    deterministic number as the headline and the stochastic one as a spread:
    a policy whose sampled performance is far below its mean performance has
    a wide posterior over where to look and has not really converged.
    """
    def policy(obs, env):
        return agent.act(obs, deterministic=deterministic)

    policy.name = "rl_ppo" + ("" if deterministic else "_sampled")
    return policy


# The two arms the paper compares, then the three controls, in table order.
ARMS = ("rays", "rl_ppo", "random_lines", "uncertainty_greedy",
        "oracle_greedy")
