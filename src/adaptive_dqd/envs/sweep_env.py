"""
sweep_env.py — the MDP.  One episode is one device, measured n_lines times.

    state       what has been measured so far, plus what the decoder currently
                believes from it
    action      the next chord (rho, theta), continuous, always valid
    reward      the gain in F1@1 that the sweep produced
    terminal    after n_lines sweeps — the budget is spent

THE REWARD IS A DIFFERENCE, AND THAT IS THE WHOLE DESIGN

    r_t = F1@1( decode(s_{t+1}) )  -  F1@1( decode(s_t) )

The return of an episode telescopes:

    sum_t r_t = F1@1(final) - F1@1(nothing measured)

and F1@1(nothing measured) does not depend on the policy.  So maximising
return is *exactly* maximising the final F1@1 that the paper reports — the
agent is optimising the headline metric itself, not a proxy that correlates
with it.  At the same time the signal is dense: every sweep is scored the
moment it is taken, instead of one number arriving 8 steps later.  This is
potential-based shaping in its cleanest form (Ng, Harada & Russell 1999):
dense credit, zero bias in the optimal policy.

The alternative people reach for first — reward = information gain, or
entropy reduction, or number of new peaks found — is a proxy.  It is easier
to compute and it optimises the wrong thing.  Do not substitute it without
saying so in the paper.

WHAT THE AGENT SEES

    ch0  signal at visited pixels        the raw measurement (baseline ch0)
    ch1  visited mask                    (baseline ch1)
    ch2  decoder probability p           what the U-Net currently believes
    ch3  decoder entropy H(p)            where it is unsure
    ch4  sweeps remaining / MAX_LINES    a constant plane
    ch5  n_points / MAX_POINTS           a constant plane

ONE AGENT, EVERY BUDGET

ch4 is the number of sweeps LEFT divided by the largest budget in the study,
not the fraction of this episode that is left, and ch5 says how many points
each sweep buys.  Together they tell the policy the absolute budget it is
spending, so a SINGLE agent is trained across all fifteen cells of the sweep
with (n_lines, n_points) drawn per episode.

That is a deliberate mirror of the baseline's own guarantee — one U-Net
architecture, 1,949,409 parameters, every cell of the sweep — for the same
reason: if each budget got its own agent, a difference between budgets could
be a difference in how well fifteen separate training runs happened to go.
One agent, one parameter count, so a difference across the curve is a
difference in the measurement budget and nothing else.

The policy needs the absolute budget, not just the fraction spent, because
the right first sweep genuinely differs: with eight sweeps in hand you can
afford a wide reconnaissance chord that only pays off later, and with four
you cannot.

ch0 and ch1 are *exactly* the baseline network input, unchanged.  ch2-ch4 are
things the agent needs to decide where to look next and that a fixed geometry
has no use for.  The decoder never sees ch2-ch4; the policy never changes
what the decoder is shown.  So the reconstruction being scored is produced by
the same network from the same two channels as in the baseline paper, and the
only difference between arms is where the rays went.

COST

Each step is one decoder forward pass plus two Euclidean distance transforms
on a 100x100 map.  On GPU with the devices batched this is a few hundred
microseconds per step; the EDTs, on CPU, dominate.  `reward_every_step=False`
falls back to terminal-only reward if you need the speed, at the price of a
much harder credit-assignment problem — measure the cost before paying it.
"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from ..geometry import lines

OBS_CHANNELS = 6
(CH_SIGNAL, CH_VISITED, CH_PROB, CH_ENTROPY,
 CH_BUDGET, CH_RESOLUTION) = range(OBS_CHANNELS)

# The budget grid of the baseline paper: 4-8 sweeps of 40-60 points, fifteen
# cells, 160 to 480 measurement operations.  The comparison is run at every
# cell, so the headline is a CURVE against budget rather than one number at
# one budget — and the quantity the abstract quotes is the budget-saving
# factor: how few operations the learned policy needs to match the ray
# method's best cell.
MAX_LINES, MAX_POINTS = 8, 60

# The two channels handed to the decoder — the baseline's input, verbatim.
DECODER_CHANNELS = (CH_SIGNAL, CH_VISITED)


@dataclass
class EpisodeRecord:
    """Everything needed to redraw an episode in a figure or a table."""
    actions: list = field(default_factory=list)      # (rho, theta) per step
    rewards: list = field(default_factory=list)      # delta F1@1 per step
    f1: list = field(default_factory=list)           # F1@1 after each step
    coverage: list = field(default_factory=list)     # unique-pixel fraction
    spent: int = 0                                   # measurement operations


class SweepEnv:
    """
    Adaptive measurement of one device at a time.

    decoder  : callable (B, 2, H, W) float32 -> (B, H, W) probabilities.
               Frozen during RL.  See decoder/agnostic.py for why it must be
               geometry-agnostic and how it is trained.
    threshold: the binarisation threshold that goes with that decoder, chosen
               out-of-sample on training devices exactly as the baseline does.
               Never re-chosen here, and never chosen on test devices.
    """

    def __init__(self,
                 sensor_grids: Sequence[np.ndarray],
                 truth_maps: Sequence[np.ndarray],
                 decoder,
                 threshold: float,
                 n_lines: int = MAX_LINES,
                 n_points: int = MAX_POINTS,
                 tau: float = 1.0,
                 reward_every_step: bool = True,
                 seed: int = 0):
        assert len(sensor_grids) == len(truth_maps)
        self.Z = sensor_grids            # already min-max normalised to [0,1]
        self.Y = truth_maps              # binary ground-truth line maps
        self.decoder = decoder
        self.threshold = float(threshold)
        self.n_lines = int(n_lines)
        self.n_points = int(n_points)
        self.tau = float(tau)
        self.reward_every_step = reward_every_step
        self.rng = np.random.default_rng(seed)
        self.shape = self.Y[0].shape

    # ── episode lifecycle ────────────────────────────────────────────────

    def reset(self, device_index: Optional[int] = None,
              n_lines: Optional[int] = None,
              n_points: Optional[int] = None) -> np.ndarray:
        """
        Start an episode.  n_lines / n_points override the budget for THIS
        episode only, which is how one agent is trained across the whole
        sweep and how it is later evaluated cell by cell against `rays`.
        """
        self.i = (int(device_index) if device_index is not None
                  else int(self.rng.integers(len(self.Z))))
        if n_lines is not None:
            self.n_lines = int(n_lines)
        if n_points is not None:
            self.n_points = int(n_points)
        self.step_k = 0
        self.obs = np.zeros((OBS_CHANNELS, *self.shape), dtype=np.float32)
        self.obs[CH_BUDGET] = self.n_lines / MAX_LINES
        self.obs[CH_RESOLUTION] = self.n_points / MAX_POINTS
        self.record = EpisodeRecord()
        self._refresh_belief()
        self._f1 = self._score()
        self.record.f1.append(self._f1)
        return self.obs.copy()

    def step(self, action: Tuple[float, float]):
        """
        action : (u_rho, u_theta) in [0, 1]^2 — the policy's raw output.

        Returned info carries the geometric action actually taken, so a
        figure can be drawn from a rollout without re-deriving it.
        """
        u_rho, u_theta = action
        rho, theta = lines.unit_to_line(float(u_rho), float(u_theta))
        rc = lines.sweep(rho, theta, self.n_points, self.shape)

        if len(rc):
            r, c = rc[:, 0], rc[:, 1]
            self.obs[CH_SIGNAL, r, c] = self.Z[self.i][r, c]
            self.obs[CH_VISITED, r, c] = 1.0

        self.step_k += 1
        self.record.spent += self.n_points          # budget is operations
        self.obs[CH_BUDGET] = (self.n_lines - self.step_k) / MAX_LINES
        done = self.step_k >= self.n_lines

        self._refresh_belief()
        if self.reward_every_step or done:
            f1 = self._score()
            reward = f1 - self._f1
            self._f1 = f1
        else:
            reward = 0.0

        self.record.actions.append((rho, theta))
        self.record.rewards.append(float(reward))
        self.record.f1.append(self._f1)
        self.record.coverage.append(float(self.obs[CH_VISITED].mean()))

        info = {"rho": rho, "theta": theta, "f1": self._f1,
                "coverage": self.record.coverage[-1],
                "record": self.record if done else None}
        return self.obs.copy(), float(reward), done, info

    # ── the decoder in the loop ──────────────────────────────────────────

    def _refresh_belief(self) -> None:
        """Run the frozen decoder on the two measurement channels only."""
        x = self.obs[list(DECODER_CHANNELS)][None]           # (1, 2, H, W)
        p = np.asarray(self.decoder(x))[0].astype(np.float32)
        self.obs[CH_PROB] = p
        # Binary entropy, in nats, clipped away from the log singularity.
        q = np.clip(p, 1e-6, 1 - 1e-6)
        self.obs[CH_ENTROPY] = -(q * np.log(q) + (1 - q) * np.log(1 - q))

    def _score(self) -> float:
        from ..eval.metrics import tolerant_f1
        pred = self.obs[CH_PROB] > self.threshold
        return float(tolerant_f1(pred, self.Y[self.i], self.tau)["f1"])

    # ── running a whole episode under any policy ─────────────────────────

    def rollout(self, policy, device_index: int,
                n_lines: Optional[int] = None,
                n_points: Optional[int] = None) -> EpisodeRecord:
        """
        One episode under `policy`, which is any callable

            policy(obs, env) -> (u_rho, u_theta) in [0, 1]^2

        Fixed geometries, the greedy heuristics, the oracle and the learned
        agent all satisfy that signature, so every arm in the comparison is
        executed by these six lines and cannot diverge in the plumbing.
        """
        obs = self.reset(device_index, n_lines, n_points)
        done = False
        while not done:
            obs, _, done, info = self.step(policy(obs, self))
        return info["record"]
