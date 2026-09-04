"""
ppo.py — PPO over chords, with a Beta head.

WHY BETA AND NOT A SQUASHED GAUSSIAN

The action lives on [0,1]^2 by construction (geometry/lines.py), and both
edges are meaningful: u_rho = 0 and 1 are the two chords grazing opposite
corners, and the policy genuinely wants to sit near them early in an episode,
when the useful sweep is a long one across the middle... and near them again
late, when it is filling in an edge.  A tanh-squashed Gaussian has vanishing
density gradient exactly there, so it learns those actions slowly and its
log-probabilities need a change-of-variables correction that is easy to get
wrong.  A Beta distribution is supported on [0,1] natively, needs no
correction, and can be either unimodal or edge-seeking depending on whether
its parameters exceed 1.  Chou, Maturana & Scherer (2017) is the reference.

alpha, beta = 1 + softplus(.), so the density is never improper and the
initial policy is close to uniform — which is deliberate: it makes the
untrained agent equal to the `random_lines` control arm, so the training
curve starts exactly at the control's score and the plot reads as "what
learning bought", with no offset to explain.

WHAT THE TRUNK SEES

Five 100x100 planes (measurement, belief, budget).  A small strided CNN, not
a U-Net: the actor outputs four numbers, not an image, so it needs a global
summary of the diagram — which honeycomb family is visible, where the belief
is thin — and not pixel-precise localisation.  Reconstruction accuracy is the
decoder's job and stays the decoder's job.

CIRCULARITY, STATED

theta lives on a circle: theta = 0 and theta = pi are the same direction with
rho negated.  The Beta head treats [0,1] as an interval, so the policy cannot
put a single mode across that seam and must use two.  In practice the
honeycomb's own structure means the useful angles are interior and the seam
is rarely wanted, but it is a real limitation of this parameterisation and
belongs in the paper's limitations paragraph rather than nowhere.  A von
Mises head over theta with a Beta over rho would remove it.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from ..envs.sweep_env import OBS_CHANNELS


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99          # episodes are 8 steps; ~1 is right
    lam: float = 0.95            # GAE
    clip: float = 0.2
    epochs: int = 4              # optimisation passes per batch
    minibatch: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    episodes_per_batch: int = 32
    width: int = 32
    # Cells of the baseline's budget grid to draw from, one per episode.  A
    # single agent covers the whole sweep — see envs/sweep_env.py for why.
    budget_grid: tuple = tuple((r, p) for r in (4, 5, 6, 7, 8)
                               for p in (40, 50, 60))


class ActorCritic(nn.Module):
    def __init__(self, in_channels: int = OBS_CHANNELS, width: int = 32):
        super().__init__()
        w = width
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, w, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(4 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.actor = nn.Sequential(nn.Linear(4 * w, 4 * w), nn.GELU(),
                                   nn.Linear(4 * w, 4))      # a_rho,b_rho,a_th,b_th
        self.critic = nn.Sequential(nn.Linear(4 * w, 4 * w), nn.GELU(),
                                    nn.Linear(4 * w, 1))
        # Start near-uniform: zero the last actor layer so softplus(0) is the
        # same for every parameter and the initial Beta is close to Beta(1,1).
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs: torch.Tensor) -> Beta:
        h = self.trunk(obs)
        p = 1.0 + F.softplus(self.actor(h))
        return Beta(p[:, :2], p[:, 2:])

    def forward(self, obs: torch.Tensor) -> Tuple[Beta, torch.Tensor]:
        h = self.trunk(obs)
        p = 1.0 + F.softplus(self.actor(h))
        return Beta(p[:, :2], p[:, 2:]), self.critic(h).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: PPOConfig = PPOConfig(), device: str = "cuda",
                 seed: int = 0):
        torch.manual_seed(seed)
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(width=cfg.width).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    # ── acting ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        d = self.net.dist(t)
        a = d.mean if deterministic else d.sample()
        return tuple(float(v) for v in a[0].cpu().numpy())

    @torch.no_grad()
    def act_with_logp(self, obs: np.ndarray):
        t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        d, v = self.net(t)
        a = d.sample()
        return (a[0].cpu().numpy(),
                float(d.log_prob(a).sum(-1)[0]),
                float(v[0]))

    # ── collecting ───────────────────────────────────────────────────────

    def collect(self, env, device_indices: List[int], budgets=None):
        """
        One batch of complete episodes, each at its own budget.

        Episodes always run to the end of their budget, so there is no
        truncation bootstrap to get wrong — but they now have DIFFERENT
        lengths within a batch (4 to 8 steps), which is exactly why the GAE
        recursion below keys off the stored done flags rather than assuming a
        fixed stride.
        """
        O, A, LP, R, V, D = [], [], [], [], [], []
        returns_log = []
        budgets = budgets or [(env.n_lines, env.n_points)] * len(device_indices)
        for idx, (n_lines, n_points) in zip(device_indices, budgets):
            obs = env.reset(idx, n_lines, n_points)
            k0 = len(R)
            done = False
            while not done:
                a, lp, v = self.act_with_logp(obs)
                nxt, r, done, _ = env.step(a)
                O.append(obs); A.append(a); LP.append(lp)
                R.append(r); V.append(v); D.append(float(done))
                obs = nxt
            returns_log.append(sum(R[k0:]))
        return (np.stack(O), np.stack(A), np.array(LP, dtype=np.float32),
                np.array(R, dtype=np.float32), np.array(V, dtype=np.float32),
                np.array(D, dtype=np.float32), float(np.mean(returns_log)))

    def gae(self, R, V, D):
        cfg = self.cfg
        adv = np.zeros_like(R)
        last = 0.0
        for t in reversed(range(len(R))):
            nonterminal = 1.0 - D[t]
            v_next = V[t + 1] * nonterminal if t + 1 < len(R) else 0.0
            delta = R[t] + cfg.gamma * v_next - V[t]
            last = delta + cfg.gamma * cfg.lam * nonterminal * last
            adv[t] = last
        return adv, adv + V

    # ── updating ─────────────────────────────────────────────────────────

    def update(self, O, A, LP, adv, ret):
        cfg = self.cfg
        dev = self.device
        O = torch.as_tensor(O, dtype=torch.float32, device=dev)
        A = torch.as_tensor(A, dtype=torch.float32, device=dev).clamp(1e-4, 1 - 1e-4)
        LP = torch.as_tensor(LP, dtype=torch.float32, device=dev)
        adv = torch.as_tensor(adv, dtype=torch.float32, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.as_tensor(ret, dtype=torch.float32, device=dev)

        n = len(O)
        stats = {}
        for _ in range(cfg.epochs):
            for i in torch.randperm(n, device=dev).split(cfg.minibatch):
                d, v = self.net(O[i])
                lp = d.log_prob(A[i]).sum(-1)
                ratio = torch.exp(lp - LP[i])
                l1 = ratio * adv[i]
                l2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[i]
                pol = -torch.min(l1, l2).mean()
                val = F.mse_loss(v, ret[i])
                ent = d.entropy().sum(-1).mean()
                loss = pol + cfg.value_coef * val - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats = {"policy_loss": float(pol), "value_loss": float(val),
                         "entropy": float(ent)}
        return stats

    def train(self, env, train_indices: List[int], iterations: int = 200,
              log_every: int = 5, rng_seed: int = 0):
        rng = np.random.default_rng(rng_seed)
        history = []
        for it in range(1, iterations + 1):
            idx = list(rng.choice(train_indices,
                                  size=self.cfg.episodes_per_batch,
                                  replace=len(train_indices) <
                                  self.cfg.episodes_per_batch))
            # One budget cell per episode, uniformly over the grid, so the
            # agent is trained on the whole sweep at once instead of fifteen
            # times.  Uniform and not weighted: the cheap cells are the ones
            # the paper's budget-saving claim is read off, so they must not be
            # under-trained relative to 8 x 60.
            cells = self.cfg.budget_grid
            budgets = [cells[int(j)] for j in
                       rng.integers(0, len(cells), self.cfg.episodes_per_batch)]
            O, A, LP, R, V, D, mean_ret = self.collect(env, idx, budgets)
            adv, ret = self.gae(R, V, D)
            stats = self.update(O, A, LP, adv, ret)
            stats.update(iteration=it, mean_return=mean_ret)
            history.append(stats)
            if it % log_every == 0:
                print(f"  it {it:4d}  mean episode return (= final F1@1 "
                      f"minus empty-measurement F1@1) {mean_ret:.4f}  "
                      f"entropy {stats['entropy']:+.3f}")
        return history

    def save(self, path: str):
        torch.save({"state_dict": self.net.state_dict(),
                    "config": self.cfg.__dict__}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["state_dict"])
        return self
