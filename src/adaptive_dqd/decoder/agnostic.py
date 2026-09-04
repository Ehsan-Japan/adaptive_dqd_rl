"""
agnostic.py — the decoder, and the confound it exists to remove.

THE PROBLEM

The baseline trains one U-Net per geometry: the fan network only ever sees
fan measurements, so it learns the fan's blind spots as a prior and fills
them in.  An RL agent changes its geometry every step and every device, so
its decoder must handle any chord arrangement.

That asymmetry is a confound, and it cuts both ways:

  * give the agent a geometry-agnostic decoder while every fixed arm keeps
    its specialised one, and the agent is handicapped — it is being asked to
    win with a weaker reconstruction network;
  * retrain the decoder on the agent's own measurement distribution and
    leave the fixed arms with the shared one, and the agent wins on decoder
    adaptation, not on measurement.  This is the version that gets published
    by accident and retracted on inspection.

Either way the number in the abstract is not about adaptive measurement.

THE PROTOCOL

Every arm is evaluated TWICE, and both columns go in the table.

  SHARED   one decoder D_agn, trained on chords drawn at random from the
           full (rho, theta) action space, used unchanged by every arm
           including the fixed ones.  Isolates geometry: the reconstruction
           network is literally the same weights everywhere, so a difference
           between arms is a difference in where the sweeps went and in
           nothing else.  This is the scientific column.

  MATCHED  each arm gets its own decoder, retrained from scratch on that
           arm's own measurement distribution with identical training
           constants and identical epochs.  For the fan this reproduces the
           baseline paper's own setup; for the agent it is the deployment
           setting.  This is the engineering column.

Report both.  If the ordering of the arms is the same in both columns, the
result is robust and the paper is easy to defend.  If it is not, that
disagreement is the most interesting thing you will find, and it belongs in
the text rather than in a footnote.

TRAINING CONSTANTS ARE NOT TOUCHED

grid_train.train() is imported from the baseline unchanged — same Adam 1e-3,
same batch 16, same BCE-with-capped-positive-weight plus soft Dice, same
threshold selection on a validation split carved out of the TRAINING
devices.  This module only decides which measurements go in; it does not get
a vote on how the network is fitted.  Do not add a learning-rate argument
here.  If training needs to change, change it in the baseline so both papers
change together.
"""
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from .baseline_net import grid_train
from ..geometry import lines


def random_chord_input(Z: np.ndarray, n_lines: int, n_points: int,
                       rng: np.random.Generator) -> np.ndarray:
    """
    (2, H, W) input from n_lines chords drawn uniformly in unit coordinates.

    Uniform in the POLICY's coordinates, deliberately: the decoder's training
    distribution is then the distribution an untrained policy induces, which
    is the distribution the agent starts from and stays inside as it learns.
    A decoder trained only on, say, near-diagonal chords would silently
    penalise the agent for exploring away from them.
    """
    x = np.zeros((2, *Z.shape), dtype=np.float32)
    for _ in range(n_lines):
        rho, theta = lines.unit_to_line(rng.random(), rng.random())
        rc = lines.sweep(rho, theta, n_points, Z.shape)
        if len(rc):
            r, c = rc[:, 0], rc[:, 1]
            x[0, r, c] = Z[r, c]
            x[1, r, c] = 1.0
    return x


def build_agnostic_dataset(grids: Sequence[np.ndarray],
                           truths: Sequence[np.ndarray],
                           n_lines: int,
                           n_points: int,
                           repeats: int = 4,
                           vary_budget: bool = True,
                           point_grid: Sequence[int] = (40, 50, 60),
                           seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    (X, Y) for the shared decoder.

    repeats : how many random geometries per device.  The decoder has to be
              invariant to geometry, so it needs to see several per device;
              4 is enough to stop it memorising one arrangement per device
              and cheap enough to keep the dataset in memory.

    vary_budget : also draw geometries with 1..n_lines chords, and n_points
              from `point_grid`, rather than only the full budget.  This is
              not optional in practice, for two separate reasons.

              Within an episode: the agent queries the decoder after every
              single sweep, so at step 1 it hands the network a ONE-chord
              input.  A decoder that has only ever seen eight-chord inputs
              produces nonsense there, the entropy channel is nonsense with
              it, and the first two or three actions of every episode are
              taken blind.

              Across the sweep: the comparison runs at all fifteen cells of
              the 4-8 x 40-60 grid, so the decoder has to be as
              budget-agnostic as it is geometry-agnostic.  One decoder for
              every cell is the same guarantee the baseline makes with a
              fixed architecture — a difference along the curve is then a
              difference in measurement, not fifteen training runs that went
              differently.
    """
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for Z, Y in zip(grids, truths):
        for _ in range(repeats):
            if vary_budget:
                k = int(rng.integers(1, n_lines + 1))
                pts = int(rng.choice(point_grid))
            else:
                k, pts = n_lines, n_points
            Xs.append(random_chord_input(Z, k, pts, rng))
            Ys.append(Y.astype(np.float32))
    return np.stack(Xs), np.stack(Ys)


def train_agnostic(grids, truths, n_lines: int, n_points: int,
                   epochs: int = 50, repeats: int = 4, seed: int = 0):
    """Fit D_agn with the baseline's own training routine.  Returns (net, tau)."""
    X, Y = build_agnostic_dataset(grids, truths, n_lines, n_points,
                                  repeats=repeats, seed=seed)
    net, threshold, history = grid_train.train(X, Y, epochs=epochs)
    return net, threshold, history


def as_callable(net, device: str = "cuda") -> Callable[[np.ndarray], np.ndarray]:
    """
    Freeze a trained net into the plain (B,2,H,W) -> (B,H,W) probability
    function the environment expects.

    eval() and no_grad() are not a detail: BatchNorm in train mode would let
    the composition of the batch — which, for the oracle arm, is a batch of
    256 hypothetical futures — change the prediction for each of them.  That
    would make the oracle's own search invalid.
    """
    import torch
    net = net.to(device).eval()

    @torch.no_grad()
    def decode(x: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
        return torch.sigmoid(net(t)).cpu().numpy()

    return decode
