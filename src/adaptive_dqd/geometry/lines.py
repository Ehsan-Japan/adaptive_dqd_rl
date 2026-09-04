"""
lines.py — one line in the gate-voltage window, as (rho, theta).

THE ACTION SPACE.  Everything the agent can do is pick a chord of the
(V1, V2) window and sweep it.  A chord is written in normal form

    { u : u . n(theta) = rho },      n(theta) = (cos theta, sin theta)

in window-normalised coordinates u in [-1, 1]^2, with

    theta in [0, pi)                 the direction of the line's normal
    rho   in [-R(theta), +R(theta)]  signed distance from the window centre
    R(theta) = |cos theta| + |sin theta|

R(theta) is the support function of the square, so |rho| <= R(theta) is
exactly the condition that the line meets the window.  Every action is
therefore a real chord: there is no wasted region of the action space, and
no action needs to be masked or rejected.  That matters for PPO — a policy
that can emit invalid actions spends its early training learning not to.

WHY THIS PARAMETERISATION AND NOT "origin + angle"

Because it is complete and it is flat.  Every straight measurement sweep any
of the baseline arms performs is a point in this two-dimensional space:

    vcuts          theta = 0,      rho = x of the cut
    hcuts          theta = pi/2,   rho = y of the cut        (Hernandes-style)
    parallel_diag  theta = 3pi/4,  rho = (x - y)/sqrt(2)
    corner fan     the pencil through (1, 1):
                   theta = t - pi/2,  rho = sin t - cos t,   t in (0, pi/2)

`test_geometry.py` asserts this: it rebuilds each baseline family from
(rho, theta) and checks the visited-pixel sets match the baseline repo's own
`sampling.py` to the pixel.  This is the claim the paper needs — the learned
policy is NOT given a richer measurement primitive than the fan.  It is given
the same primitive and allowed to place it.  Any improvement is adaptivity,
not a bigger toolbox.

BUDGET

n_points samples, evenly spaced along the chord, exactly as ray_polyline does
along a fan ray.  Points per line is fixed, not spacing, so the budget is the
baseline's budget: n_lines x n_points measurement operations.  A short chord
therefore spends a full line's budget on a small piece of the window and
returns few unique pixels — the agent is not rewarded for that and learns to
avoid it, so no exploit exists here.  Unique-pixel coverage is reported
alongside the budget for every arm regardless; see eval/compare.py.
"""
from typing import Tuple

import numpy as np

# The window in normalised coordinates.  The real device window is 2 x 2 mV
# with a per-device origin (baseline CLAUDE.md: offset_scale = 0.35); mapping
# to [-1, 1]^2 makes a policy transferable across devices, which is the whole
# point of learning one.
LO, HI = -1.0, 1.0


def support(theta: np.ndarray) -> np.ndarray:
    """R(theta) = |cos| + |sin| — the largest |rho| that still meets the square."""
    return np.abs(np.cos(theta)) + np.abs(np.sin(theta))


# |rho| is capped at (1 - MARGIN) R(theta) rather than at R(theta) itself.
# At exactly |rho| = R the line is TANGENT to the square: it touches one
# corner and the chord has zero length, so the action is valid in the
# geometric sense and useless in the physical one.  Two consequences if the
# margin is dropped: the extreme actions measure nothing at all, and the
# clipping arithmetic in chord() has s_hi == s_lo and returns an empty array
# that every caller then has to guard against.  5% keeps the action space a
# clean rectangle — which is what the Beta head needs — while guaranteeing
# every action is a chord of finite length.  Near-corner chords are still
# short and still nearly worthless; the agent is left to discover that, since
# discovering it is evidence the reward is doing its job.
MARGIN = 0.05


def unit_to_line(u_rho: float, u_theta: float) -> Tuple[float, float]:
    """
    Map a policy output in [0, 1]^2 to (rho, theta).

    u_theta -> theta = pi * u_theta                the pencil of directions
    u_rho   -> rho   = (2 u_rho - 1)(1 - MARGIN) R(theta)

    rho is scaled by R(theta), not by a constant, so u_rho = 0 is always the
    chord near one corner and u_rho = 1 the one near the opposite corner, at
    every theta.  The policy's output therefore means the same thing at every
    angle, which is what lets a single Beta head cover the space evenly
    instead of wasting most of its mass on lines that miss the window.
    """
    theta = float(np.pi * np.clip(u_theta, 0.0, 1.0))
    rho = float((2.0 * np.clip(u_rho, 0.0, 1.0) - 1.0)
                * (1.0 - MARGIN) * support(theta))
    return rho, theta


def line_to_unit(rho: float, theta: float) -> Tuple[float, float]:
    """
    Inverse of unit_to_line — how the fixed arms are expressed as policies.

    A line closer to a corner than the margin allows is not representable and
    comes back clipped to the edge of the unit square.  For the fan this
    binds only past ~60 rays (the outermost ray of an n-ray fan sits at
    |rho|/R = tan(pi/4 - 90deg/(n+1))), so it never binds at the budgets in
    the study; the clip is asserted against in tests/test_geometry.py so it
    cannot start binding silently if the budget grows.
    """
    theta = float(theta % np.pi)
    r = (1.0 - MARGIN) * support(theta)
    return (float(np.clip((rho / r + 1.0) / 2.0, 0.0, 1.0)),
            float(theta / np.pi))


def chord(rho: float, theta: float, n_points: int) -> np.ndarray:
    """
    (n_points, 2) points evenly spaced along the chord, in [-1, 1]^2.

    The line is parameterised as p(s) = rho * n + s * d with d the unit
    tangent (-sin, cos).  Clipping to the square is a 1-D interval
    intersection in s (Liang-Barsky), which is exact and branch-free enough
    to run inside the RL loop millions of times.

    Returns an empty (0, 2) array if the line misses the window, which
    |rho| <= R(theta) forbids — kept as a guard for hand-written chords.
    """
    n = np.array([np.cos(theta), np.sin(theta)])
    d = np.array([-np.sin(theta), np.cos(theta)])
    p0 = rho * n

    s_lo, s_hi = -np.inf, np.inf
    for k in (0, 1):
        if abs(d[k]) < 1e-12:
            if not (LO - 1e-9 <= p0[k] <= HI + 1e-9):
                return np.empty((0, 2))
            continue
        t1 = (LO - p0[k]) / d[k]
        t2 = (HI - p0[k]) / d[k]
        s_lo = max(s_lo, min(t1, t2))
        s_hi = min(s_hi, max(t1, t2))
    if s_hi <= s_lo:
        return np.empty((0, 2))

    s = np.linspace(s_lo, s_hi, int(n_points))
    return np.clip(p0[None, :] + s[:, None] * d[None, :], LO, HI)


def to_pixels(pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    (N, 2) normalised points -> (N, 2) integer (row, col).

    Same convention as the baseline's voltage_to_pixel: x drives the column,
    y the row, nearest cell by rounding, clipped into the grid.  Identical
    rounding is what makes a coverage number here comparable with a coverage
    number there.
    """
    height, width = shape
    if not len(pts):
        return np.empty((0, 2), dtype=int)
    col = np.rint((pts[:, 0] - LO) / (HI - LO) * (width - 1)).astype(int)
    row = np.rint((pts[:, 1] - LO) / (HI - LO) * (height - 1)).astype(int)
    return np.stack([np.clip(row, 0, height - 1),
                     np.clip(col, 0, width - 1)], axis=1)


def sweep(rho: float, theta: float, n_points: int,
          shape: Tuple[int, int]) -> np.ndarray:
    """One measurement sweep -> the (row, col) pixels it lands on, in order.

    Duplicates are NOT removed: the returned length is the budget actually
    spent (n_points measurement operations), while np.unique of it is the
    information actually gained.  Both numbers are needed and they differ,
    which is precisely the honesty note in the baseline's sampling.py.
    """
    return to_pixels(chord(rho, theta, n_points), shape)


# ──────────────────────────────────────────────────────────────────────────
#  The fixed families, written in the action space
#
#  These exist so the baselines and the agent are executed by the SAME code
#  path.  A fixed arm is a policy that ignores its observation and returns
#  the k-th element of one of these lists.  Nothing downstream can tell the
#  difference, so no arm gets an accidental advantage from a different
#  rasteriser, a different clip rule or a different point count.
# ──────────────────────────────────────────────────────────────────────────

def family_vcuts(n_lines: int) -> np.ndarray:
    """Vertical cuts, evenly spaced in x — cell centres, as the baseline does."""
    xs = LO + (np.arange(n_lines) + 0.5) / n_lines * (HI - LO)
    return np.stack([xs, np.zeros(n_lines)], axis=1)


def family_hcuts(n_lines: int) -> np.ndarray:
    """Horizontal cuts — the Hernandes et al. line-cut mask."""
    ys = LO + (np.arange(n_lines) + 0.5) / n_lines * (HI - LO)
    return np.stack([ys, np.full(n_lines, np.pi / 2)], axis=1)


def family_parallel_diag(n_lines: int) -> np.ndarray:
    """
    Parallel oblique lines along (-1, -1) — the arm that beat the fan.

    The line through (x, y) with tangent (-1,-1)/sqrt(2) has normal
    (1, -1)/sqrt(2), i.e. theta = 3pi/4 (mod pi), and rho = (y - x)/sqrt(2)
    for that normal's sign convention.  rho is swept over its full range
    [-R, R] at cell centres, so the family spans corner to corner.
    """
    theta = 3.0 * np.pi / 4.0
    r = support(theta)                                    # = sqrt(2)
    rhos = -r + (np.arange(n_lines) + 0.5) / n_lines * 2 * r
    return np.stack([rhos, np.full(n_lines, theta)], axis=1)


def family_rays(n_lines: int) -> np.ndarray:
    """
    The ray-based method — the corner fan, as a pencil of chords through (1,1).

    This is `rays` in dqd.study.sampling: THE method the paper compares
    against, at every budget.

    Baseline angles are linspace(0, 90, n+2)[1:-1] degrees, the ray heading
    in (-cos t, -sin t) from the (max Vx, max Vy) corner.  Raw normal form
    would be theta_raw = t - pi/2, which is negative for every t in (0, 90);
    reducing it into [0, pi) reverses the normal, so rho reverses with it:

        theta = t + pi/2,   rho = (1,1) . n(theta) = cos t - sin t

    Sanity: t = 45 deg gives theta = 135 deg, rho = 0 — the window's main
    diagonal, which is the fan's middle ray.  A fan ray starts ON the corner,
    so the ray-into-the-window and the full chord are the same segment; no
    special-casing is needed to reproduce ray_polyline's clipping.
    """
    t = np.deg2rad(np.linspace(0.0, 90.0, n_lines + 2)[1:-1])
    theta = t + np.pi / 2.0
    rho = np.cos(t) - np.sin(t)
    return np.stack([rho, theta % np.pi], axis=1)


FAMILIES = {
    # The method under comparison.  Named "rays" to match
    # dqd.study.sampling.STRATEGIES exactly — one vocabulary across both
    # repositories and both papers.
    "rays": family_rays,
    # Verification only.  These are NOT arms of the comparison; they exist so
    # tests/test_geometry.py can show the action space contains the other
    # geometries the baseline measured, which is what rules out "the agent
    # won because it was handed a better measurement primitive".
    "parallel_diag": family_parallel_diag,
    "hcuts": family_hcuts,
    "vcuts": family_vcuts,
}
