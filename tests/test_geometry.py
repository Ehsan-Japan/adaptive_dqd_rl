"""
test_geometry.py — the action space contains every baseline geometry.

This is not a unit test.  It is the load-bearing claim of the comparison,
executed:

    the learned policy is not given a richer measurement primitive than the
    corner fan.  It is given the same primitive and allowed to place it.

If `family_rays(8)` did not reproduce, pixel for pixel, the eight rays that
`dqd.ml.ray_peaks` fires at 8 x 60, then any improvement the agent showed
could be the new rasteriser rather than the new policy, and the paper would
be measuring the wrong thing.  So it is asserted, on every run, against the
baseline's own code rather than against a copy of it.

    pytest tests/ -v

The baseline repo must be importable — submodule or $DQD_BASELINE.
"""
import numpy as np
import pytest

from adaptive_dqd.eval.metrics import BASELINE_SRC          # noqa: F401
from adaptive_dqd.geometry import lines

from dqd.ml.ray_peaks import fan_angles, ray_polyline, voltage_to_pixel
from dqd.study import sampling

SHAPE = (100, 100)
N_LINES, N_POINTS = 8, 60

# A window matching the baseline's convention: 2 x 2, arbitrary origin.  The
# arbitrary origin is the point — normalised coordinates must reproduce the
# baseline geometry for ANY per-device offset, since offset_scale = 0.35
# shifts every device differently.
UX = np.linspace(-1.37, 0.63, SHAPE[1])
UY = np.linspace(-0.91, 1.09, SHAPE[0])


def _baseline_pixels(pts):
    row, col = voltage_to_pixel(pts[:, 0], pts[:, 1], UX, UY)
    return set(zip(row.tolist(), col.tolist()))


def _ours(rho, theta):
    rc = lines.sweep(rho, theta, N_POINTS, SHAPE)
    return set(map(tuple, rc.tolist()))


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def test_rays_matches_baseline():
    """Every fan ray, as a chord, hits the pixels the baseline's ray hits."""
    table = lines.family_rays(N_LINES)
    for (rho, theta), angle in zip(table, fan_angles(N_LINES)):
        ours = _ours(rho, theta)
        theirs = _baseline_pixels(ray_polyline(angle, N_POINTS, UX, UY))
        # Sampling is nearest-cell on both sides and the endpoints are the
        # same corner, so the sets should be equal up to rounding at the two
        # ends of the chord.  Anything below 0.98 means the geometry differs.
        assert _jaccard(ours, theirs) > 0.98, (
            f"fan ray at {angle:.1f} deg: Jaccard {_jaccard(ours, theirs):.3f}")


def test_parallel_diag_matches_baseline():
    ours = set()
    for rho, theta in lines.family_parallel_diag(N_LINES):
        ours |= _ours(rho, theta)
    theirs = set()
    for pts in sampling.parallel_diag_lines(N_LINES, N_POINTS, UX, UY):
        theirs |= _baseline_pixels(pts)
    assert _jaccard(ours, theirs) > 0.9


def test_hcuts_matches_baseline():
    ours = set()
    for rho, theta in lines.family_hcuts(N_LINES):
        ours |= _ours(rho, theta)
    theirs = set()
    for pts in sampling.hcut_lines(N_LINES, N_POINTS, UX, UY):
        theirs |= _baseline_pixels(pts)
    assert _jaccard(ours, theirs) > 0.95


def test_every_action_is_a_real_chord():
    """No action in [0,1]^2 maps to a line that misses the window.

    PPO explores the corners of its output space early; if any of them were
    invalid the agent would spend its first thousand episodes learning to
    avoid a region that should not exist.  Checked on a dense grid plus the
    exact corners.
    """
    us = np.linspace(0.0, 1.0, 41)
    for u in us:
        for v in us:
            rho, theta = lines.unit_to_line(u, v)
            assert len(lines.chord(rho, theta, N_POINTS)) == N_POINTS, (u, v)


def test_unit_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u, v = rng.random(), rng.random()
        rho, theta = lines.unit_to_line(u, v)
        u2, v2 = lines.line_to_unit(rho, theta)
        assert abs(u - u2) < 1e-9 and abs(v - v2) < 1e-9


def test_rays_is_representable_as_a_policy():
    """
    The margin must not clip any fan ray at the budgets the study uses.

    `policies.rays()` routes through line_to_unit -> unit_to_line, so if the
    margin ever clipped a ray, the fan arm evaluated HERE would quietly stop
    being the fan arm evaluated in the baseline paper — the exact silent
    divergence this repository exists to prevent.
    """
    for n in (4, 5, 6, 7, 8, 16, 32):
        for rho, theta in lines.family_rays(n):
            u, v = lines.line_to_unit(rho, theta)
            assert 0.0 < u < 1.0, f"{n} rays: fan ray clipped by MARGIN"
            rho2, theta2 = lines.unit_to_line(u, v)
            assert abs(rho - rho2) < 1e-9 and abs(theta - theta2) < 1e-9
