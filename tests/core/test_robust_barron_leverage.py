"""Barron adaptive kernel identities + studentized-leverage unmasking (diffpnp port)."""
import numpy as np

from ds_msp.core.robust import (robust_cost, robust_weight, studentized_sq)


def test_barron_reduces_to_known_kernels():
    s = np.linspace(0.0, 50.0, 40)
    c = 2.0
    # alpha=2 -> L2 weight (omega == 1); exact up to the removable-singularity guard eps
    assert np.allclose(robust_weight(s, "barron", c, alpha=2.0), 1.0, atol=1e-3)
    # alpha=0 -> Cauchy-class weight 1/(1+s/(c^2*b)) with b=|0-2|=2 (Barron's c'=sqrt(2)c)
    w0 = robust_weight(s, "barron", c, alpha=0.0)
    assert np.allclose(w0, 1.0 / (1.0 + s / (c * c * 2.0)), atol=1e-4)
    # alpha=-2 -> Geman-McClure-class; weight monotonically decreasing, in (0,1]
    wgm = robust_weight(s, "barron", c, alpha=-2.0)
    assert wgm[0] == 1.0 and np.all(np.diff(wgm) <= 1e-9) and np.all(wgm > 0)


def test_barron_cost_monotone_and_redescending_weight():
    s = np.linspace(0.0, 100.0, 50)
    cost = robust_cost(s, "barron", 2.0, alpha=-2.0)
    assert np.all(np.diff(cost) >= -1e-9)                    # cost non-decreasing in s
    w = robust_weight(s, "barron", 2.0, alpha=-2.0)
    assert w[-1] < 0.05                                      # gross outliers strongly muted


def test_studentized_residual_unmasks_high_leverage_point():
    """A self-masking high-leverage point: its raw residual is small, but its hat value is
    large, so the studentized residual inflates it well above the inliers."""
    rng = np.random.default_rng(0)
    n, p = 40, 6
    J = rng.normal(size=(2 * n, p))
    # make point 0 high-leverage: scale its two rows up a lot
    J[0:2] *= 25.0
    r = rng.normal(scale=0.3, size=2 * n)                    # all residuals small/inlier-like
    s_tilde = studentized_sq(J, r, block=2)
    s_raw = (r.reshape(n, 2) ** 2).sum(1)
    # raw residual of the leverage point is ordinary; studentized is the largest by far
    assert s_raw[0] < np.median(s_raw) * 5
    assert np.argmax(s_tilde) == 0
    assert s_tilde[0] > 3.0 * np.median(s_tilde)


def test_studentized_equals_raw_when_no_leverage():
    """With many well-spread points, hat values are ~p/n«1, so s̃ ≈ ‖r‖² for inliers."""
    rng = np.random.default_rng(1)
    n, p = 300, 6
    J = rng.normal(size=(2 * n, p))
    r = rng.normal(scale=0.5, size=2 * n)
    s_tilde = studentized_sq(J, r, block=2)
    s_raw = (r.reshape(n, 2) ** 2).sum(1)
    assert np.median(np.abs(s_tilde - s_raw) / (s_raw + 1e-9)) < 0.1
