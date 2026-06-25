"""Outlier handling by reweighting (not rejection) — the >50% improvement, regression-guarded.

Mirrors ``scripts/benchmark_outliers.py`` at small scale: robust IRLS pose vs naive L2 under
gross outliers, and the studentized-leverage gain on a self-masking high-leverage corner.
"""
import cv2
import numpy as np

from ds_msp.core.lie import se3_exp
from ds_msp.models.radtan import RadTanModel
from ds_msp.rig.pose_init import robust_pose_irls

F, W, H = 800.0, 1280, 960


def _err(T, Tgt):
    return np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ Tgt[:3, :3]) - 1) / 2, -1, 1)))


def _l2(model, X, uv):
    rays, ok = model.unproject(uv)
    ok = ok & (rays[:, 2] > 1e-6)
    pn = (rays[ok, :2] / rays[ok, 2:3]).astype(np.float64)
    okp, rv, tv = cv2.solvePnP(X[ok].astype(np.float64), pn, np.eye(3), None,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel()
    return T


def test_reweight_beats_naive_l2_by_more_than_half_at_20pct():
    rng = np.random.default_rng(0)
    l2e, rwe = [], []
    for _ in range(20):
        model = RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
        X = rng.uniform(-0.35, 0.35, size=(40, 3)); X[:, 2] += 2.0
        Tgt = se3_exp(np.r_[rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.3, 0.3, 3)])
        uv, _ = model.project((Tgt[:3, :3] @ X.T).T + Tgt[:3, 3])
        uv += rng.normal(scale=0.3, size=uv.shape)
        bad = rng.random(40) < 0.20
        uv[bad] += rng.uniform(-50, 50, size=(int(bad.sum()), 2))
        l2e.append(_err(_l2(model, X, uv), Tgt))
        rwe.append(_err(robust_pose_irls(model, X, uv), Tgt))
    l2m, rwm = np.median(l2e), np.median(rwe)
    assert rwm < 0.5 * l2m, f"reweight {rwm:.3f} not <50% of L2 {l2m:.3f}"
    assert rwm < 1.0                                        # robust pose stays sub-degree


def test_studentize_helps_self_masking_leverage_outlier():
    rng = np.random.default_rng(3)
    ns, st = [], []
    for _ in range(40):
        model = RadTanModel(F, F, W / 2, H / 2, -0.05, 0.01, 0.0, 0.0, 0.0)
        X = rng.uniform(-0.25, 0.25, size=(10, 3)); X[:, 2] += 2.0
        X[0] = [1.2, 1.2, 1.3]                              # high-leverage corner
        Tgt = se3_exp(np.r_[rng.uniform(-0.12, 0.12, 3), rng.uniform(-0.2, 0.2, 3)])
        uv, val = model.project((Tgt[:3, :3] @ X.T).T + Tgt[:3, 3])
        if not val.all():
            continue
        uv += rng.normal(scale=0.3, size=uv.shape)
        uv[0] += [10.0, -10.0]                              # modest, self-masking shift
        Tn = robust_pose_irls(model, X, uv, studentize=False)
        Ts = robust_pose_irls(model, X, uv, studentize=True)
        ns.append(_err(Tn, Tgt)); st.append(_err(Ts, Tgt))
    assert np.median(st) < 0.5 * np.median(ns), \
        f"studentize {np.median(st):.3f} not <50% of plain {np.median(ns):.3f}"
