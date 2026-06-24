"""Staged global bundle adjustment for the rig.

Residual composition mirrors MC-Calib's ``UniversalReprojectionError``
(OptimizationCeres.h:681): ``X_cam = T_c_g @ T_g_o @ X_obj``, then ``model.project``.
The board-in-object poses are baked into ``Object3D.pts_3d`` (fixed here); the optimizer
refines the camera extrinsics ``T_c_g`` (ref camera held fixed), the per-frame object
poses ``T_g_o``, and — when ``fix_intrinsics=False`` — the per-camera intrinsics.

Pose retraction follows ``calib.bundle``: ``R <- R @ so3_exp(δω)``, ``t <- t + δt`` with
the local tangent ordered ``[δω(3), δt(3)]``, so the projection Jacobian chains cleanly
through ``-R[X]_×`` and ``I`` with no ``J_r`` term. Huber loss, δ=1.0 px
(CameraGroup.cpp:288). Solved densely via :func:`core.optimize.lm_solve` — the parameter
count is small at calibration scale (a handful of cameras, hundreds of object poses).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from ..calib.bundle import _skew_batch
from ..core.lie import so3_exp
from ..core.optimize import lm_solve, schur_lm
from .types import ObjectObs, RigState

Key = Tuple[int, int]


def _state_from_rig(rig: RigState) -> dict:
    return {
        "cam_R": {c: rig.T_c_g[c][:3, :3].copy() for c in rig.T_c_g},
        "cam_t": {c: rig.T_c_g[c][:3, 3].copy() for c in rig.T_c_g},
        "obj_R": {k: rig.object_poses[k][:3, :3].copy() for k in rig.object_poses},
        "obj_t": {k: rig.object_poses[k][:3, 3].copy() for k in rig.object_poses},
        "intr": {c: np.asarray(rig.cameras[c].params, float).copy() for c in rig.cameras},
    }


def _rig_from_state(rig: RigState, state: dict) -> RigState:
    out = copy.copy(rig)
    out.T_c_g = {}
    for c in state["cam_R"]:
        T = np.eye(4)
        T[:3, :3] = state["cam_R"][c]
        T[:3, 3] = state["cam_t"][c]
        out.T_c_g[c] = T
    out.object_poses = {}
    for k in state["obj_R"]:
        T = np.eye(4)
        T[:3, :3] = state["obj_R"][k]
        T[:3, 3] = state["obj_t"][k]
        out.object_poses[k] = T
    out.cameras = {c: type(rig.cameras[c]).from_params(state["intr"][c]) for c in rig.cameras}
    return out


def build_problem(rig: RigState, object_obs: List[ObjectObs], *,
                  fix_intrinsics: bool = True):
    """Assemble the BA callbacks ``(state0, residual, jacobian, retract, K)`` for one
    pass, without solving. ``residual``/``jacobian``/``retract`` follow the
    :func:`core.optimize.lm_solve` contract; ``K`` is the tangent dimension. Exposed so
    tests can finite-difference-check the analytic Jacobian (the chain in §9.1 of the
    implementation doc)."""
    ref_cam = rig.ref_cam_id
    cam_ids = [c for c in sorted(rig.cameras) if c != ref_cam]
    obj_keys = sorted(rig.object_poses)
    classes = {c: type(rig.cameras[c]) for c in rig.cameras}
    Pn = {c: len(classes[c].param_names) for c in rig.cameras}
    bounds = {c: classes[c].param_bounds() for c in rig.cameras}

    # tangent column layout
    col, cam_col, obj_col, intr_col = 0, {}, {}, {}
    for c in cam_ids:
        cam_col[c] = col; col += 6
    for k in obj_keys:
        obj_col[k] = col; col += 6
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            intr_col[c] = col; col += Pn[c]
    K = col

    # precompute per-observation object points (board poses baked into pts_3d)
    obs_data = []
    total_rows = 0
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        if (o.object_id, o.frame_id) not in rig.object_poses:  # frame never got a pose
            continue
        Xo = rig.objects[o.object_id].pts_3d[o.point_rows]      # (N,3) object frame
        obs_data.append((o, Xo))
        total_rows += 2 * len(Xo)

    def _project_all(state):
        out = []
        for o, Xo in obs_data:
            key = (o.object_id, o.frame_id)
            Xg = (state["obj_R"][key] @ Xo.T).T + state["obj_t"][key]
            Xc = (state["cam_R"][o.cam_id] @ Xg.T).T + state["cam_t"][o.cam_id]
            out.append((o, Xo, key, Xg, Xc))
        return out

    def residual(state):
        m_cache = {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}
        r = np.zeros(total_rows)
        row = 0
        for o, Xo, key, Xg, Xc in _project_all(state):
            uv, valid = m_cache[o.cam_id].project(Xc)
            diff = np.zeros_like(o.pts_2d, float)
            diff[valid] = uv[valid] - o.pts_2d[valid]
            r[row:row + 2 * len(Xo)] = diff.ravel()
            row += 2 * len(Xo)
        return r

    def jacobian(state):
        m_cache = {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}
        J = np.zeros((total_rows, K))
        row = 0
        for o, Xo, key, Xg, Xc in _project_all(state):
            N = len(Xo)
            cam = o.cam_id
            R_cam = state["cam_R"][cam]
            R_obj = state["obj_R"][key]
            uv, J_point, J_param, valid = m_cache[cam].project_jacobian(Xc)
            mask = valid[:, None, None].astype(float)
            Jp = J_point * mask                                  # (N,2,3)
            # object pose: dXc/dω = R_cam @ (-R_obj[Xo]_x); dXc/dt = R_cam
            dXc_dw_o = -np.einsum('ij,njk->nik', R_cam @ R_obj, _skew_batch(Xo))
            Jw_o = np.einsum('nij,njc->nic', Jp, dXc_dw_o)       # (N,2,3)
            Jt_o = np.einsum('nij,jc->nic', Jp, R_cam)
            c0 = obj_col[key]
            J[row:row + 2 * N, c0:c0 + 3] = Jw_o.reshape(2 * N, 3)
            J[row:row + 2 * N, c0 + 3:c0 + 6] = Jt_o.reshape(2 * N, 3)
            # camera extrinsic (skip ref): dXc/dω = -R_cam[Xg]_x; dXc/dt = I
            if cam in cam_col:
                dXc_dw_c = -np.einsum('ij,njk->nik', R_cam, _skew_batch(Xg))
                Jw_c = np.einsum('nij,njc->nic', Jp, dXc_dw_c)
                Jt_c = Jp                                        # J_point @ I
                cc = cam_col[cam]
                J[row:row + 2 * N, cc:cc + 3] = Jw_c.reshape(2 * N, 3)
                J[row:row + 2 * N, cc + 3:cc + 6] = Jt_c.reshape(2 * N, 3)
            # intrinsics
            if not fix_intrinsics:
                ic = intr_col[cam]
                J[row:row + 2 * N, ic:ic + Pn[cam]] = (J_param * mask).reshape(2 * N, Pn[cam])
            row += 2 * N
        return J

    def retract(state, d):
        s = {k: (v.copy() if isinstance(v, np.ndarray) else dict(v)) for k, v in state.items()}
        s["cam_R"], s["cam_t"] = dict(state["cam_R"]), dict(state["cam_t"])
        s["obj_R"], s["obj_t"] = dict(state["obj_R"]), dict(state["obj_t"])
        s["intr"] = dict(state["intr"])
        for c in cam_ids:
            o = cam_col[c]
            s["cam_R"][c] = state["cam_R"][c] @ so3_exp(d[o:o + 3])
            s["cam_t"][c] = state["cam_t"][c] + d[o + 3:o + 6]
        for k in obj_keys:
            o = obj_col[k]
            s["obj_R"][k] = state["obj_R"][k] @ so3_exp(d[o:o + 3])
            s["obj_t"][k] = state["obj_t"][k] + d[o + 3:o + 6]
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                lb, ub = bounds[c]
                s["intr"][c] = np.clip(state["intr"][c] + d[ic:ic + Pn[c]], lb, ub)
        return s

    return _state_from_rig(rig), residual, jacobian, retract, K


def _obs_blocks(model, R_cam, R_obj, Xo, Xg, Xc, pts_2d, want_intr):
    """Per-observation residual + Jacobian blocks, the single source of the BA chain
    (board baked into Xo): returns ``(r (2N,), Jw_o, Jt_o (2N,3), Jw_c, Jt_c (2N,3),
    J_param (2N,P) or None)``. Object pose: ∂Xc/∂ω = R_cam R_obj(-[Xo]_x), ∂Xc/∂t = R_cam.
    Camera: ∂Xc/∂ω = -R_cam[Xg]_x, ∂Xc/∂t = I."""
    N = len(Xo)
    uv, J_point, J_param, valid = model.project_jacobian(Xc)
    mask = valid[:, None, None].astype(float)
    Jp = J_point * mask
    r = np.zeros((N, 2))
    r[valid] = uv[valid] - pts_2d[valid]
    dXc_dw_o = -np.einsum('ij,njk->nik', R_cam @ R_obj, _skew_batch(Xo))
    Jw_o = np.einsum('nij,njc->nic', Jp, dXc_dw_o).reshape(2 * N, 3)
    Jt_o = np.einsum('nij,jc->nic', Jp, R_cam).reshape(2 * N, 3)
    dXc_dw_c = -np.einsum('ij,njk->nik', R_cam, _skew_batch(Xg))
    Jw_c = np.einsum('nij,njc->nic', Jp, dXc_dw_c).reshape(2 * N, 3)
    Jt_c = Jp.reshape(2 * N, 3)
    Jpar = (J_param * mask).reshape(2 * N, -1) if want_intr else None
    return r.ravel(), Jw_o, Jt_o, Jw_c, Jt_c, Jpar


def build_schur_problem(rig: RigState, object_obs: List[ObjectObs], *,
                        fix_intrinsics: bool = True):
    """Assemble the rig BA for :func:`core.optimize.schur_lm`, mapping the **per-frame
    object poses to the eliminated block-diagonal ``local`` blocks** and
    ``{camera extrinsics, intrinsics}`` to the ``shared`` block.

    Each reprojection residual touches exactly one object pose (its frame) plus a slice of
    the shared state, so the Hessian is the block-arrow that the Schur trick collapses:
    eliminate every 6-DoF object pose with a 6×6 inverse, solve the small shared system,
    back-substitute. This is the sparse analogue of the dense :func:`build_problem` and
    the source of the speed-up on rigs with many frames (v-slam Ch.8 / lio-slam SMW).

    Returns ``(state0, residual, linearize, retract, shared_dim, n_groups)``.
    """
    ref_cam = rig.ref_cam_id
    cam_ids = [c for c in sorted(rig.cameras) if c != ref_cam]
    classes = {c: type(rig.cameras[c]) for c in rig.cameras}
    Pn = {c: len(classes[c].param_names) for c in rig.cameras}
    bounds = {c: classes[c].param_bounds() for c in rig.cameras}

    # shared layout: non-ref camera extrinsics, then (optionally) per-camera intrinsics
    col, cam_col, intr_col = 0, {}, {}
    for c in cam_ids:
        cam_col[c] = col; col += 6
    if not fix_intrinsics:
        for c in sorted(rig.cameras):
            intr_col[c] = col; col += Pn[c]
    shared_dim = col

    # groups = object poses; gather each group's observations once
    groups = sorted(rig.object_poses)
    gobs = {k: [] for k in groups}
    for o in object_obs:
        k = (o.object_id, o.frame_id)
        if o.cam_id in rig.cameras and k in rig.object_poses:
            gobs[k].append((o, rig.objects[o.object_id].pts_3d[o.point_rows]))
    n_groups = len(groups)

    def _models(state):
        return {c: classes[c].from_params(state["intr"][c]) for c in rig.cameras}

    def _xc(state, o, Xo, key):
        Xg = (state["obj_R"][key] @ Xo.T).T + state["obj_t"][key]
        Xc = (state["cam_R"][o.cam_id] @ Xg.T).T + state["cam_t"][o.cam_id]
        return Xg, Xc

    def residual(state):
        mc = _models(state)
        out = []
        for k in groups:
            for o, Xo in gobs[k]:
                _, Xc = _xc(state, o, Xo, k)
                uv, valid = mc[o.cam_id].project(Xc)
                d = np.zeros_like(o.pts_2d)
                d[valid] = uv[valid] - o.pts_2d[valid]
                out.append(d.ravel())
        return np.concatenate(out) if out else np.zeros(0)

    def linearize(state):
        mc = _models(state)
        r_list, A_list, B_list = [], [], []
        for k in groups:
            R_obj = state["obj_R"][k]
            rs, As, Bs = [], [], []
            for o, Xo in gobs[k]:
                cam = o.cam_id
                Xg, Xc = _xc(state, o, Xo, k)
                r, Jw_o, Jt_o, Jw_c, Jt_c, Jpar = _obs_blocks(
                    mc[cam], state["cam_R"][cam], R_obj, Xo, Xg, Xc, o.pts_2d,
                    not fix_intrinsics)
                m = len(r)
                A = np.zeros((m, shared_dim))
                if cam in cam_col:
                    cc = cam_col[cam]
                    A[:, cc:cc + 3] = Jw_c
                    A[:, cc + 3:cc + 6] = Jt_c
                if not fix_intrinsics:
                    ic = intr_col[cam]
                    A[:, ic:ic + Pn[cam]] = Jpar
                B = np.empty((m, 6))
                B[:, :3] = Jw_o
                B[:, 3:] = Jt_o
                rs.append(r); As.append(A); Bs.append(B)
            r_list.append(np.concatenate(rs))
            A_list.append(np.vstack(As))
            B_list.append(np.vstack(Bs))
        return r_list, A_list, B_list

    def retract(state, d_shared, d_local):
        s = dict(state)
        s["cam_R"], s["cam_t"] = dict(state["cam_R"]), dict(state["cam_t"])
        s["obj_R"], s["obj_t"] = dict(state["obj_R"]), dict(state["obj_t"])
        s["intr"] = dict(state["intr"])
        for c in cam_ids:
            o = cam_col[c]
            s["cam_R"][c] = state["cam_R"][c] @ so3_exp(d_shared[o:o + 3])
            s["cam_t"][c] = state["cam_t"][c] + d_shared[o + 3:o + 6]
        if not fix_intrinsics:
            for c in sorted(rig.cameras):
                ic = intr_col[c]
                lb, ub = bounds[c]
                s["intr"][c] = np.clip(state["intr"][c] + d_shared[ic:ic + Pn[c]], lb, ub)
        for i, k in enumerate(groups):
            s["obj_R"][k] = state["obj_R"][k] @ so3_exp(d_local[i, :3])
            s["obj_t"][k] = state["obj_t"][k] + d_local[i, 3:]
        return s

    return _state_from_rig(rig), residual, linearize, retract, shared_dim, n_groups


def refine(rig: RigState, object_obs: List[ObjectObs], *,
           fix_intrinsics: bool = True, max_iter: int = 60,
           robust_kernel: str = "huber", robust_scale="auto", gnc_iters: int = 0,
           gnc_start: float = 0.0, verbose: bool = False, sparse: bool = True) -> RigState:
    """One BA pass. Returns a refined copy of ``rig``.

    ``fix_intrinsics=True`` reproduces ``refineCameraGroupAndObjects`` (poses only);
    ``False`` reproduces ``refineCameraGroupAndObjectsAndIntrinsics`` (full joint).

    **Robust weighting, no rejection.** Every observation is kept; outliers are
    down-weighted by IRLS (``w = ρ'(r)/r``). ``robust_scale="auto"`` re-estimates the
    inlier scale by MAD each iteration so the kernel adapts to the actual noise instead of
    a hand-set pixel threshold. A redescending ``cauchy`` kernel mutes gross outliers
    smoothly; ``gnc_iters>0`` anneals the scale from ``gnc_start`` down (graduated
    non-convexity) so the redescending fit cannot get trapped by a bad initial residual.

    ``sparse=True`` (default) Schur-eliminates the per-frame object poses
    (:func:`build_schur_problem` + :func:`core.optimize.schur_lm`); ``sparse=False`` uses
    the dense solver (kept for tests).
    """
    rk = dict(robust_kernel=robust_kernel, robust_scale=robust_scale,
              gnc_iters=gnc_iters, gnc_start=gnc_start)
    if sparse:
        state0, residual, linearize, retract, shared_dim, n_groups = build_schur_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics)
        if shared_dim == 0 or n_groups == 0:           # nothing shared to solve -> dense
            return refine(rig, object_obs, fix_intrinsics=fix_intrinsics, max_iter=max_iter,
                          verbose=verbose, sparse=False, **rk)
        res = schur_lm(state0, residual, linearize, retract, n_groups=n_groups,
                       shared_dim=shared_dim, local_dim=6, block=2, max_iter=max_iter, **rk)
    else:
        state0, residual, jacobian, retract, K = build_problem(
            rig, object_obs, fix_intrinsics=fix_intrinsics)
        if K == 0:
            return rig
        res = lm_solve(state0, residual, jacobian, retract, block=2, max_iter=max_iter, **rk)
    if verbose:
        print(f"  BA: rms {res.rms:.4f}px iters={res.iterations} "
              f"intr={'free' if not fix_intrinsics else 'fixed'} "
              f"{'sparse' if sparse else 'dense'} kernel={robust_kernel}")
    return _rig_from_state(rig, res.state)


def _per_obs_errors(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, np.ndarray]:
    """Per-camera array of per-point reprojection errors (px)."""
    errs: Dict[int, list] = {}
    for o in object_obs:
        if o.cam_id not in rig.cameras:
            continue
        key = (o.object_id, o.frame_id)
        if key not in rig.object_poses:
            continue
        Xo = rig.objects[o.object_id].pts_3d[o.point_rows]
        Xg = (rig.object_poses[key][:3, :3] @ Xo.T).T + rig.object_poses[key][:3, 3]
        Xc = (rig.T_c_g[o.cam_id][:3, :3] @ Xg.T).T + rig.T_c_g[o.cam_id][:3, 3]
        uv, valid = rig.cameras[o.cam_id].project(Xc)
        errs.setdefault(o.cam_id, []).append(
            np.linalg.norm(uv[valid] - o.pts_2d[valid], axis=1))
    return {c: np.concatenate(v) if v else np.zeros(0) for c, v in errs.items()}


def reprojection_rms(rig: RigState, object_obs: List[ObjectObs]) -> Dict[int, float]:
    """Per-camera reprojection RMS (px) over all observations."""
    return {c: float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan")
            for c, e in _per_obs_errors(rig, object_obs).items()}


def reprojection_metrics(rig: RigState, object_obs: List[ObjectObs],
                         inlier_px: float = 1.0) -> Dict[int, dict]:
    """Per-camera **robust** reprojection metrics. Naive all-corner RMS lies on a robust
    fit — it scores the size of the outliers the model deliberately down-weighted
    (docs/learn/robust_losses_and_evaluation.md). Report instead: ``median`` (50% break-
    down), ``inlier_rms`` (RMS over corners under ``inlier_px``), and ``inlier_frac``."""
    out = {}
    for c, e in _per_obs_errors(rig, object_obs).items():
        if not len(e):
            out[c] = dict(median=float("nan"), inlier_rms=float("nan"), inlier_frac=0.0,
                          rms=float("nan"))
            continue
        inl = e < inlier_px
        out[c] = dict(
            median=float(np.median(e)),
            inlier_rms=float(np.sqrt(np.mean(e[inl] ** 2))) if inl.any() else float("nan"),
            inlier_frac=float(inl.mean()),
            rms=float(np.sqrt(np.mean(e ** 2))),
        )
    return out
