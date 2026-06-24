"""Hand-eye linking for non-overlapping camera groups
(``handeyeBootstraptTranslationCalibration``, geometrytools.cpp:621).

When ``rig.extrinsics`` yields more than one connected component (camera groups that
never co-observe an object), the inter-group transform cannot come from direct
covisibility. It is recovered from the *motion* of the object as seen by a camera in
each group: cluster the motions, run repeated Tsai hand-eye solves on diverse motion
subsets, gate by rotational consistency (15 deg), and aggregate.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from .averaging import average_rotation, average_translation


def _rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def handeye_bootstrap(poses_a: List[np.ndarray], poses_b: List[np.ndarray], *,
                      nb_cluster: int = 20, nb_pick: int = 6, nb_it: int = 200,
                      gate_deg: float = 15.0, seed: int = 0) -> np.ndarray:
    """Estimate ``T_b_a`` (group-a-ref -> group-b-ref) from paired absolute object poses
    ``poses_a[i]`` / ``poses_b[i]`` (object->camera 4x4) observed in the same frame.
    """
    n = len(poses_a)
    if n < 3:
        return _single_handeye(poses_a, poses_b)
    feats = np.array([np.r_[Ta[:3, 3], Tb[:3, 3]] for Ta, Tb in zip(poses_a, poses_b)],
                     np.float32)
    k = min(nb_cluster, n)
    _, labels, _ = cv2.kmeans(
        feats, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01), 5,
        cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    by_cluster: Dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(i)

    rng = np.random.default_rng(seed)
    Rs, ts = [], []
    for _ in range(nb_it):
        clusters = rng.permutation(list(by_cluster))[:nb_pick]
        sel = [int(rng.choice(by_cluster[c])) for c in clusters]
        if len(sel) < 3:
            continue
        # motions between consecutive selected poses
        Ra_g, ta_g, Rb_g, tb_g = [], [], [], []
        for u, v in zip(sel[:-1], sel[1:]):
            Ma = poses_a[v] @ np.linalg.inv(poses_a[u])
            Mb = poses_b[v] @ np.linalg.inv(poses_b[u])
            Ra_g.append(Ma[:3, :3]); ta_g.append(Ma[:3, 3])
            Rb_g.append(Mb[:3, :3]); tb_g.append(Mb[:3, 3])
        try:
            Rx, tx = cv2.calibrateHandEye(Ra_g, ta_g, Rb_g, tb_g,
                                          method=cv2.CALIB_HAND_EYE_TSAI)
        except cv2.error:
            continue
        # consistency: X should satisfy Mb @ X ~ X @ Ma for each motion
        X = np.eye(4); X[:3, :3] = Rx; X[:3, 3] = tx.ravel()
        worst = 0.0
        for Ma_R, Ma_t, Mb_R, Mb_t in zip(Ra_g, ta_g, Rb_g, tb_g):
            Ma = np.eye(4); Ma[:3, :3] = Ma_R; Ma[:3, 3] = Ma_t
            Mb = np.eye(4); Mb[:3, :3] = Mb_R; Mb[:3, 3] = Mb_t
            E = np.linalg.inv(Mb @ X) @ (X @ Ma)
            worst = max(worst, _rot_angle_deg(E[:3, :3]))
        if worst < gate_deg:
            Rs.append(Rx); ts.append(np.asarray(tx).ravel())

    if len(Rs) > 3:
        R = average_rotation(Rs)
        t = average_translation(np.array(ts))
    else:
        return _single_handeye(poses_a, poses_b)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def _single_handeye(poses_a, poses_b) -> np.ndarray:
    """Fallback Horaud-style solve over all available motions."""
    Ra, ta, Rb, tb = [], [], [], []
    for u, v in zip(range(len(poses_a) - 1), range(1, len(poses_a))):
        Ma = poses_a[v] @ np.linalg.inv(poses_a[u])
        Mb = poses_b[v] @ np.linalg.inv(poses_b[u])
        Ra.append(Ma[:3, :3]); ta.append(Ma[:3, 3])
        Rb.append(Mb[:3, :3]); tb.append(Mb[:3, 3])
    Rx, tx = cv2.calibrateHandEye(Ra, ta, Rb, tb, method=cv2.CALIB_HAND_EYE_HORAUD)
    T = np.eye(4); T[:3, :3] = Rx; T[:3, 3] = tx.ravel()
    return T


def link_groups(groups: List[List[int]], extr: Dict[int, np.ndarray],
                object_obs) -> Dict[int, np.ndarray]:
    """Merge non-overlapping camera groups into one frame.

    Each group's cameras keep their intra-group ``T_c_g`` (relative to the group's own
    reference). We estimate the transform from each non-base group's reference camera to
    the base group's reference camera via hand-eye on the object motion seen by both,
    then re-base every camera in that group. Returns a flat ``{cam: T_c_g0}``.
    """
    base = groups[0]
    base_ref = base[0]
    out = {c: extr[c].copy() for c in base}

    # object pose per (group-ref camera, frame), needed to pair motions across groups
    def ref_poses(group):
        ref = group[0]
        poses = {}
        for o in object_obs:
            if o.cam_id == ref and o.T_c_o is not None:
                poses[o.frame_id] = o.T_c_o
        return ref, poses

    base_ref, base_poses = ref_poses(base)
    for group in groups[1:]:
        gref, gposes = ref_poses(group)
        common = sorted(set(base_poses) & set(gposes))
        if len(common) >= 3:
            pa = [base_poses[f] for f in common]
            pb = [gposes[f] for f in common]
            T_gref_baseref = handeye_bootstrap(pa, pb)         # base-ref -> g-ref
        else:
            T_gref_baseref = np.eye(4)
        # re-base: camera in group has T_c_gref; want T_c_baseref = T_c_gref @ T_gref_baseref
        for c in group:
            out[c] = extr[c] @ T_gref_baseref
    return out
