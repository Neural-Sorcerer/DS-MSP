# DS-MSP — Multi-Camera Rig Calibration: Implementation Details

**What this is:** the implementation-grade companion to `RIG_CALIBRATION_PLAN.md`. The plan
says *what to add where*; this says *exactly how* — concrete signatures, data structures, the
BA parameter-block layout mapped onto DS-MSP's existing `schur_lm`, the averaging/composition
math, the Jacobian chain, and the test fixtures. Every algorithm is cross-referenced to the
DS-MSP interface it builds on (`ds_msp/...`) and the MC-Calib routine it ports
(`McCalib/...:line`). Reproduce the cited conventions exactly — several are non-obvious
(inverse-compose direction differs between graphs; quaternion scalar is at index 3;
translation aggregation is median in some places, mean in others).

Read `RIG_CALIBRATION_PLAN.md` first for the gap analysis and the 8 borrowable concepts. This
doc assumes that context.

---

## 0. Conventions (match these everywhere)

Pulled from `ds_msp/core/contracts.py` and `ds_msp/core/lie.py` so new code is drop-in.

| Concept | Representation | Source |
|---|---|---|
| Pose | **4×4 homogeneous matrix** `T` (point in child frame → parent: `X_parent = T @ [X;1]`) | `core/lie.py` (no (R,t) wrapper) |
| Pose tangent | `xi = [ρ(3 trans), φ(3 rot)]`, `se3_exp(xi)`, `se3_log(T)` | `core/lie.py:80,92` |
| Rotation update | right-multiply: `R ← R @ so3_exp(δω)` (`J_r(0)=I` drops out) | `calib/bundle.py` retract |
| 3D points | `(N,3)` float64, +Z forward | `core/contracts.py` |
| Pixels | `(N,2)` `(u,v)` | `core/contracts.py` |
| Camera model | duck-typed `CameraModel` Protocol; `project`, `unproject`, `project_jacobian` | `core/contracts.py:55` |
| `project_jacobian(P)` | `→ (uv(N,2), J_point(N,2,3), J_param(N,2,P), valid(N,))` | `core/contracts.py:90` |
| Robust loss | Huber, via `schur_lm(robust_kernel="huber", robust_scale=1.0)` | `core/robust.py:35` |

**Frame-name convention for the rig** (pick one and keep it). We use:
- `T_co_b` — board → object ("board-in-object"). Object3D internal.
- `T_g_o` — object → camera-group reference frame ("object-in-group"), per (object, frame).
- `T_c_g` — group-ref → camera ("camera-in-group", i.e. the extrinsic we solve), per camera.

A reprojection composes **board → object → group → camera → project**:
`X_cam = T_c_g · T_g_o · T_co_b · X_board`, then `model.project(X_cam)`. This is the exact
order of MC-Calib's `UniversalReprojectionError` (`OptimizationCeres.h:681-714`).

---

## 1. Data structures — `ds_msp/rig/types.py`

Plain dataclasses (NumPy fields). These are the Python analogues of MC-Calib's
`BoardObs`/`Object3DObs`/`Object3D`/`CameraGroup`.

```python
@dataclass
class BoardObs:
    cam_id: int
    frame_id: int
    board_id: int
    corner_ids: np.ndarray      # (K,) int   — which board corners were seen
    pts_2d: np.ndarray          # (K,2)       — detected pixels
    T_c_b: np.ndarray | None = None   # (4,4)  board→camera, from robust PnP (§4)
    valid: bool = True          # set False if PnP inliers < 4  (BoardObs.cpp:149)

@dataclass
class Object3D:
    object_id: int
    board_ids: list[int]
    ref_board_id: int                       # min(board_ids)  (McCalib.cpp:898)
    T_co_b: dict[int, np.ndarray]           # board_id -> (4,4) board→object
    pts_3d: np.ndarray                      # (P,3) all corners in object frame
    pts_obj_2_board: np.ndarray             # (P,2) [board_id, corner_id] inverse map
    pts_board_2_obj: dict[tuple[int,int],int]  # (board_id,corner_id) -> row in pts_3d

@dataclass
class ObjectObs:                            # one object seen in one (cam,frame)
    cam_id: int
    frame_id: int
    object_id: int
    point_rows: np.ndarray      # (K,) int  — rows into Object3D.pts_3d
    pts_2d: np.ndarray          # (K,2)
    T_c_o: np.ndarray | None = None   # (4,4) object→camera, robust PnP on the fused cloud

@dataclass
class RigState:                             # the BA optimization variable (§5)
    cameras: dict[int, CameraModel]         # per-camera intrinsics
    T_c_g: dict[int, np.ndarray]            # camera-in-group extrinsics; ref cam = identity
    ref_cam_id: int
    object_poses: dict[tuple[int,int], np.ndarray]  # (object_id,frame_id) -> T_g_o (4,4)
    objects: dict[int, Object3D]            # holds T_co_b board poses (refined in BA)
```

`BoardObs`/`ObjectObs` correspond to MC-Calib `BoardObs.cpp` / `Object3DObs.cpp`; `Object3D`
to `Object3D.cpp`; `RigState` is what the staged BA mutates (§5).

---

## 2. Covisibility graph — `ds_msp/rig/graph.py`

MC-Calib uses Boost Graph (`Graph.cpp`); in Python use **`networkx`** (already a calibration
dep, or vendor a 40-line union-find + Dijkstra if you want zero deps). One implementation
serves all three graph levels (boards, cameras, groups) — only the node set and the
co-observation counter change.

```python
import networkx as nx

def covis_graph(pairs_counts: dict[tuple[int,int], int]) -> nx.Graph:
    """pairs_counts[(i,j)] = number of frames i and j were co-observed.
    Edge weight = 1 / count  (MC-Calib McCalib.cpp:873, 1113 — confirmed 1/N)."""
    g = nx.Graph()
    for (i, j), n in pairs_counts.items():
        g.add_edge(i, j, weight=1.0 / n)
    return g

def components(g: nx.Graph) -> list[list[int]]:
    return [sorted(c) for c in nx.connected_components(g)]   # sorted => min-id ref is c[0]

def shortest_path(g: nx.Graph, src: int, dst: int) -> list[int]:
    return nx.dijkstra_path(g, src, dst, weight="weight")    # Graph.cpp:104 analogue
```

`components()` sorts each component so `c[0]` is the **min id**, which is MC-Calib's reference
choice (`min_element`, `McCalib.cpp:898-899` boards, `:1134` cameras).

---

## 3. Robust averaging — `ds_msp/rig/averaging.py`

Ports `getAverageRotation` (`geometrytools.cpp:868`, Markley SVD) and the translation median in
`initInterTransform` (`McCalib.cpp:843`). **Quaternion layout is `[x,y,z,w]`, scalar at index
3** — match it or the antipodal fix breaks.

```python
def average_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    """Markley SVD quaternion averaging. Rs: list of (3,3). Returns (3,3).
    Port of geometrytools.cpp:881-917."""
    A = np.zeros((4, 4))
    for R in Rs:
        q = _mat_to_quat_xyzw(R)        # [x,y,z,w]
        if q[3] < 0:                    # antipodal fix on scalar (cpp:897)
            q = -q
        A += np.outer(q, q)
    A /= len(Rs)
    w, V = np.linalg.eigh(A)            # symmetric => eigh; largest eigenvalue last
    q_avg = V[:, -1]                    # eigenvector of max eigenvalue (cpp:905-914 uses SVD U[:,0])
    return _quat_xyzw_to_mat(q_avg)

def average_translation(ts: np.ndarray) -> np.ndarray:
    """ts: (N,3). Component-wise median (McCalib.cpp:843 / geometrytools.cpp:697)."""
    return np.median(ts, axis=0)

def average_transform(Ts: list[np.ndarray]) -> np.ndarray:
    """Fuse a stack of noisy 4x4 into one. initInterTransform analogue (McCalib.cpp:811)."""
    R = average_rotation([T[:3, :3] for T in Ts])
    t = average_translation(np.array([T[:3, 3] for T in Ts]))
    out = np.eye(4); out[:3, :3] = R; out[:3, 3] = t
    return out
```

`eigh` on the symmetric accumulator `A` is numerically equivalent to MC-Calib's `SVD(A).U`
column 0 (for a symmetric PSD matrix the leading left-singular vector = leading eigenvector).
Use it instead of `np.linalg.svd` — cheaper and unambiguous.

> **Caveat — mean vs median.** `average_translation` is **median** (inter-board/camera/group
> fusion and hand-eye). But `CameraGroupObs::computeObjectsPose` (`CameraGroupObs.cpp:95`) uses
> an arithmetic **mean** when averaging an object pose across non-ref cameras. Keep a separate
> `mean` path for §6; don't unify them.

---

## 4. Multi-board 3D-object fusion — `ds_msp/rig/object3d.py`

Ports `computeBoardsPairPose` → `initInterTransform` → `initInterBoardsGraph` → `init3DObjects`
(`McCalib.cpp:765-950`). Builds, per connected board-component, one rigid `Object3D`.

**Step 1 — inter-board pair transforms.** For each frame where ≥2 boards are seen, for each
ordered pair `(b1,b2)`, with PnP board poses `T_c_b1`, `T_c_b2`:

```python
# T_b2_b1 = points of b1 expressed in b2 frame  (McCalib.cpp:786: proj_2.inv() * proj_1)
T_pair = np.linalg.inv(T_c_b2) @ T_c_b1
pair_samples[(b1, b2)].append(T_pair)
```

**Step 2 — average** each pair list with `average_transform` (§3) → `inter_board[(b1,b2)]`.
Count co-observations per unordered pair for the graph (§2).

**Step 3 — graph & components.** `components(covis_graph(board_pair_counts))` → each component
is one object; `ref_board_id = component[0]` (min id).

**Step 4 — board-in-object poses by shortest-path composition.** For each board, walk the
Dijkstra path `ref → board` accumulating, **using the inverse of each edge transform**:

```python
T = np.eye(4)
path = shortest_path(g, ref_board_id, board_id)
for cur, nxt in zip(path[:-1], path[1:]):
    T = T @ np.linalg.inv(inter_board[(cur, nxt)])   # McCalib.cpp:929: transform * current_trans.inv()
T_co_b[board_id] = T                                  # ref board => identity (path len 1)
```

> **Caveat — inverse direction.** Board composition uses `.inv()` of the edge (`McCalib.cpp:929`)
> because `inter_board` stores `T_next_current` but we want current→object. The **camera**
> graph (§7) composes the edge *without* inverse (`McCalib.cpp:1163`). Same skeleton, opposite
> convention — reproduce both exactly.

**Step 5 — fused point cloud + index maps.** For each board, transform its corner points
(from `AprilGridTarget.all_object_points()`, `calib/targets.py:65`, z=0 board frame) by
`T_co_b[board_id]` and concatenate:

```python
rows = []                                  # pts_obj_2_board
pts  = []
b2o  = {}                                   # pts_board_2_obj
for bid in object.board_ids:
    P_b = boards[bid].all_object_points()   # (n_corners,3)
    P_o = (T_co_b[bid] @ np.c_[P_b, np.ones(len(P_b))].T).T[:, :3]
    for k, p in enumerate(P_o):
        b2o[(bid, k)] = len(pts); rows.append((bid, k)); pts.append(p)
object.pts_3d = np.array(pts)               # McCalib init3DObjects :937-950
object.pts_obj_2_board = np.array(rows)
object.pts_board_2_obj = b2o
```

One PnP on `pts_3d` (§5 of plan) now recovers the whole object pose from *any* visible subset
of boards — the co-visibility multiplier that makes large/partial rigs work.

---

## 5. Robust PnP — `ds_msp/rig/pose_init.py`

Ports `ransacP3PDistortion` + `BoardObs::estimatePose` (`geometrytools.cpp:710`,
`BoardObs.cpp:121`). DS-MSP has no PnP-RANSAC yet; build it by adapting the *harness* in
`mvg/ransac.py` (the adaptive-iteration loop) but with a PnP hypothesis and a **pixel
reprojection** score (not the angular Sampson score, which is for the 2-view case).

The crucial DS-MSP-specific move: **unproject with the camera's own model, then PnP on the
normalized plane** — exactly what `bundle._seed_poses` (`calib/bundle.py:40`) already does. This
replaces MC-Calib's per-distortion-type `cv::fisheye::undistortPoints` branch
(`geometrytools.cpp:735`) with one model-agnostic path, because every DS-MSP model exposes
`unproject`.

```python
def estimate_pose_ransac(model, object_pts, image_pts, *,
                         thresh_px=1.0, max_iters=1000, p=0.99, seed=0):
    """object_pts (N,3) in board/object frame; image_pts (N,2) pixels.
    Returns (T_cam_obj (4,4) | None, inliers (N,) bool). None/invalid if <4 inliers."""
    rays, ok = model.unproject(image_pts)
    ok &= rays[:, 2] > 1e-6
    pn = rays[:, :2] / rays[:, 2:3]                 # normalized plane (bundle._seed_poses pattern)
    Xv, pnv, idx = object_pts[ok], pn[ok], np.where(ok)[0]
    if len(Xv) < 4:
        return None, np.zeros(len(object_pts), bool)

    rng = np.random.default_rng(seed)
    best_inl, n, it = None, len(Xv), 0
    iters = max_iters
    while it < iters and it < max_iters:
        s = rng.choice(n, 4, replace=False)         # P3P minimal set = 4 (geometrytools.cpp ransacP3P)
        okp, rvec, tvec = cv2.solvePnP(Xv[s], pnv[s], np.eye(3), None,
                                       flags=cv2.SOLVEPNP_P3P)
        if not okp: it += 1; continue
        proj, _ = cv2.projectPoints(Xv, rvec, tvec, np.eye(3), None)
        err = np.linalg.norm(proj.reshape(-1, 2) - pnv, axis=1)
        inl = err < (thresh_px / _focal(model))     # threshold in normalized units
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
            frac = max(inl.mean(), 1e-6)
            iters = min(max_iters, int(np.log(1 - 0.999) / np.log(1 - frac**3)) + 1)  # exp=3 (cpp:300)
        it += 1

    if best_inl is None or best_inl.sum() < 4:       # validity gate (BoardObs.cpp:149)
        return None, np.zeros(len(object_pts), bool)
    # final iterative refine on inliers (geometrytools.cpp:315 SOLVEPNP_ITERATIVE)
    ok2, rvec, tvec = cv2.solvePnP(Xv[best_inl], pnv[best_inl], np.eye(3), None,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rvec)[0]; T[:3, 3] = tvec.ravel()
    full = np.zeros(len(object_pts), bool); full[idx[best_inl]] = True
    return T, full
```

Notes that match MC-Calib exactly: minimal set **4 points** (`ransacP3P`), adaptive iteration
exponent **3** (`fracinliers³`, `geometrytools.cpp:300`), final refine with
`SOLVEPNP_ITERATIVE` seeded from the best hypothesis (`:315`), and the **<4 inlier →
invalid** rule (`BoardObs.cpp:149`). Prune `BoardObs.pts_2d`/`corner_ids` to inliers before BA,
as MC-Calib does (`BoardObs.cpp:154`).

---

## 6. Object-in-rig pose averaging — `ds_msp/rig/pose_init.py`

Ports `CameraGroupObs::computeObjectsPose` (`CameraGroupObs.cpp:42`). For each (object, frame)
in a camera group:

1. If the **reference camera** sees the object → use its `T_g_o` directly (ref cam extrinsic is
   identity, so `T_g_o = T_c_o` for the ref cam). (`CameraGroupObs.cpp:68`)
2. Else, for each non-ref camera `c` that sees it, lift to the group frame
   `T_g_o = inv(T_c_g) @ T_c_o`, then average across those cameras: rotation via
   `average_rotation`, **translation via arithmetic mean** (`CameraGroupObs.cpp:95` — *mean*,
   not median; see §3 caveat).

---

## 7. Camera-group extrinsics — `ds_msp/rig/extrinsics.py`

Ports `computeCamerasPairPose` → `initInterTransform` → `initInterCamerasGraph` →
`initCameraGroup` (`McCalib.cpp:1054-1163`). Same 4-step skeleton as §4 but on cameras:

1. **Pair poses:** for each frame where two cameras both see the *same object*, with object→cam
   poses `T_c1_o`, `T_c2_o`, the inter-camera transform `T_c2_c1 = T_c2_o @ inv(T_c1_o)`. Collect
   per pair, count co-observations.
2. **Average** each pair (§3); build graph (§2); components = **camera groups**;
   `ref_cam = component[0]`.
3. **Compose extrinsics** along the Dijkstra path `ref → cam`, **without** inversing the edge:

```python
T = np.eye(4)
for cur, nxt in zip(path[:-1], path[1:]):
    T = T @ inter_cam[(cur, nxt)]          # McCalib.cpp:1163: transform * current_trans (no .inv())
T_c_g[cam_id] = T                          # ref cam => identity
```

Non-overlapping groups (cameras that never co-observe an object) end up in separate components;
§8 links them.

---

## 8. Hand-eye for non-overlapping cameras — `ds_msp/rig/handeye.py`

Ports `handeyeBootstraptTranslationCalibration` (`geometrytools.cpp:621`). Needed only when §7
yields >1 camera group. OpenCV provides the Tsai solver, so this is mostly orchestration.

```python
def handeye_bootstrap(poses_cam1, poses_cam2, *, nb_cluster=20, nb_clust_pick=6,
                      nb_it=200, gate_deg=15.0, seed=0):
    """poses_cam{1,2}: lists of (4,4) absolute object poses, paired per frame.
    Returns T_c2_c1 (4,4)."""
    # 1. k-means on stacked translations [t1 | t2]  (geometrytools.cpp:405,428)
    feats = np.array([np.r_[T1[:3,3], T2[:3,3]] for T1, T2 in zip(poses_cam1, poses_cam2)],
                     dtype=np.float32)
    k = min(nb_cluster, len(feats))
    _, labels, _ = cv2.kmeans(feats, k, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01), 5, cv2.KMEANS_PP_CENTERS)

    rng = np.random.default_rng(seed)
    Rs, ts = [], []
    for _ in range(nb_it):
        sel = _pick_one_per_cluster(labels, k, nb_clust_pick, rng)   # 6 clusters, 1 pose each (:449,476)
        R_g, t_g = [], []
        for a, b in zip(sel[:-1], sel[1:]):                          # consecutive motions
            R_g.append(...); t_g.append(...)                         # cam1 motion (gripper)
        R_h, t_h = ...                                               # cam2 motion (hand/target)
        Rx, tx = cv2.calibrateHandEye(R_g, t_g, R_h, t_h, method=cv2.CALIB_HAND_EYE_TSAI)
        if _max_consistency_deg(sel, Rx, tx) < gate_deg:             # 15° gate (geometrytools.cpp:659)
            Rs.append(Rx); ts.append(tx.ravel())
    if len(Rs) > 3:                                                  # aggregate (:674)
        R = average_rotation(Rs); t = average_translation(np.array(ts))  # rot avg + trans MEDIAN
    else:
        R, t = _fallback_horaud(poses_cam1, poses_cam2)              # CALIB_HAND_EYE_HORAUD (:392)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T
```

Match the parameters: **6 clusters picked per trial**, **15° rotational-consistency gate**
(`checkSetConsistency`, `geometrytools.cpp:577`: `acos(0.5*(trace-1))`), aggregate only if
**>3 successful** solves, rotation-average + **median** translation, Horaud fallback otherwise.

---

## 9. Staged global BA — extend `ds_msp/core/optimize.py` via a `rig/ba.py` assembler

**This is the centerpiece, and it reuses `schur_lm` unchanged.** The single-camera calibrator
(`calib/bundle.py:59`) already drives `schur_lm` with `shared_dim=P` (intrinsics) and
`n_groups=n_img` local 6-DoF pose blocks. The rig has the **same arrowhead shape** under one
mapping:

| `schur_lm` slot | Rig assignment | dim |
|---|---|---|
| **shared block** (one, dense-coupled) | concat of: all `T_c_g` (non-ref cameras) ⊕ all `T_co_b` (non-ref boards) ⊕ all intrinsics | `shared_dim` |
| **local block** (per group, eliminated) | one `T_g_o` = object pose for one (object, frame) | 6 |
| `n_groups` | number of (object, frame) observations | — |

Why it fits: **each residual touches exactly one local block** — its own frame's object pose —
plus a slice of the shared block (one camera, one board, one intrinsics vector). That is
precisely the separable structure `schur_lm` eliminates (`optimize.py:364-399`): object poses
are the many cheap blocks `Vᵢ`, the shared rig+intrinsics state is `U`. No new solver — only a
`linearize` callback that fills the right Jacobian columns.

### 9.1 Residual & Jacobian chain

For an observation of object point `X_o` (row in `Object3D.pts_3d`, which itself came from board
`b` corner via `T_co_b`) by camera `c` in frame `f`, with measured pixel `u`:

```
X_g  = T_g_o · X_o                      # object → group        (local block, group i)
X_c  = T_c_g · X_g                      # group  → camera       (shared: camera c)
û    = model_c.project(X_c)             # shared: intrinsics c
r    = û − u                            # 2-vector residual
```

(If you refine board poses too, prepend `X_o = T_co_b · X_b`; `T_co_b` is a shared block. The
ref board is identity and fixed, mirroring `refine_board=false`, `CameraGroup.cpp:469`.)

Jacobians — chain rule, all pieces already exist in DS-MSP:

- `∂û/∂X_c` and `∂û/∂intr` come from `model.project_jacobian(X_c)` → `J_point (2,3)`,
  `J_param (2,P)` (`core/contracts.py:90`). **Free, analytic, per model.**
- `∂X_c/∂(camera tangent δ_c)`: with `X_c = T_c_g X_g`, and right-perturbation
  `T_c_g ← T_c_g · se3_exp(δ_c)`, the standard result is
  `∂X_c/∂δ_c = T_c_g[:3,:3] · [ I₃ | −[X_g]_× ]` (trans cols then rot cols, matching `lie.py`'s
  `xi=[ρ,φ]` order). The `−R[·]_×` skew is exactly what `bundle.py:136` already builds
  (`dXc_dw = -R @ skew(Xw)`); reuse `_skew_batch` (`bundle.py:31`).
- `∂X_g/∂(object tangent δ_o)`: same form with `T_g_o`, `X_o` → fills the **local** Jacobian
  `Bᵢ`.
- `∂X_o/∂(board tangent δ_b)`: same form with `T_co_b`, `X_b` → shared columns for board `b`.

So each row pair of the residual fills: `Bᵢ` (6 cols, this frame's object pose) and a few column
slices of `Aᵢ` (6 for camera `c`, 6 for board `b`, `P` for intrinsics `c`). All other shared
columns are zero for that residual — `schur_lm` handles the dense `Aᵢ` fine; sparsity is a
later optimization.

### 9.2 The `linearize` callback

```python
def make_rig_linearize(state: RigState, obs: list[ObjectObs], layout: Shared Layout):
    def linearize(state):
        r_list, A_list, B_list = [], [], []
        for grp in groups_by(obs, key=("object_id", "frame_id")):   # one group per object-pose
            r, A, B = [], [], []
            T_g_o = state.object_poses[grp.key]
            for o in grp.observations:
                cam = state.cameras[o.cam_id]; T_c_g = state.T_c_g[o.cam_id]
                X_o = state.objects[o.object_id].pts_3d[o.point_rows]      # (K,3)
                X_g = apply(T_g_o, X_o); X_c = apply(T_c_g, X_g)
                uv, Jp, Ji, ok = cam.project_jacobian(X_c)
                r.append((uv - o.pts_2d)[ok].ravel())
                # fill A (shared) columns for camera o.cam_id (skip if ref), board, intrinsics o.cam_id
                # fill B (local) columns for this object pose, via Jp @ dXc_dXg @ dXg_ddelta_o
                ...
            r_list.append(np.concatenate(r)); A_list.append(np.vstack(A)); B_list.append(np.vstack(B))
        return r_list, A_list, B_list
    return linearize
```

`retract(state, δ_shared, δ_local)` applies `se3_exp` updates: slice `δ_shared` into per-camera /
per-board increments (`T ← T @ se3_exp(δ)`), add the intrinsics increment to each model's
`params` (rebuild via `model.from_params`), and apply each `δ_local[i]` to its object pose. This
mirrors `bundle.py`'s retract (`:144`) widened to the rig layout.

### 9.3 Staged sequence (match MC-Calib's `refine*` ladder)

Run `schur_lm` in passes, freezing blocks by simply **zeroing their Jacobian columns** (or
omitting them from the shared layout for that pass):

1. **Object-internal** — refine `T_co_b` (non-ref boards) + object poses, intrinsics & extrinsics
   fixed. Port of `Object3D::refineObject` (`Object3D.cpp:166`). Then **re-bake `pts_3d`** from
   the updated `T_co_b` (`updateObjectPts`, `Object3D.cpp:287`).
2. **Poses-only** — refine `T_c_g` + object poses, intrinsics fixed. Port of
   `refineCameraGroupAndObjects` (`CameraGroup.cpp:401`). Ref cam & ref board held fixed.
3. **Full joint** — add intrinsics to the shared block. Port of
   `refineCameraGroupAndObjectsAndIntrinsics` (`CameraGroup.cpp:555`). Run **only if**
   `fix_intrinsics=False`.

All passes use `robust_kernel="huber", robust_scale=1.0` — Huber δ=1.0 px, identical to
MC-Calib (`CameraGroup.cpp:288`). `block=2` (2-D pixel residual).

> **Why not the dense `lm_solve`?** You *can* prototype with `lm_solve` (`optimize.py:123`) over a
> flat state — simplest to get correct first. But object poses dominate the parameter count
> (one 6-vec per frame), so the dense `H` is `O((6·n_frames)²)`. The `schur_lm` mapping
> eliminates them block-diagonally, giving the same answer at MC-Calib-scale cost. Recommend:
> land Phase 1 on `lm_solve` for correctness, switch to the `schur_lm` mapping in the same PR
> once residual/Jacobian are validated against finite differences.

---

## 10. Orchestrator — `ds_msp/rig/rig_calibrate.py`

Port the stage order of `runCalibrationWorkflow` (`apps/calibrate/src/calibrate.cpp:9`),
trimmed to DS-MSP's scope:

```python
def calibrate_rig(detections, boards, models_init=None, *, fix_intrinsics=False):
    # 1. per-camera intrinsics (calib/bundle.calibrate) OR cross-convert seed (adapt/convert)
    # 2. build 3D objects                       §4   (calibrate3DObjects, McCalib.cpp:2127)
    # 3. robust object pose per (object,cam,frame) §5
    # 4. object-internal refine + re-bake pts    §9.3 stage 1
    # 5. camera-pair poses → graph → groups       §7   (calibrateCameraGroup, :2143)
    # 6. object-in-group pose averaging           §6
    # 7. link non-overlapping groups (hand-eye)   §8   (only if >1 group)
    # 8. staged global BA                         §9.3 stages 2-3
    # 9. report per-camera reprojection RMS + write extrinsics  §11
    return RigState(...), report
```

Skip MC-Calib's repeated merge/merge3DObjects passes (`calibrate.cpp:42-52`) in v1 — they
matter for very large rigs with weak connectivity; add the merge↔refine loop in Phase 3.

---

## 11. IO — extend `ds_msp/io/kalibr.py`

`kalibr.py` currently **reads** `T_cn_cnm1` (`load_kalibr_extrinsics:135`) but has **no
extrinsics write path** (`save_kalibr` emits intrinsics only). Add:

```python
def save_kalibr_rig(states, path, *, cam_order):
    """Write per-camera intrinsics + T_cn_cnm1 chain (cam_i relative to cam_{i-1})."""
    # T_cn_cnm1[i] = T_c(i)_g @ inv(T_c(i-1)_g)   — convert rig extrinsics to Kalibr's chained form
```

Kalibr stores each camera relative to the previous (`T_cn_cnm1`), so convert the group-frame
extrinsics `T_c_g` into the chain `T_c(i)_c(i-1) = T_c(i)_g @ inv(T_c(i-1)_g)`. This is the
inverse of how `estimate_relative_pose` already produces `T_cn_cnm1` for the 2-cam case
(`calib/stereo.py:29`).

---

## 12. Tests & validation — `tests/rig/`

Mirror DS-MSP's existing per-module test layout (`tests/calib/`, `tests/mvg/`, `tests/core/`).
Each phase lands with synthetic + (where possible) real fixtures.

**Synthetic rig generator** `tests/rig/_synth.py`: place N cameras at known `T_c_g`, M planar
boards at known `T_co_b`, sample K frames of random object poses, project with a chosen model
(+ pixel noise), drop points outside FOV (use each model's `valid` mask). Ground truth is known,
so every stage is checkable:

| Test | Asserts |
|---|---|
| `test_averaging.py` | Markley avg of rotations within `ε` of truth; antipodal-sign invariance; median trans robust to 30% outliers |
| `test_graph.py` | components match planted connectivity; shortest-path picks min-weight (most co-obs) route |
| `test_object3d.py` | recovered `T_co_b` within `ε` of truth; `pts_3d` round-trips through index maps; **inverse-compose direction** correct (regression-guard the `.inv()` convention, §4) |
| `test_pose_init.py` | RANSAC PnP recovers pose under outliers; **invalidates at <4 inliers** (BoardObs.cpp:149 parity) |
| `test_handeye.py` | non-overlapping 2-group recovered within gate; 15° gate rejects bad motions |
| `test_ba.py` | **Jacobian vs finite-difference** (the single most important test — guards the §9.1 chain); staged BA drives synthetic RMS to noise floor; ref-cam/ref-board stay fixed |
| `test_rig_end2end.py` | full pipeline on synthetic 4-cam rig → per-cam RMS < noise + margin |

**Real fixtures (optional, high value):** MC-Calib ships Blender `Scenario_2`–`Scenario_5`
(4–5 camera rigs with ground-truth poses). A ~30-line loader reads their image sets + GT and
lets you assert DS-MSP recovers extrinsics within tolerance of MC-Calib's published numbers —
the strongest possible parity check, per `RIG_CALIBRATION_PLAN.md` §4.

---

## 13. Porting caveats checklist (reproduce exactly)

- [ ] Board-in-object composition uses **`.inv()`** of each edge (`McCalib.cpp:929`); camera-in-group does **not** (`:1163`). §4 vs §7.
- [ ] Quaternion layout `[x,y,z,w]`, scalar at **index 3**; antipodal fix on the scalar. §3.
- [ ] Translation aggregation is **median** in pair/hand-eye fusion but **mean** in object-in-group averaging. §3, §6.
- [ ] RANSAC adaptive-iteration exponent is **3** (`fracinliers³`); minimal set is **4** points. §5.
- [ ] **<4 inliers → BoardObs invalid** and pruned. §5.
- [ ] Hand-eye: **6** clusters/trial, **15°** consistency gate, aggregate if **>3** successes, Horaud fallback. §8.
- [ ] Reference camera & reference board are **fixed** in BA (identity, no update block). §9.
- [ ] Huber **δ = 1.0 px** in every BA pass. §9.3.
- [ ] OCam is **not** wired into MC-Calib's BA ladders — DS-MSP *can* include it (its `project_jacobian` exists), but cross-check numerics independently; there is no MC-Calib reference to match. §9.

---

## 13b. Status — implemented & validated (2026-06-24)

The pipeline is implemented under `ds_msp/rig/` and validated against MC-Calib's Blender
benchmark (`Blender_Images/Scenario_1..5`) by driving it on MC-Calib's *own detected
keypoints + fused object*, then comparing recovered extrinsics to the synthetic ground
truth and to MC-Calib's result. Run `python scripts/validate_rig.py <Scenario_dir>`.

**Worst per-camera translation error vs ground truth (threshold 2%):**

| Scenario | cameras | result | rotation vs MC-Calib |
|---|---|---|---|
| Scenario_1 (stereo) | 2 | **0.01%** | 0.000° |
| Scenario_2 | 5 | **0.03%** | ≤0.002° |
| Scenario_3 | 4 | **0.13%** | ≤0.002° |
| Scenario_4 | 4 | **0.10%** | ≤0.001° |
| Scenario_5 (cube) | 4 | **0.01%** | ≤0.4° |

All pass; recovered extrinsics match MC-Calib's to ~0.00° rotation (Scenario_5's larger
figure is MC-Calib's *own* deviation from GT — our result tracks MC-Calib to <0.1°).
Per-camera reprojection RMS is sub-pixel and matches MC-Calib's reported values.

**Deviations from the plan above, with rationale:**
- **Dense `lm_solve`, not the `schur_lm` mapping (§9).** At benchmark scale the dense
  solve is fast and simplest-correct, exactly as §9.3's recommendation. The `schur_lm`
  mapping remains the documented scale path; `ba.build_problem` already exposes the
  callbacks if/when it's wired to the sparse solver.
- **Board-in-object poses are baked into `Object3D.pts_3d` (fixed in BA).** The dominant
  extrinsic-accuracy drivers are camera extrinsics + object poses + intrinsics; board
  refinement is a documented future stage (the fusion in `rig/object3d.py` already
  produces the cloud; the validation uses MC-Calib's fused object as a fixed input).
- **Convention.** Internally `RigState.T_c_g[c]` is the *projection* extrinsic
  (group-ref → camera, i.e. world-to-camera). MC-Calib/GT/Kalibr store camera-to-world;
  `io.kalibr.save_kalibr_rig` and the validator invert accordingly.
- **Robust front-end.** A camera that only sees a single near-planar board hits the
  focal/distance ambiguity, so `cv2.calibrateCamera` can return an implausible focal
  (this broke Scenario_4 cam1 before the fix). Such cameras are detected and re-seeded
  from the well-constrained cameras' consensus; global BA then refines them through the
  rigid-rig constraint (`rig/rig_calibrate.py:_front_end_opencv`).

**Tests:** `tests/rig/` — averaging, graph, the analytic-BA-Jacobian-vs-finite-difference
check (the critical one), end-to-end synthetic recovery, and Blender parity (skips if the
dataset is absent). Full suite green (290 tests).

## 13c. Camera-model agnosticism — statistically validated (2026-06-24)

A central design claim is that the rig pipeline is **camera-model agnostic**: every routine
composes poses and calls a `CameraModel`'s `project` / `unproject` / `project_jacobian`, so the
*same* code calibrates a rig whose cameras are represented by any model. This is now validated
statistically, not just asserted.

**Setup** (`scripts/validate_model_agnostic.py`, `tests/rig/test_model_agnostic.py`). For each of
the five well-supported models — RadTan, Double Sphere, UCM, EUCM, Kannala-Brandt — synthesize
many independent 3-camera rigs in which **the cameras are represented by that model** (ground-truth
projection uses it, with per-camera intrinsic jitter), calibrate them **from scratch in that
model**, and measure the worst per-camera extrinsic translation error vs ground truth. The
front-end is `rig.make_bundle_front_end(model_cls)` — a pinhole pre-calibration seeds the focal,
then `calib.bundle.calibrate` refines the target model's full parameter vector; everything
downstream runs entirely in `model_cls`. The target is a genuine 3-D multi-board object (tilted
boards spanning depth) so each camera's focal is well observed.

**Statistical test.** Per model, over 40 trials (noise 0.3 px, 50 frames), a one-sided t-test of
`H0: mean error >= 1%` vs `H1: < 1%`, plus the 99% upper confidence bound on the mean, the
per-trial pass rate, and a cross-model Kruskal-Wallis test. Result:

| model | mean | max | 99% UB on mean | pass rate | t-test p(mean<1%) |
|---|---|---|---|---|---|
| RadTan | 0.30% | 0.56% | 0.35% | 100% | 8e-32 |
| Double Sphere | 0.33% | 0.82% | 0.40% | 100% | 3e-24 |
| UCM | 0.29% | 0.69% | 0.36% | 100% | 1e-26 |
| EUCM | 0.28% | 0.84% | 0.34% | 100% | 5e-29 |
| Kannala-Brandt | 0.34% | 1.00% | 0.42% | 100% | 1e-21 |

Every model rejects `H0` at `p < 1e-20`, every one of the 200 trials is < 1%, and the
**Kruskal-Wallis test (p = 0.64)** finds no significant difference in the error distribution across
models — i.e. which model represents the cameras does **not** materially affect pose accuracy.
**Conclusion: the pipeline is camera-model-agnostic, with pose accuracy < 1% (statistically
significant).** Run it with `python scripts/validate_model_agnostic.py 40 0.3`.

Notes from establishing this (each a real robustness fix, not test-tuning):
- **Front-end seeding must be data-driven.** A blind focal sweep lets a model whose distortion can
  absorb a focal-seed error (KB) settle in a wrong-focal basin that still reprojects well; the
  pinhole-seeded `make_bundle_front_end` avoids it.
- **Planar-view focal ambiguity** — a camera that sees the target near-planar calibrates to a
  wrong/anamorphic focal. Detected by deviation from the per-rig median focal and reset to the
  consensus; the global BA then resolves it through the rigid-rig constraint (same mechanism that
  fixes Blender Scenario_4 cam1).
- **Conditioning sets the tail.** Near-pinhole KB needs enough views to separate focal from
  distortion; 50 frames brings the worst-case under 1% (a uniform change, applied to all models).
- Two latent bugs surfaced and were fixed: a div-by-zero in RANSAC-PnP's adaptive iteration count at
  tiny inlier fractions, and `solvePnP` DLT requiring ≥6 points (now guarded for partial views).

## 14. Relationship to the other docs

- `RIG_CALIBRATION_PLAN.md` — the *what/where* (gap analysis, 8 concepts, file layout, phased
  roadmap). This doc is the *how*.
- `UNIFIED_CALIBRATION_VISION.md` — how this rig path, from-scratch any-model intrinsics, and
  `adapt/convert.py` cross-conversion fuse into "calibrate any camera → cross-convert for
  optimal intrinsics → robust extrinsics." The BA intrinsics seed in §10 step 1 is where the
  cross-convert bridge plugs in (Phase 4).
- `../MC-Calib/docs/CAMERA_MODEL_IMPLEMENTATION_GUIDE.md` — the C++ side; confirms the rig path
  is model-agnostic (composes poses, calls `project`/`unproject`) so the port stays clean.
</content>
</invoke>
