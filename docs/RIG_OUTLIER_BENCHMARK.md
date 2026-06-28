# Outlier-handling benchmark — better weighting, no rejection

Per-view PnP on a synthetic camera, median over 60 views, a controlled fraction of corners corrupted with a 50 px gross shift. The per-view front-end is the whole story here (no downstream BA to mask it).

* **L2** — `solvePnP` over all corners, no robustness (naive baseline).
* **reject** — RANSAC P3P + inlier refine (hard rejection, MC-Calib-style).
* **reweight** — `robust_pose_irls`: RANSAC warm-start + redescending Cauchy IRLS (MAD auto-scale, GNC, studentized leverage) over every corner — no rejection.

| outlier % | L2 dR° | L2 dt | reject dR° | reject dt | reweight dR° | reweight dt | dR° reduction vs L2 |
|---|---|---|---|---|---|---|---|
| 0% | 0.045 | 0.0015 | 0.045 | 0.0015 | 0.046 | 0.0015 | -3.5% |
| 5% | 1.054 | 0.0359 | 0.046 | 0.0017 | 0.050 | 0.0017 | 95.2% |
| 10% | 164.051 | 3.9544 | 0.050 | 0.0018 | 0.051 | 0.0019 | 100.0% |
| 20% | 177.290 | 4.1124 | 0.048 | 0.0017 | 0.049 | 0.0017 | 100.0% |
| 30% | 177.837 | 4.1943 | 0.060 | 0.0019 | 0.059 | 0.0019 | 100.0% |
| 40% | 177.702 | 4.2501 | 0.059 | 0.0021 | 0.070 | 0.0023 | 100.0% |

**Mean rotation-error reduction vs naive L2 at ≥10 % outliers: 100.0%** (PASS &gt;50%) — by *weighting*, every corner is kept.

## Studentized leverage — the self-masking outlier a residual kernel cannot see

One far-off-axis corner with a modest mis-decode: high leverage lets it pull the pose while keeping its own residual small, so a residual-only kernel never down-weights it. Studentizing the residual recovers it: median rotation error **0.820° → 0.165° (79.9% lower)**.
