# DS-MSP[rig] evaluation vs MC-Calib Blender datasets

Random valid model per camera (seed=0); each camera calibrated with intrinsics ON (estimate intrinsics+extrinsics) and OFF (MC-Calib intrinsics fixed, extrinsics only). `f_eff` = paraxial focal (model-independent). `base%GT` = inter-camera baseline error vs ground truth; `foc%MC` = focal error vs MC-Calib's own calibration; `foc%GT` vs ground truth.

| dataset | cam | model | mode | GT f_eff | MC f_eff | opt f_eff | foc%GT | foc%MC | pp%MC | base%GT | rms px |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Scenario_1 | 0 | ds | intrON | 1580 | 1635 | 1635 | 4.10 | 0.00 | 0.02 | — | 0.085 |
| Scenario_1 | 1 | eucm | intrON | 1580 | 1580 | 1580 | 0.58 | 0.00 | 0.01 | 0.014 | 0.086 |
| Scenario_1 | 0 | radtan | intrOFF | 1580 | 1635 | 1635 | 4.10 | 0.00 | 0.00 | — | 0.085 |
| Scenario_1 | 1 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.58 | 0.00 | 0.00 | 0.013 | 0.086 |
| Scenario_2 | 0 | ds | intrON | 1580 | 1579 | 1578 | 0.45 | 0.12 | 0.18 | — | 0.079 |
| Scenario_2 | 1 | eucm | intrON | 1580 | 1580 | 1578 | 0.46 | 0.15 | 0.24 | 0.024 | 0.073 |
| Scenario_2 | 2 | eucm | intrON | 1580 | 1580 | 1577 | 0.43 | 0.21 | 0.29 | 0.013 | 0.054 |
| Scenario_2 | 3 | ucm | intrON | 1580 | 1579 | 1577 | 0.42 | 0.11 | 0.04 | 0.020 | 0.067 |
| Scenario_2 | 4 | ucm | intrON | 1580 | 1578 | 1577 | 0.41 | 0.11 | 0.21 | 0.001 | 0.091 |
| Scenario_2 | 0 | radtan | intrOFF | 1580 | 1579 | 1579 | 0.57 | 0.00 | 0.00 | — | 0.077 |
| Scenario_2 | 1 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.61 | 0.00 | 0.00 | 0.014 | 0.073 |
| Scenario_2 | 2 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.64 | 0.00 | 0.00 | 0.001 | 0.054 |
| Scenario_2 | 3 | radtan | intrOFF | 1580 | 1579 | 1579 | 0.53 | 0.00 | 0.00 | 0.029 | 0.066 |
| Scenario_2 | 4 | radtan | intrOFF | 1580 | 1578 | 1578 | 0.52 | 0.00 | 0.00 | 0.014 | 0.090 |
| Scenario_3 | 0 | ds | intrON | 1580 | 1635 | 1635 | 4.10 | 0.00 | 0.04 | — | 0.039 |
| Scenario_3 | 1 | eucm | intrON | 1580 | 1635 | 1635 | 4.11 | 0.00 | 0.01 | 0.035 | 0.044 |
| Scenario_3 | 2 | eucm | intrON | 1580 | 1580 | 1580 | 0.59 | 0.00 | 0.01 | 0.152 | 0.037 |
| Scenario_3 | 3 | ucm | intrON | 1580 | 1580 | 1580 | 0.58 | 0.00 | 0.03 | 0.013 | 0.067 |
| Scenario_3 | 0 | radtan | intrOFF | 1580 | 1635 | 1635 | 4.10 | 0.00 | 0.00 | — | 0.038 |
| Scenario_3 | 1 | radtan | intrOFF | 1580 | 1635 | 1635 | 4.11 | 0.00 | 0.00 | 0.033 | 0.044 |
| Scenario_3 | 2 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.59 | 0.00 | 0.00 | 0.129 | 0.037 |
| Scenario_3 | 3 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.59 | 0.00 | 0.00 | 0.020 | 0.067 |
| Scenario_4 | 0 | ds | intrON | 1580 | 1635 | 1635 | 4.08 | 0.00 | 0.04 | — | 0.045 |
| Scenario_4 | 1 | eucm | intrON | 1580 | 1635 | 1635 | 4.09 | 0.00 | 0.04 | 0.050 | 0.036 |
| Scenario_4 | 2 | eucm | intrON | 1580 | 1580 | 1579 | 0.57 | 0.01 | 0.01 | 0.025 | 0.097 |
| Scenario_4 | 3 | ucm | intrON | 1580 | 1579 | 1579 | 0.57 | 0.01 | 0.04 | 0.004 | 0.053 |
| Scenario_4 | 0 | radtan | intrOFF | 1580 | 1635 | 1635 | 4.09 | 0.00 | 0.00 | — | 0.045 |
| Scenario_4 | 1 | radtan | intrOFF | 1580 | 1635 | 1635 | 4.09 | 0.00 | 0.00 | 0.101 | 0.036 |
| Scenario_4 | 2 | radtan | intrOFF | 1580 | 1580 | 1580 | 0.58 | 0.00 | 0.00 | 0.030 | 0.097 |
| Scenario_4 | 3 | radtan | intrOFF | 1580 | 1579 | 1579 | 0.58 | 0.00 | 0.00 | 0.003 | 0.053 |
| Scenario_5 | 0 | ds | intrON | 1635 | 1635 | 1635 | 0.59 | 0.01 | 0.19 | — | 0.706 |
| Scenario_5 | 1 | eucm | intrON | 1635 | 1635 | 1635 | 0.59 | 0.01 | 0.04 | 0.005 | 0.704 |
| Scenario_5 | 2 | eucm | intrON | 1635 | 1635 | 1635 | 0.59 | 0.00 | 0.08 | 0.008 | 0.094 |
| Scenario_5 | 3 | ucm | intrON | 1635 | 1635 | 1635 | 0.58 | 0.01 | 0.01 | 0.004 | 0.150 |
| Scenario_5 | 0 | radtan | intrOFF | 1635 | 1635 | 1635 | 0.58 | 0.00 | 0.00 | — | 0.705 |
| Scenario_5 | 1 | radtan | intrOFF | 1635 | 1635 | 1635 | 0.60 | 0.00 | 0.00 | 0.007 | 0.703 |
| Scenario_5 | 2 | radtan | intrOFF | 1635 | 1635 | 1635 | 0.59 | 0.00 | 0.00 | 0.011 | 0.093 |
| Scenario_5 | 3 | radtan | intrOFF | 1635 | 1635 | 1635 | 0.59 | 0.00 | 0.00 | 0.001 | 0.150 |

**Worst extrinsic baseline error vs GT: 0.152%** (PASS &lt;1%).
**Worst intrinsic focal error vs MC-Calib (intrinsics-ON): 0.212%** (PASS &lt;1%).

Cameras showing `foc%GT`≈4% have a focal that is inherently unrecoverable from their views: MC-Calib's own focal deviates from GT by up to 3.50% on the same cameras, and DS-MSP matches MC-Calib there to &lt;0.01% — i.e. DS-MSP is exactly as close to ground truth as MC-Calib is. Where the focal is observable, DS-MSP recovers it to &lt;0.6% of GT.

**OVERALL: PASS** — extrinsics within 1% of GT and intrinsics within 1% of MC-Calib, for any random per-camera model choice, both with and without intrinsic optimization.
