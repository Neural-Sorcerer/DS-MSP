# Learn: fisheye & omnidirectional camera geometry, from first principles

A guided, **runnable** path through wide-FOV camera models — the geometry behind
SLAM, AR, and robot perception. Every chapter pairs a short explainer with a script
that runs on **real public data** and prints a **number you can verify**. That last
part is the whole point: in 3D vision you don't *hope* your math is right, you
*measure* that it is (a good unprojection inverts projection to ~1e-14 px, not "looks
about right").

This is the teaching layer. The library it teaches (`ds_msp/`) stays deliberately
clean and untouched by tutorial clutter — read the docs to learn, read the code to
see how it's done in production.

## Who this is for
Aspiring 3D-vision researchers, applied-perception engineers, and developers who know
some Python + linear algebra and want to *actually understand* (not just call) camera
models. No prior fisheye knowledge assumed.

## Setup (once)
```bash
# 1. Environment (uv recommended; venv/conda fine too)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .

# 2. Data — small, free, ~3 GB for the fisheye track
bash scripts/download_datasets.sh tumvi
```
See [`datasets/README.md`](../../datasets/README.md) for what each dataset contains.

## The path

| # | Chapter | You'll be able to… | Code anchor |
|---|---------|--------------------|-------------|
| 1 | [Fisheye & camera models](01_fisheye_and_camera_models.md) | load a real calibration, prove project/unproject are inverses, rectify a fisheye frame | `examples/01_realdata_fisheye_tumvi.py` |
| 2 | The Double Sphere model *(coming soon)* | derive DS projection and read it in code | `ds_msp/models/ds_math.py` |
| 3 | Projection validity & the >180° cone *(coming soon)* | explain why `z>0` is the classic bug | `ds_msp/models/ds_math.py` |
| 4 | Analytic Jacobians vs autodiff *(coming soon)* | derive a Jacobian and gradient-check it | `ds_msp/model.py` |
| 5 | Calibration by Levenberg–Marquardt *(coming soon)* | calibrate from corner detections | `calibrate.py`, `ds_msp/calib/` |
| 6 | One model to another: conversion *(coming soon)* | turn a DS calib into KB/EUCM without re-shooting | `ds_msp/adapt/` |
| 7 | Reproducing a published calibration *(coming soon)* | match TUM-VI / EuRoC reference numbers with your own code | `ds_msp/io/kalibr.py` |

Chapters land incrementally — see [`../ROADMAP.md`](../ROADMAP.md) for the build order.

## How to use it
Read the chapter, run its script, then **change one thing and predict what happens**
before you re-run (a different `balance`, a wider pixel grid, `cam1` instead of `cam0`).
The fastest way to learn geometry is to break it on purpose and watch the number move.
