# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.3.0] — 2026-06-20

First public, CI-tested, PyPI-ready release.

### Added
- **Calibration from real images.** `ds_msp.calib.detect_aprilgrid` (AprilGrid detection
  adapter, optional `[calib]` extra) + `AprilGridTarget` (board geometry); a robust loss
  (`loss=` / `f_scale=`) for `ds_msp.calib.calibrate`.
- **Learning curriculum** (`docs/learn/`) with five runnable examples on real TUM-VI data,
  including the calibration capstone (detect → bundle-adjust → match the published reference).
- **Continuous integration** — GitHub Actions running `ruff` + `import-linter` + `mypy` +
  `pytest` on Python 3.10–3.12, plus README badges.
- **Benchmarks** — `benchmarks/benchmark.py`: accuracy vs OpenCV (~1e-13 px) and the
  analytic-vs-finite-difference Jacobian speedup.
- **Dataset guide** (`datasets/README.md`) mapping each roadmap tier to its data.
- **Packaging metadata** — license (MIT), classifiers, project URLs, keywords; this CHANGELOG.

### Changed
- Minimum Python is now **3.10** (the NumPy/SciPy stack requires it).
- README refactored into a structured, guided page.

### Fixed
- `ds_msp.model` now correctly re-exports `ds_project_jacobian` (was referenced by
  `calibrate.py` and the README but missing from the re-export list).
- `import-linter` contract for service-layer independence (`ops`/`adapt`/`calib`).
- §7.2 README math now renders correctly on GitHub.

## [0.2.0]

- Multi-model camera library (UCM, EUCM, Kannala-Brandt, RadTan, OCamCalib) behind one
  `CameraModel` contract, with model conversion and Kalibr YAML I/O.
- `pip install -e .` packaging fix (setuptools flat-layout discovery).
- Double Sphere core with the correct `> 180°` half-space projection validity, analytic
  Jacobians, OpenCV-compatible API, and TI Jacinto LDC export.

[0.3.0]: https://github.com/Munna-Manoj/DS-MSP/releases/tag/v0.3.0
