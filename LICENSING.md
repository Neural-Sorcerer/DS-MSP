# Licensing

DS-MSP is **dual-licensed**. The generic library (standard camera models, geometry,
detection, I/O, evaluation) is permissively MIT-licensed; the project's core
contribution — the **DS+ / EUCM+ models** and the **robust from-scratch
calibration/conversion engine** — is released under a **noncommercial** license that
requires attribution.

## At a glance

| Part | License | Commercial use? |
|------|---------|-----------------|
| The generic library (everything except the rows below) | **MIT** (see [`LICENSE`](LICENSE)) | ✅ allowed |
| `ds_msp/models/dsplus.py`, `ds_msp/models/dsplus_math.py` (**DS+**) | **PolyForm Noncommercial 1.0.0** (see [`LICENSE-NONCOMMERCIAL.txt`](LICENSE-NONCOMMERCIAL.txt)) | ❌ noncommercial only, with attribution |
| `ds_msp/models/eucmplus.py`, `ds_msp/models/eucmplus_math.py` (**EUCM+**) | **PolyForm Noncommercial 1.0.0** | ❌ noncommercial only, with attribution |
| `ds_msp/geometry/resection.py` (robust intrinsic seed + RANSAC resection) | **PolyForm Noncommercial 1.0.0** | ❌ noncommercial only, with attribution |
| `ds_msp/calib/bundle.py` (robust auto-initialized `calibrate()`) | **PolyForm Noncommercial 1.0.0** | ❌ noncommercial only, with attribution |
| `ds_msp/adapt/convert.py` (robust image-free model conversion) | **PolyForm Noncommercial 1.0.0** | ❌ noncommercial only, with attribution |

SPDX expression for the distribution: `MIT AND LicenseRef-PolyForm-Noncommercial-1.0.0`.

> **Practical effect.** The robust engine sits under `calibrate()`/`convert()`, so the
> *maintained library taken as a whole* is noncommercial-to-use even though most individual
> files remain MIT-labeled. The MIT pieces (standard KB/DS/UCM/EUCM/RadTan models, geometry
> helpers, I/O, detection) stay independently usable for any purpose, including commercially.

## What this means

- **Research / academic / personal / nonprofit / government use:** free, including the
  noncommercial files, as long as you keep the required notice (see below). Modification and
  redistribution for noncommercial purposes are permitted under PolyForm Noncommercial 1.0.0.
- **Commercial use of the noncommercial files** — the DS+/EUCM+ models or the robust
  calibration/conversion engine (`resection.py`, `bundle.py`, `convert.py`), in a product, a
  paid service, or any anticipated commercial application — is **not** granted by the
  noncommercial license and requires a separate commercial license from the author.
- **The generic library is MIT** — you may use it commercially without restriction. The
  MIT-licensed KB / Double Sphere / UCM / EUCM / RadTan models, geometry helpers, I/O and
  detection are unrestricted; only the *Plus* models and the *robust* calibrate/convert engine
  are noncommercial. (Building your own initializer/optimizer on top of the MIT models is fine.)

## Required attribution

Per the PolyForm "Notices" clause, any copy of the noncommercial files (or works based on them)
must carry:

> Required Notice: Copyright (c) 2025-2026 Munna-Manoj.
> DS+ / EUCM+ camera models and the robust calibration/conversion engine,
> from DS-MSP (https://github.com/Munna-Manoj/DS-MSP), by Munna-Manoj.
> Any use must credit the author "Munna-Manoj" and this repository.

For academic work, please also cite the project — see [`CITATION.cff`](CITATION.cff).

## Commercial licensing

To use the noncommercial files (DS+/EUCM+ or the robust calibrate/convert engine) commercially,
contact the author **Munna-Manoj** via <https://github.com/Munna-Manoj> to arrange a commercial
license.
