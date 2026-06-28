# Licensing

DS-MSP is **dual-licensed**. Most of the library is permissively MIT-licensed; the two
flagship camera-model implementations (DS+ and EUCM+) are released under a
**noncommercial** license that requires attribution.

## At a glance

| Part | License | Commercial use? |
|------|---------|-----------------|
| The library (everything below except the rows that follow) | **MIT** (see [`LICENSE`](LICENSE)) | ✅ allowed |
| `ds_msp/models/dsplus.py`, `ds_msp/models/dsplus_math.py` (**DS+**) | **PolyForm Noncommercial 1.0.0** (see [`LICENSE-NONCOMMERCIAL.txt`](LICENSE-NONCOMMERCIAL.txt)) | ❌ noncommercial only, with attribution |
| `ds_msp/models/eucmplus.py`, `ds_msp/models/eucmplus_math.py` (**EUCM+**) | **PolyForm Noncommercial 1.0.0** | ❌ noncommercial only, with attribution |

SPDX expression for the distribution: `MIT AND LicenseRef-PolyForm-Noncommercial-1.0.0`.

## What this means

- **Research / academic / personal / nonprofit / government use:** free, including DS+/EUCM+,
  as long as you keep the required notice (see below). Modification and redistribution for
  noncommercial purposes are permitted under PolyForm Noncommercial 1.0.0.
- **Commercial use of DS+ or EUCM+** (in a product, a paid service, or any anticipated
  commercial application) is **not** granted by the noncommercial license and requires a
  separate commercial license from the author.
- **The rest of the library is MIT** — you may use it commercially without restriction. If you
  need a commercial wide-FOV model, the MIT-licensed KB / Double Sphere / UCM / EUCM / RadTan
  models are unrestricted; only the *Plus* variants are noncommercial.

## Required attribution

Per the PolyForm "Notices" clause, any copy of DS+/EUCM+ (or works based on them) must carry:

> Required Notice: Copyright (c) 2025-2026 Munna-Manoj.
> DS+ (Double Sphere Plus) and EUCM+ (Extended UCM Plus) camera models,
> from DS-MSP (https://github.com/Munna-Manoj/DS-MSP), by Munna-Manoj.
> Any use must credit the author "Munna-Manoj" and this repository.

For academic work, please also cite the project — see [`CITATION.cff`](CITATION.cff).

## Commercial licensing

To use DS+ or EUCM+ commercially, contact the author **Munna-Manoj** via
<https://github.com/Munna-Manoj> to arrange a commercial license.
