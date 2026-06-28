"""Model-name <-> class registry, with MC-Calib name aliases.

MC-Calib selects a camera's model by a string in its config (``camera_models: [radtan,
double_sphere, ...]`` or the legacy ``distortion_per_camera`` ints 0=radtan / 1=kb) and
writes that string back as ``camera_model`` in ``calibrated_cameras_data.yml``. DS-MSP's
own model ``.name`` is ``"ds"`` where MC-Calib uses ``"double_sphere"``; everything else
matches. This module is the single place that bridges the two so the rig front-end and the
MC-Calib reader/writer stay byte-compatible.
"""

from __future__ import annotations

from typing import Dict, Type

from ..core.contracts import CameraModel
from .double_sphere import DoubleSphereModel
from .dsplus import DSPlusModel
from .eucm import EUCMModel
from .eucmplus import EUCMPlusModel
from .kb import KannalaBrandtModel
from .ocam import OCamModel
from .radtan import RadTanModel
from .ucm import UCMModel

#: Canonical DS-MSP name -> class.
_BY_NAME: Dict[str, Type[CameraModel]] = {
    "radtan": RadTanModel,
    "ds": DoubleSphereModel,
    "ucm": UCMModel,
    "eucm": EUCMModel,
    "kb": KannalaBrandtModel,
    "ocam": OCamModel,
    "dsplus": DSPlusModel,
    "eucmplus": EUCMPlusModel,
}

#: Accepted aliases (MC-Calib strings + legacy ints) -> canonical DS-MSP name.
_ALIAS: Dict[str, str] = {
    "double_sphere": "ds", "doublesphere": "ds",
    "kannala": "kb", "kannala_brandt": "kb", "fisheye": "kb",
    "brown": "radtan", "perspective": "radtan", "opencv": "radtan",
    "ocam_calib": "ocam", "ocamcalib": "ocam", "scaramuzza": "ocam",
    "ds+": "dsplus", "ds_plus": "dsplus", "doublespheres_plus": "dsplus",
    "eucm+": "eucmplus", "eucm_plus": "eucmplus",
    "0": "radtan", "1": "kb",                      # legacy distortion_model ints
}

#: DS-MSP name -> the string MC-Calib writes in calibrated_cameras_data.yml.
_TO_MCCALIB: Dict[str, str] = {
    "radtan": "radtan", "ds": "double_sphere", "ucm": "ucm",
    "eucm": "eucm", "kb": "kb", "ocam": "ocam", "dsplus": "dsplus",
    "eucmplus": "eucmplus",
}

#: Models valid for a (forward-facing) pinhole camera, and for a fisheye camera, used by the
#: random model-of-choice assignment. KB is fisheye-only; RadTan/Brown is pinhole-only; UCM /
#: EUCM / Double-Sphere handle both. OCam is intentionally *not* in these auto pools: it is
#: fully selectable by name (``model_class("ocam")``) and writes/reads in MC-Calib format, but
#: its from-scratch front-end initialization (a forward projection polynomial seeded from
#: pinhole bearings) is not yet robust on every real camera, so it is opt-in rather than part
#: of the validated random sweep.
PINHOLE_MODELS = ("radtan", "ucm", "eucm", "ds")
FISHEYE_MODELS = ("kb", "ucm", "eucm", "ds")
#: Full validity sets including OCam (for explicit selection / capability listing).
PINHOLE_MODELS_ALL = ("radtan", "ucm", "eucm", "ds", "ocam")
FISHEYE_MODELS_ALL = ("kb", "ucm", "eucm", "ds", "ocam")


def canonical_name(name) -> str:
    """Normalize any accepted model spec (DS-MSP name, MC-Calib name, or legacy int)."""
    s = str(name).strip().lower()
    if s in _BY_NAME:
        return s
    if s in _ALIAS:
        return _ALIAS[s]
    raise KeyError(f"unknown camera model {name!r}; "
                   f"known: {sorted(_BY_NAME)} (+ aliases {sorted(_ALIAS)})")


def model_class(name) -> Type[CameraModel]:
    """Resolve a model spec to its DS-MSP class."""
    return _BY_NAME[canonical_name(name)]


def mccalib_name(name) -> str:
    """The string MC-Calib uses for this model (``ds`` -> ``double_sphere``)."""
    return _TO_MCCALIB[canonical_name(name)]
