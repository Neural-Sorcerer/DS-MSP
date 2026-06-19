"""
Kalibr camchain YAML I/O.

Reads/writes the standard Kalibr per-camera format so DS-MSP interoperates with
the calibration ecosystem. Field orderings are model-specific and verified
against Kalibr's C++ headers:

  ===========  ==============  ================  ==============================  ====================
  model        camera_model    distortion_model  intrinsics order                distortion_coeffs
  ===========  ==============  ================  ==============================  ====================
  DS           ds              none              [xi, alpha, fx, fy, cx, cy]     []
  EUCM         eucm            none              [alpha, beta, fx, fy, cx, cy]   []
  KB           pinhole         equidistant       [fx, fy, cx, cy]                [k1, k2, k3, k4]
  RadTan       pinhole         radtan            [fx, fy, cx, cy]                [k1, k2, p1, p2]  (no k3)
  UCM          omni            none              [xi_mei, fx, fy, cx, cy]        []
  ===========  ==============  ================  ==============================  ====================

Notes:
- Kalibr's omni mirror parameter ``xi_mei = alpha / (1 - alpha)`` (Mei), not the
  unified ``alpha``; converted on the fly.
- Kalibr radtan has only 4 coeffs: a non-zero ``k3`` is dropped on export (with a
  warning), since the on-disk format cannot represent it.
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import yaml

from ..models.double_sphere import DoubleSphereModel
from ..models.eucm import EUCMModel
from ..models.kb import KannalaBrandtModel
from ..models.radtan import RadTanModel
from ..models.ucm import UCMModel


def to_kalibr_cam(model, width: int, height: int) -> dict:
    """Serialize a model into a Kalibr ``camN`` stanza (dict)."""
    name = model.name
    if name == "ds":
        block = dict(camera_model="ds",
                     intrinsics=[model.xi, model.alpha, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    elif name == "eucm":
        block = dict(camera_model="eucm",
                     intrinsics=[model.alpha, model.beta, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    elif name == "kb":
        block = dict(camera_model="pinhole",
                     intrinsics=[model.fx, model.fy, model.cx, model.cy],
                     distortion_model="equidistant",
                     distortion_coeffs=[model.k1, model.k2, model.k3, model.k4])
    elif name == "radtan":
        if abs(model.k3) > 1e-12:
            warnings.warn("Kalibr radtan has no k3; dropping k3=%.3g on export." % model.k3)
        block = dict(camera_model="pinhole",
                     intrinsics=[model.fx, model.fy, model.cx, model.cy],
                     distortion_model="radtan",
                     distortion_coeffs=[model.k1, model.k2, model.p1, model.p2])
    elif name == "ucm":
        xi_mei = model.alpha / (1.0 - model.alpha)
        block = dict(camera_model="omni",
                     intrinsics=[xi_mei, model.fx, model.fy, model.cx, model.cy],
                     distortion_model="none", distortion_coeffs=[])
    else:
        raise ValueError(f"No Kalibr mapping for model '{name}'")
    block["resolution"] = [int(width), int(height)]
    return block


def from_kalibr_cam(block: dict):
    """Reconstruct a model from a Kalibr ``camN`` stanza (dict)."""
    cm = block["camera_model"]
    dm = block.get("distortion_model", "none")
    I = [float(v) for v in block["intrinsics"]]
    D = [float(v) for v in block.get("distortion_coeffs", [])]

    if cm == "ds":
        xi, alpha, fx, fy, cx, cy = I
        return DoubleSphereModel(fx, fy, cx, cy, xi, alpha)
    if cm == "eucm":
        alpha, beta, fx, fy, cx, cy = I
        return EUCMModel(fx, fy, cx, cy, alpha, beta)
    if cm == "omni":
        xi_mei, fx, fy, cx, cy = I
        if dm not in ("none", None, ""):
            raise NotImplementedError("omni + distortion is not representable by UCMModel")
        alpha = xi_mei / (1.0 + xi_mei)
        return UCMModel(fx, fy, cx, cy, alpha)
    if cm == "pinhole":
        fx, fy, cx, cy = I
        if dm == "equidistant":
            k1, k2, k3, k4 = D
            return KannalaBrandtModel(fx, fy, cx, cy, k1, k2, k3, k4)
        if dm == "radtan":
            k1, k2, p1, p2 = D
            return RadTanModel(fx, fy, cx, cy, k1, k2, p1, p2, 0.0)
        if dm in ("none", None, ""):
            return RadTanModel(fx, fy, cx, cy)  # plain pinhole (zero distortion)
        raise NotImplementedError(f"Unsupported pinhole distortion_model '{dm}'")
    raise NotImplementedError(f"Unsupported camera_model '{cm}'")


def save_kalibr(model, path: str, width: int, height: int, cam: str = "cam0") -> None:
    """Write a single-camera Kalibr camchain YAML file."""
    data = {cam: to_kalibr_cam(model, width, height)}
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


def load_kalibr(path: str, cam: str = "cam0"):
    """Read a model from a Kalibr camchain YAML file.

    Returns the model. Also accessible as ``(model, (width, height))`` via
    :func:`load_kalibr_with_resolution`.
    """
    model, _ = load_kalibr_with_resolution(path, cam)
    return model


def load_kalibr_with_resolution(path: str, cam: str = "cam0") -> Tuple[object, Tuple[int, int]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if cam not in data:
        cam = sorted(k for k in data if k.startswith("cam"))[0]
    block = data[cam]
    model = from_kalibr_cam(block)
    res = block.get("resolution", [0, 0])
    return model, (int(res[0]), int(res[1]))
