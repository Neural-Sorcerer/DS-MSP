"""Generic calibration (bundle adjustment) for any CameraModel.

``calibrate`` and ``AprilGridTarget`` are dependency-light. ``detect_aprilgrid``
needs the optional ``aprilgrid`` backend (``pip install ds_msp[calib]``) and is
imported lazily so this package stays importable without it.
"""

from .bundle import calibrate
from .targets import AprilGridTarget

__all__ = ["calibrate", "AprilGridTarget", "detect_aprilgrid"]


def __getattr__(name: str):
    # Lazy: only pull in the OpenCV+aprilgrid detection adapter on demand.
    if name == "detect_aprilgrid":
        from .detect import detect_aprilgrid
        return detect_aprilgrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
