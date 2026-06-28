"""Neutral data layer: observation/correspondence containers + the dataset abstraction.

Depends only on ``core`` and NumPy. Consumed by every calibration service (``calib``,
``rig``) and the IO/detection adapters, so shared record types never force one service to
import another.
"""

from .dataset import CalibDataset
from .observations import (
    BoardObs,
    Object3D,
    ObjectObs,
    Observation,
    RigState,
)

__all__ = [
    "Observation",
    "CalibDataset",
    "BoardObs",
    "Object3D",
    "ObjectObs",
    "RigState",
]
