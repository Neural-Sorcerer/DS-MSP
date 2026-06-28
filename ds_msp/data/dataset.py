"""``CalibDataset`` — the single indexable collection calibration consumes.

The torch-``Dataset`` analogue: build one from any detector or IO source, then hand it to
single-camera calibration (via :meth:`CalibDataset.as_parallel_lists`) or group it for
multi-camera rig calibration (via :meth:`CalibDataset.by_camera` / :meth:`by_frame`).

Keeping this here means ``calib``'s historical ``(X_world_list, keypoints_list,
visibility_list)`` signature becomes a *view* over a dataset rather than a separate data
model, and ``rig`` consumes the same records — one abstraction, no per-service forks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .observations import Observation


@dataclass
class CalibDataset:
    """An ordered collection of :class:`Observation` records."""

    observations: List[Observation]

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, i: int) -> Observation:
        return self.observations[i]

    def __iter__(self):
        return iter(self.observations)

    def by_camera(self) -> Dict[int, List[Observation]]:
        """Group observations by ``cam_id`` (insertion order preserved per camera)."""
        out: Dict[int, List[Observation]] = {}
        for o in self.observations:
            out.setdefault(o.cam_id, []).append(o)
        return out

    def by_frame(self) -> Dict[int, List[Observation]]:
        """Group observations by ``frame_id``."""
        out: Dict[int, List[Observation]] = {}
        for o in self.observations:
            out.setdefault(o.frame_id, []).append(o)
        return out

    def as_parallel_lists(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Return ``(X_world_list, keypoints_list, visibility_list)`` — the legacy
        single-camera view that :func:`ds_msp.calib.calibrate` consumes."""
        X = [o.points_3d for o in self.observations]
        kp = [o.pixels for o in self.observations]
        vis = [o.visibility for o in self.observations]
        return X, kp, vis

    @classmethod
    def from_parallel_lists(cls,
                            X_world_list: Sequence[np.ndarray],
                            keypoints_list: Sequence[np.ndarray],
                            visibility_list: Sequence[np.ndarray],
                            *, cam_id: int = 0) -> "CalibDataset":
        """Build a dataset from the legacy parallel lists (one frame per element)."""
        obs = [
            Observation(points_3d=X, pixels=kp, visibility=vis,
                        cam_id=cam_id, frame_id=i)
            for i, (X, kp, vis) in enumerate(zip(X_world_list, keypoints_list, visibility_list))
        ]
        return cls(obs)
