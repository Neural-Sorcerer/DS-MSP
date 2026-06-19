"""
Sampling strategies for model conversion.

Pure numpy. Produces the pixel/ray correspondences the converter fits against.
"""

from __future__ import annotations

import numpy as np


def sample_image_grid(width: int, height: int, n_samples: int = 500) -> np.ndarray:
    """Regular grid of pixel centers, aspect-ratio preserving (FCA-style).

    Returns ``(M, 2)`` float64 pixels with ``M ≈ n_samples``.
    """
    nx = max(2, int(round(np.sqrt(n_samples * width / height))))
    ny = max(2, int(round(np.sqrt(n_samples * height / width))))
    cw = width / nx
    ch = height / ny
    xs = (np.arange(nx) + 0.5) * cw
    ys = (np.arange(ny) + 0.5) * ch
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float64)
