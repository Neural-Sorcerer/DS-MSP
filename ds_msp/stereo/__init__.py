"""Stereo depth on wide-FOV cameras (Tier-1). Sphere-sweep runs straight on calibrated
fisheye views — no rectification — using only the camera's project/unproject."""

from .sphere_sweep import inverse_depth_samples, sphere_sweep, sweep_to_points

__all__ = ["sphere_sweep", "inverse_depth_samples", "sweep_to_points"]
