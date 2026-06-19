"""
Double Sphere Camera Model - Clean Implementation
==================================================

A minimal, production-ready implementation of the Double Sphere camera model
for fisheye cameras. Easy to understand, test, and integrate into other projects.

Author: Advanced 3D Vision
License: MIT
"""

import numpy as np
import cv2
import json
from typing import Tuple, Optional


class DoubleSphereCamera:
    """
    Double Sphere Camera Model (Usenko et al., 2018)
    
    A camera model for wide-angle and fisheye lenses with closed-form
    unprojection, making it ideal for real-time applications and PnP.
    
    Parameters
    ----------
    fx, fy : float
        Focal lengths in pixels
    cx, cy : float
        Principal point coordinates
    xi, alpha : float
        Double Sphere distortion parameters
    width, height : int
        Image dimensions in pixels
    
    Examples
    --------
    >>> # Create camera from calibration
    >>> cam = DoubleSphereCamera(
    ...     fx=711.57, fy=711.24, cx=949.18, cy=518.81,
    ...     xi=0.183, alpha=0.809, width=1920, height=1080
    ... )
    >>> 
    >>> # Undistort image
    >>> img_undist, K_new = cam.undistort_image(img)
    >>> 
    >>> # Solve PnP
    >>> success, rvec, tvec = cam.solve_pnp(points_3d, points_2d)
    """
    
    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float,
                 width: Optional[int] = None, height: Optional[int] = None,
                 is_flip: bool = False):
        # The projection model needs only the 6 intrinsics. `width`/`height` are
        # used solely by the image-level helpers (undistortion maps, K_new); they
        # are optional so the model can be built for pure project/unproject/PnP
        # without inventing meaningless image dimensions.
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.alpha = alpha
        self.width = width
        self.height = height
        self.is_flip = is_flip

        # Cache for undistortion
        self._mapx = None
        self._mapy = None
        self._K_new = None

    def _require_dims(self, what: str) -> None:
        if self.width is None or self.height is None:
            raise ValueError(
                f"{what} requires image dimensions; construct with "
                f"width=... and height=... (only needed for image-level ops)."
            )

    @property
    def K(self) -> np.ndarray:
        """Pinhole intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        """Double Sphere distortion coefficients [xi, alpha]."""
        return np.array([self.xi, self.alpha], dtype=np.float64)

    def __repr__(self) -> str:
        dims = f", width={self.width}, height={self.height}" if self.width else ""
        return (f"DoubleSphereCamera(fx={self.fx:.3f}, fy={self.fy:.3f}, "
                f"cx={self.cx:.3f}, cy={self.cy:.3f}, xi={self.xi:.4f}, "
                f"alpha={self.alpha:.4f}{dims})")
    
    @classmethod
    def from_json(cls, json_path: str):
        """Load camera from calibration JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 1. Top-level calibration output format
        if 'fx' in data and 'fy' in data and 'cx' in data and 'cy' in data:
            fx = data['fx']
            fy = data['fy']
            cx = data['cx']
            cy = data['cy']
            xi = data.get('xi', 0.0)
            alpha = data.get('alpha', 0.5)
            width = data.get('image_width', data.get('width', 1920))
            height = data.get('image_height', data.get('height', 1080))
            is_flip = data.get('is_flip', False)
            return cls(fx, fy, cx, cy, xi, alpha, width, height, is_flip)
            
        # 2. Nested intrinsics format
        elif 'intrinsics' in data:
            intrinsic = data['intrinsics']
            width = data.get('image_width', 640)
            height = data.get('image_height', 480)
            return cls(
                fx=intrinsic['fx'], fy=intrinsic['fy'],
                cx=intrinsic['cx'], cy=intrinsic['cy'],
                xi=intrinsic['xi'], alpha=intrinsic['alpha'],
                width=width, height=height
            )
            
        # 3. Third-party nested resolution format
        else:
            try:
                cam_data = list(data.values())[0]
                intrinsic = cam_data['intrinsics'][0]['intrinsics']
                resolution = cam_data['resolution'][0]
                width, height = resolution[0], resolution[1]
                return cls(
                    fx=intrinsic['fx'], fy=intrinsic['fy'],
                    cx=intrinsic['cx'], cy=intrinsic['cy'],
                    xi=intrinsic['xi'], alpha=intrinsic['alpha'],
                    width=width, height=height
                )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(f"Unsupported calibration JSON format: {e}")
    
    # ========================================================================
    # Core Projection/Unprojection
    # ========================================================================

    def project(self, points_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D pixel coordinates.
        """
        u, v, valid = ds_project(points_3d, self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)
        
        # Flip x-coordinates if driver provides flipped images
        if self.is_flip:
            u = (self.width - 1) - u
        
        return np.stack([u, v], axis=-1), valid
    
    def unproject(self, points_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unproject 2D pixels to 3D unit rays (closed-form).
        """
        u, v = points_2d[..., 0], points_2d[..., 1]
        
        # Flip x-coordinates if driver provides flipped images
        if self.is_flip:
            u = (self.width - 1) - u
            
        return ds_unproject(np.stack([u, v], axis=-1), self.fx, self.fy, self.cx, self.cy, self.xi, self.alpha)



    
    # ========================================================================
    # Image Undistortion
    # ========================================================================
    
    def compute_K_new(self, balance: float = 0.5) -> np.ndarray:
        """
        Compute optimal K matrix for undistorted image.
        
        Parameters
        ----------
        balance : float
            0.0 = more FOV (40% of original focal length)
            0.5 = balanced (60% of original) - default
            1.0 = less FOV (80% of original)
            
        Returns
        -------
        K_new : (3, 3) array
            New intrinsic matrix
        """
        self._require_dims("compute_K_new")
        return balanced_pinhole_K(self.fx, self.fy, self.width, self.height, balance)
    
    def get_undistortion_maps(self, K_new: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate undistortion maps for cv2.remap.
        
        Parameters
        ----------
        K_new : (3, 3) array, optional
            New intrinsic matrix. If None, computed automatically.
            
        Returns
        -------
        mapx, mapy : (H, W) arrays
            Maps for cv2.remap
        K_new : (3, 3) array
            Intrinsic matrix used
        """
        self._require_dims("get_undistortion_maps")
        if K_new is None:
            K_new = self.compute_K_new()

        if self._mapx is not None and self._K_new is not None:
            if np.array_equal(K_new, self._K_new):
                return self._mapx, self._mapy, self._K_new
        
        fx_new, fy_new = K_new[0, 0], K_new[1, 1]
        cx_new, cy_new = K_new[0, 2], K_new[1, 2]
        
        # Create undistorted pixel grid
        x = np.arange(self.width, dtype=np.float32)
        y = np.arange(self.height, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x, y, indexing='xy')
        
        # Convert to normalized coordinates and create rays
        mx = (x_grid - cx_new) / fx_new
        my = (y_grid - cy_new) / fy_new
        rays = np.stack([mx, my, np.ones_like(mx)], axis=-1)
        
        # Project back to distorted image
        distorted_pts, valid = self.project(rays)
        
        mapx = distorted_pts[..., 0].astype(np.float32)
        mapy = distorted_pts[..., 1].astype(np.float32)
        mapx[~valid] = -1
        mapy[~valid] = -1
        
        self._mapx, self._mapy, self._K_new = mapx, mapy, K_new
        
        return mapx, mapy, K_new
    
    def undistort_image(self, img: np.ndarray, K_new: Optional[np.ndarray] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort fisheye image to pinhole.
        
        Parameters
        ----------
        img : (H, W, C) or (H, W) array
            Input distorted image
        K_new : (3, 3) array, optional
            New intrinsic matrix
            
        Returns
        -------
        img_undist : array
            Undistorted image
        K_new : (3, 3) array
            Intrinsic matrix for undistorted image
        """
        mapx, mapy, K_new = self.get_undistortion_maps(K_new)
        img_undist = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return img_undist, K_new
    
    # ========================================================================
    # Keypoint Transformation
    # ========================================================================
    
    def undistort_points(self, points_dist: np.ndarray, K_new: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform keypoints from distorted to undistorted space."""
        rays, valid = self.unproject(points_dist)
        rays_norm = rays / (rays[:, 2:3] + 1e-10)
        
        u = K_new[0, 0] * rays_norm[:, 0] + K_new[0, 2]
        v = K_new[1, 1] * rays_norm[:, 1] + K_new[1, 2]
        
        return np.column_stack([u, v]), valid
    
    def distort_points(self, points_undist: np.ndarray, K_new: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """Transform keypoints from undistorted to distorted space."""
        mx = (points_undist[:, 0] - K_new[0, 2]) / K_new[0, 0]
        my = (points_undist[:, 1] - K_new[1, 2]) / K_new[1, 1]
        rays = np.column_stack([mx, my, np.ones(len(mx))])
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        
        return self.project(rays)
    
    # ========================================================================
    # PnP Pose Estimation
    # ========================================================================
    
    def solve_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray,
                  method: int = cv2.SOLVEPNP_ITERATIVE
                 ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Solve PnP for fisheye camera.
        
        This method handles fisheye distortion by unprojecting to rays
        and solving PnP in normalized space.
        
        Parameters
        ----------
        points_3d : (N, 3) array
            3D points in world coordinates
        points_2d : (N, 2) array
            2D keypoints in distorted image
        method : int
            OpenCV PnP method (e.g., cv2.SOLVEPNP_ITERATIVE)
            
        Returns
        -------
        success : bool
        rvec : (3,) array or None
            Rotation vector
        tvec : (3,) array or None
            Translation vector
        """
        rays, valid = self.unproject(points_2d)

        # PnP runs in the pinhole-normalized plane (x/z, y/z), which is only
        # defined for rays in front of the camera (z > 0). Rays at or beyond
        # 90 deg would project to sign-flipped / unbounded coordinates and
        # corrupt the solve, so keep only the front-facing, valid rays.
        usable = valid & (rays[:, 2] > 1e-6)
        if not usable.all():
            points_3d = points_3d[usable]
            rays = rays[usable]
            if len(points_3d) < 4:
                return False, None, None

        rays_norm = rays / rays[:, 2:3]
        points_2d_norm = rays_norm[:, :2]
        
        success, rvec, tvec = cv2.solvePnP(
            points_3d.astype(np.float64),
            points_2d_norm.astype(np.float64),
            np.eye(3, dtype=np.float64),
            np.zeros(5, dtype=np.float64),
            flags=method
        )
        
        if success:
            rvec = rvec.squeeze()
            tvec = tvec.squeeze()
        
        return success, rvec, tvec
    
    # ========================================================================
    # Visualization
    # ========================================================================
    
    def draw_axes(self, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray,
                  axis_length: float = 0.1, K: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """
        Draw 3D coordinate axes on image.
        
        Parameters
        ----------
        img : array
            Image to draw on
        rvec, tvec : (3,) arrays
            Pose (rotation and translation vectors)
        axis_length : float
            Length of axes in meters
        K : (3, 3) array, optional
            If None, draws on distorted image. Otherwise, draws on undistorted.
            
        Returns
        -------
        img_out : array
            Image with drawn axes
        """
        img_out = img.copy()
        
        # Define axes in 3D
        axes_3d = np.array([
            [0, 0, 0],
            [axis_length, 0, 0],  # X: Red
            [0, axis_length, 0],  # Y: Green
            [0, 0, axis_length]   # Z: Blue
        ])
        
        # Transform to camera coordinates
        R, _ = cv2.Rodrigues(rvec)
        axes_cam = (R @ axes_3d.T).T + tvec
        
        # Project
        if K is None:
            # Distorted image
            axes_2d, valid = self.project(axes_cam)
        else:
            # Undistorted image
            axes_2d_hom = (K @ axes_cam.T).T
            axes_2d = axes_2d_hom[:, :2] / axes_2d_hom[:, 2:3]
            valid = axes_cam[:, 2] > 0
        
        if not valid.all():
            return img_out
        
        axes_2d_int = axes_2d.astype(np.int32)
        origin = tuple(axes_2d_int[0])
        
        # Draw axes
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[1]), (0, 0, 255), 3, tipLength=0.3)  # X: Red
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[2]), (0, 255, 0), 3, tipLength=0.3)  # Y: Green
        cv2.arrowedLine(img_out, origin, tuple(axes_2d_int[3]), (255, 0, 0), 3, tipLength=0.3)  # Z: Blue
        cv2.circle(img_out, origin, 5, (255, 255, 255), -1)
        
        return img_out


# ============================================================================
# Convenience Functions
# ============================================================================

def undistort_fisheye(img: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                      xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quick function to undistort a fisheye image."""
    h, w = img.shape[:2]
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha, w, h)
    return cam.undistort_image(img)


def solve_pnp_fisheye(points_3d: np.ndarray, points_2d: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      xi: float, alpha: float
                     ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """Quick function to solve PnP for fisheye camera."""
    cam = DoubleSphereCamera(fx, fy, cx, cy, xi, alpha)
    return cam.solve_pnp(points_3d, points_2d)

# ============================================================================
# Standalone Core Functions (Optimized for Calibration)
# ============================================================================

def balanced_pinhole_K(fx: float, fy: float, width: int, height: int,
                       balance: float = 0.5) -> np.ndarray:
    """
    Build a pinhole intrinsic matrix for the undistorted (rectified) image.

    The new focal length is a fraction of the original, controlled by `balance`:
        balance 0.0 -> 0.4x  (widest FOV, more of the scene kept)
        balance 0.5 -> 0.6x  (default)
        balance 1.0 -> 0.8x  (narrowest FOV, least peripheral stretch)
    The principal point is placed at the image center.

    This is the single source of truth shared by DoubleSphereCamera.compute_K_new,
    the LDC mesh generator, and the cv2-style estimateNewCameraMatrix wrapper.
    """
    focal_scale = 0.4 + balance * 0.4
    f_new = ((fx + fy) / 2.0) * focal_scale
    return np.array([
        [f_new, 0.0, width / 2.0],
        [0.0, f_new, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def ds_project(points_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standalone projection function.
    Returns: u, v, valid
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    d1 = np.sqrt(x*x + y*y + z*z)
    z1 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z1*z1)
    den = alpha * d2 + (1.0 - alpha) * z1

    # Projectability is a half-space condition, NOT simply z > 0.
    # Using z > 0 would clip the field of view to < 180 deg and silently drop
    # the very wide-angle rays the Double Sphere model exists to represent.
    # Per Usenko et al. 2018 (Eq. 43-45): the point is projectable iff
    #     z > -w2 * d1,
    # with w1 piecewise in alpha and w2 derived from w1 and xi.
    if alpha > 0.5:
        w1 = (1.0 - alpha) / alpha
    else:
        w1 = alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(max(2.0 * w1 * xi + xi * xi + 1.0, 1e-12))

    valid = (z > -w2 * d1) & (den > 1e-8)
    den = np.maximum(den, 1e-8)

    u = fx * x / den + cx
    v = fy * y / den + cy

    return u, v, valid

def ds_project_jacobian(points_3d: np.ndarray, fx: float, fy: float,
                        cx: float, cy: float, xi: float, alpha: float
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytic Jacobian of the Double Sphere projection.

    Closed-form derivatives are exact, allocation-free, and far cheaper than
    finite differences (no step-size tuning, no cancellation), which makes
    Levenberg-Marquardt / Gauss-Newton calibration both faster and more robust.

    Parameters
    ----------
    points_3d : (..., 3) array
        Points in camera coordinates.

    Returns
    -------
    u, v : (...,) arrays
        Projected pixel coordinates.
    J_point : (..., 2, 3) array
        d(u, v) / d(x, y, z).
    J_intr : (..., 2, 6) array
        d(u, v) / d(fx, fy, cx, cy, xi, alpha).
    valid : (...,) bool array
        Projectability mask (same condition as ds_project).
    """
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]

    d1 = np.sqrt(x*x + y*y + z*z)
    d1 = np.maximum(d1, 1e-12)
    z1 = z + xi * d1
    d2 = np.sqrt(x*x + y*y + z1*z1)
    d2 = np.maximum(d2, 1e-12)
    den = alpha * d2 + (1.0 - alpha) * z1

    if alpha > 0.5:
        w1 = (1.0 - alpha) / alpha
    else:
        w1 = alpha / (1.0 - alpha)
    w2 = (w1 + xi) / np.sqrt(max(2.0 * w1 * xi + xi * xi + 1.0, 1e-12))
    valid = (z > -w2 * d1) & (den > 1e-8)

    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    inv = 1.0 / den
    inv2 = inv * inv
    u = fx * x * inv + cx
    v = fy * y * inv + cy

    # Shared sub-expressions for the denominator's derivatives.
    A = alpha * z1 / d2 + (1.0 - alpha)      # appears in d(den)/dxi and d(den)/dz
    B = 1.0 + xi * z1 / d1
    Cz = 1.0 + xi * z / d1

    dden_dx = alpha * x * B / d2 + (1.0 - alpha) * xi * x / d1
    dden_dy = alpha * y * B / d2 + (1.0 - alpha) * xi * y / d1
    dden_dz = Cz * A
    dden_dxi = d1 * A
    dden_dalpha = d2 - z1

    # Jacobian w.r.t. the 3D point.
    J_point = np.empty(points_3d.shape[:-1] + (2, 3), dtype=np.float64)
    J_point[..., 0, 0] = fx * (den - x * dden_dx) * inv2
    J_point[..., 0, 1] = fx * (-x * dden_dy) * inv2
    J_point[..., 0, 2] = fx * (-x * dden_dz) * inv2
    J_point[..., 1, 0] = fy * (-y * dden_dx) * inv2
    J_point[..., 1, 1] = fy * (den - y * dden_dy) * inv2
    J_point[..., 1, 2] = fy * (-y * dden_dz) * inv2

    # Jacobian w.r.t. intrinsics [fx, fy, cx, cy, xi, alpha].
    J_intr = np.zeros(points_3d.shape[:-1] + (2, 6), dtype=np.float64)
    J_intr[..., 0, 0] = x * inv          # du/dfx
    J_intr[..., 0, 2] = 1.0              # du/dcx
    J_intr[..., 0, 4] = -fx * x * inv2 * dden_dxi
    J_intr[..., 0, 5] = -fx * x * inv2 * dden_dalpha
    J_intr[..., 1, 1] = y * inv          # dv/dfy
    J_intr[..., 1, 3] = 1.0              # dv/dcy
    J_intr[..., 1, 4] = -fy * y * inv2 * dden_dxi
    J_intr[..., 1, 5] = -fy * y * inv2 * dden_dalpha

    return u, v, J_point, J_intr, valid


def ds_unproject(points_2d: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 xi: float, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone unprojection function.
    Returns: rays, valid
    """
    u, v = points_2d[..., 0], points_2d[..., 1]
    
    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx*mx + my*my
    
    # Validity check 1: Sphere intersection check
    s = 1.0 - (2.0 * alpha - 1.0) * r2
    valid_s = s >= 0
    s = np.maximum(s, 0.0)
    
    # Closed-form unprojection
    mz = (1.0 - alpha*alpha * r2) / (alpha * np.sqrt(s) + (1.0 - alpha))
    
    # Validity check 2: Feasible ray scale check (prevents nan in sqrt)
    sqrt_arg = mz*mz + (1.0 - xi*xi) * r2
    valid_sqrt = sqrt_arg >= 0
    
    # Combined validity mask
    valid = valid_s & valid_sqrt
    
    # Safe square root calculation
    sqrt_safe = np.sqrt(np.maximum(sqrt_arg, 0.0))
    k = (mz * xi + sqrt_safe) / np.maximum(mz*mz + r2, 1e-10)
    
    ray = np.stack([k * mx, k * my, k * mz - xi], axis=-1)
    norm = np.linalg.norm(ray, axis=-1, keepdims=True)
    ray = ray / np.maximum(norm, 1e-10)
    
    # Safely zero out invalid rays to prevent NaN propagation
    ray[~valid] = 0.0
    
    return ray, valid
