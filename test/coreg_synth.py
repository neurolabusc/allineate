"""Synthetic capture-range definitions + ERMS transform-distance scorer for the
fast path `-cost fast`/`-cost fastcr` (Phase 0 deliverable, committed before fitting constants to it).

The capture suite defines *known* world-space FIXED->MOVING affines within the
supported envelope (adult brain; see AGENTS.md). A synthetic moving image is produced by resampling a
real fixed template through the inverse of the known affine; the fast path must then
recover that affine to within the ERMS floor.

ERMS: root-mean-square displacement (mm) between two world affines over a solid sphere
of radius R centred at the world origin. For A,B in homogeneous 4x4 form with linear
parts La,Lb and translations ta,tb, D=La-Lb, t=ta-tb, and E[x x^T]=(R^2/5) I over the
ball, the closed form is  ERMS = sqrt( (R^2/5)*||D||_F^2 + ||t||^2 ).
"""
import numpy as np

R_SPHERE_MM = 100.0


def erms(A, B, R=R_SPHERE_MM):
    """RMS displacement (mm) between two 4x4 world affines over radius-R sphere."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    D = A[:3, :3] - B[:3, :3]
    t = A[:3, 3] - B[:3, 3]
    return float(np.sqrt((R * R / 5.0) * np.sum(D * D) + np.sum(t * t)))


def _rot(ax, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    M = np.eye(4)
    if ax == 0:
        M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    elif ax == 1:
        M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    else:
        M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def _trans(tx, ty, tz):
    M = np.eye(4); M[:3, 3] = [tx, ty, tz]; return M


def _scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def _shear(sxy):
    M = np.eye(4); M[0, 1] = sxy; return M


# Named capture cases: world-space FIXED->MOVING affine (mm). Identity + separate and
# combined perturbations at / inside the supported envelope boundaries.
CAPTURE_CASES = {
    "identity":      np.eye(4),
    "trans_x":       _trans(30, 0, 0),
    "trans_xyz":     _trans(18, -22, 14),
    "rot_x":         _rot(0, 30),
    "rot_z":         _rot(2, -25),
    "rot_xyz":       _rot(0, 12) @ _rot(1, -15) @ _rot(2, 20),
    "gscale_up":     _scale(1.33, 1.33, 1.33),
    "gscale_dn":     _scale(0.75, 0.75, 0.75),
    "ascale":        _scale(1.20, 0.85, 1.10),
    "shear":         _shear(0.08),
    "combined":      _trans(12, -8, 6) @ _rot(0, 10) @ _rot(1, -8) @ _rot(2, 14) @ _scale(1.1, 0.92, 1.05),
}

# ERMS floors (mm) over a 100 mm sphere, for the fast-path capture suite.
ERMS_FLOOR_MM = 1.0
ERMS_IDENTITY_FLOOR_MM = 0.1
