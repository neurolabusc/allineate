#!/usr/bin/env python3
"""Synthetic capture-range and partial-FOV tests for the fast path.

For each known world affine A (FIXED->MOVING) in coreg_synth.CAPTURE_CASES we build a
moving image whose voxel content equals a generated asymmetric phantom but whose header world
matrix is A @ S_fixed. The true FIXED->MOVING transform is then exactly A, so the fast
path must recover F2M with ERMS(F2M, A) <= floor over a 100 mm sphere.

The partial-FOV cases crop one end of the moving acquisition while preserving its
world coordinates. The true transform remains identity; the fit must not distort
the affine merely to pull unavailable fixed anatomy inside the moving grid.
"""
import os, sys, json, subprocess, tempfile, shutil
import numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(__file__))
import coreg_synth as cs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "allineate")


def make_fixed(tmp):
    """Create a detailed, asymmetric 3D fixture without external imaging data."""
    n = 72
    kk, jj, ii = np.mgrid[0:n, 0:n, 0:n]
    def blob(ci, cj, ck, radius, amplitude):
        return amplitude * np.exp(-((ii-ci)**2 + (jj-cj)**2 + (kk-ck)**2) /
                                  (2.0 * radius * radius))
    data = (blob(36, 36, 36, 15, 80) + blob(49, 30, 37, 6, 180) +
            blob(27, 47, 32, 7, 110) + blob(34, 34, 52, 4, 220) +
            blob(22, 28, 24, 3, 150)).astype(np.float32)
    data[((ii-40)**2/16**2 + (jj-38)**2/11**2 + (kk-33)**2/9**2) < 1] += 35
    S = np.diag([2.0, 2.0, 2.0, 1.0])
    S[:3, 3] = -(n - 1)
    path = os.path.join(tmp, "fixed.nii.gz")
    fx = nib.Nifti1Image(data, S)
    fx.set_sform(S, code=1); fx.set_qform(S, code=0)
    fx.set_data_dtype(np.float32); nib.save(fx, path)
    return path


def run_case(name, A, fixed, tmp):
    fx = nib.load(fixed)
    Sf = fx.affine.astype(np.float64)
    Sm = A @ Sf
    mov = nib.Nifti1Image(np.asarray(fx.dataobj, dtype=np.float32), Sm)
    mov.set_sform(Sm, code=1); mov.set_qform(Sm, code=0)
    mp = os.path.join(tmp, f"mov_{name}.nii.gz")
    nib.save(mov, mp)
    jp = os.path.join(tmp, f"m_{name}.json")
    op = os.path.join(tmp, f"o_{name}.nii.gz")
    cmd = [BIN, mp, fixed, "-cost", "fastcr", "-savemat", jp, op]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timeout (>120s)"
    if r.returncode != 0 or not os.path.exists(jp):
        return None, r.stderr.strip()
    M = np.array(json.load(open(jp))["fixed_to_moving"], dtype=np.float64)
    return cs.erms(M, A), None


def run_partial_fov(cost, fixed, tmp):
    """Crop 32 mm from one moving-grid edge without changing world-space anatomy."""
    fx = nib.load(fixed)
    data = np.asarray(fx.dataobj, dtype=np.float32)
    cut = 16
    Sm = fx.affine.astype(np.float64)
    Sm[:3, 3] += Sm[:3, 2] * cut
    mov = nib.Nifti1Image(data[:, :, cut:], Sm)
    mov.set_sform(Sm, code=1); mov.set_qform(Sm, code=0)
    mp = os.path.join(tmp, f"mov_partial_{cost}.nii.gz")
    jp = os.path.join(tmp, f"m_partial_{cost}.json")
    nib.save(mov, mp)
    cmd = [BIN, mp, fixed, "-cost", cost, "-savemat", jp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "timeout (>120s)"
    if r.returncode != 0 or not os.path.exists(jp):
        return None, r.stderr.strip()
    M = np.array(json.load(open(jp))["fixed_to_moving"], dtype=np.float64)
    return cs.erms(M, np.eye(4)), None


def main():
    tmp = tempfile.mkdtemp(prefix="cf_synth_")
    try:
        fixed = make_fixed(tmp)
        npass = 0; nfail = 0
        print("case            ERMS(mm)  floor   result")
        for name, A in cs.CAPTURE_CASES.items():
            floor = cs.ERMS_IDENTITY_FLOOR_MM if name == "identity" else cs.ERMS_FLOOR_MM
            e, err = run_case(name, A, fixed, tmp)
            if e is None:
                print(f"{name:14s}   ---      {floor:.2f}   FAIL ({err[:50]})"); nfail += 1; continue
            ok = e <= floor
            print(f"{name:14s}  {e:7.3f}   {floor:.2f}   {'PASS' if ok else 'FAIL'}")
            npass += ok; nfail += (not ok)
        for cost, floor in (("fastcr", 0.10), ("fasthel", 1.00)):
            name = f"partial_{cost}"
            e, err = run_partial_fov(cost, fixed, tmp)
            if e is None:
                print(f"{name:14s}   ---      {floor:.2f}   FAIL ({err[:50]})"); nfail += 1; continue
            ok = e <= floor
            print(f"{name:14s}  {e:7.3f}   {floor:.2f}   {'PASS' if ok else 'FAIL'}")
            npass += ok; nfail += (not ok)
        print(f"\n{npass} passed, {nfail} failed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
