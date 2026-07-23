#!/usr/bin/env python3
"""Synthetic capture-range, partial-FOV, and hard-zero-base tests for the fast path.

For each known world affine A (FIXED->MOVING) in coreg_synth.CAPTURE_CASES we build a
moving image whose voxel content equals a generated asymmetric phantom but whose header world
matrix is A @ S_fixed. The true FIXED->MOVING transform is then exactly A, so the fast
path must recover F2M with ERMS(F2M, A) <= floor over a 100 mm sphere.

The partial-FOV cases crop one end of the moving acquisition while preserving its
world coordinates. The true transform remains identity; the fit must not distort
the affine merely to pull unavailable fixed anatomy inside the moving grid.

The hard-zero case masks the tracked MNI template with its tracked brain mask and
registers the tracked T1w1mm image. This is the compact regression for the coarse
multi-start: rigid-only coarse capture lands in a wrong basin (masked NCC ~0.13),
while carrying the scale-bracketed strategy to 2 mm recovers NCC ~0.48.
"""
import argparse, os, sys, json, subprocess, tempfile, shutil
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


def run_hard_zero_base(tmp):
    """Return (masked_ncc, p1_equals_p4, error) for the stripped-template regression."""
    moving = os.path.join(ROOT, "benchmark", "inputs", "T1w1mm.nii.gz")
    template = os.path.join(ROOT, "benchmark", "templates", "MNI152_T1_1mm.nii.gz")
    mask_path = os.path.join(ROOT, "benchmark", "masks", "MNI152_T1_1mm_brain_mask.nii.gz")
    for path in (moving, template, mask_path):
        if not os.path.exists(path):
            return None, False, f"missing tracked fixture: {path}"

    ti = nib.load(template)
    td = np.asarray(ti.dataobj, dtype=np.float32)
    md = np.asarray(nib.load(mask_path).dataobj, dtype=np.float32)
    base_path = os.path.join(tmp, "MNI152_T1_1mm_brain.nii.gz")
    base = nib.Nifti1Image(np.where(md > 0.5, td, 0.0).astype(np.float32),
                           ti.affine, ti.header)
    base.set_data_dtype(np.float32)
    nib.save(base, base_path)

    outputs = []
    for threads in (1, 4):
        out = os.path.join(tmp, f"hardzero_p{threads}.nii.gz")
        try:
            r = subprocess.run([BIN, moving, base_path, "-p", str(threads), out],
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return None, False, f"p{threads} timeout (>120s)"
        if r.returncode != 0 or not os.path.exists(out):
            return None, False, f"p{threads}: {r.stderr.strip()[-120:]}"
        outputs.append(np.asarray(nib.load(out).dataobj, dtype=np.float32))

    roi = md > 0.5
    ncc = float(np.corrcoef(outputs[1][roi], td[roi])[0, 1])
    return ncc, np.array_equal(outputs[0], outputs[1]), None


def main():
    global BIN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allineate", default=BIN)
    args = parser.parse_args()
    BIN = os.path.abspath(args.allineate)
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
        ncc, same, err = run_hard_zero_base(tmp)
        if ncc is None:
            print(f"{'hardzero_ncc':14s}   ---      0.40   FAIL ({err[:50]})")
            nfail += 1
        else:
            ok = ncc >= 0.40
            print(f"{'hardzero_ncc':14s}  {ncc:7.3f}   0.40   {'PASS' if ok else 'FAIL'}")
            npass += ok; nfail += (not ok)
            print(f"{'hardzero_p1=p4':14s}  {'yes' if same else 'no ':>7s}          "
                  f"{'PASS' if same else 'FAIL'}")
            npass += same; nfail += (not same)
        print(f"\n{npass} passed, {nfail} failed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
