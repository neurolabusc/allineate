#!/usr/bin/env python3
"""Self-contained -qwarp gate (part of `make test`).

Exercises the exclusive `allineate <moving> <stationary> -qwarp <output>` grammar and the output
geometry contract, PLUS an end-to-end correctness check (§1b): a synthetic phantom deformed by a
smooth analytic displacement, requiring the warp output to be finite, source-range-bounded, and to
improve Pearson-to-base over the unwarped source by a fixed margin. The AFNI numerical-equivalence
gate is a SEPARATE developer script (test/qwarp_compare.py) over the gitignored qwarp/ data; the
C-API boundary/atomic-failure gate is test/test_qwarp_capi.c.
"""
import os, sys, subprocess, tempfile, shutil, atexit, argparse
import numpy as np
import nibabel as nib

npass = nfail = 0
def check(name, cond, detail=""):
    global npass, nfail
    if cond: npass += 1; print(f"  [PASS] {name}")
    else:    nfail += 1; print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

def pearson(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    da = np.sqrt((a*a).sum()); db = np.sqrt((b*b).sum())
    return float((a*b).sum() / (da*db)) if da > 0 and db > 0 else float("nan")

def save(data, aff, path):
    im = nib.Nifti1Image(np.ascontiguousarray(data), aff)
    im.set_sform(aff, code=1); im.set_qform(aff, code=1)
    nib.save(im, path)

def run(bin_, args):
    p = subprocess.run([bin_] + args, capture_output=True, text=True, timeout=120)
    return p.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allineate", default="./allineate")
    args = ap.parse_args()
    exe = args.allineate
    tmp = tempfile.mkdtemp(prefix="qwarp_cli_")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)   # cleanup even on early exit/exception
    j = lambda n: os.path.join(tmp, n)

    # A small shared-grid pair: an asymmetric blob (moving) and a shifted blob (stationary).
    n = 40
    aff = np.diag([1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = -(n / 2.0)
    kk, jj, ii = np.mgrid[0:n, 0:n, 0:n]
    def blob(ci, cj, ck, r):
        return (100.0 * np.exp(-((ii-ci)**2 + (jj-cj)**2 + (kk-ck)**2) / (2.0*r*r))).astype(np.float32)
    save(blob(20, 20, 20, 7) + blob(26, 18, 22, 3), aff, j("mov.nii.gz"))
    save(blob(21, 19, 20, 7) + blob(27, 17, 22, 3), aff, j("sta.nii.gz"))
    # Mismatched-dims stationary and a 4D moving, for rejection tests.
    aff2 = np.diag([1.0, 1.0, 1.0, 1.0]); aff2[:3, 3] = -((n+4) / 2.0)
    save(np.zeros((n+4, n, n), np.float32), aff2, j("sta_bad.nii.gz"))
    save(np.stack([blob(20, 20, 20, 7), blob(20, 20, 20, 7)], axis=-1), aff, j("mov4d.nii.gz"))

    mov, sta, out = j("mov.nii.gz"), j("sta.nii.gz"), j("out.nii.gz")

    print("1. valid -qwarp invocation")
    rc = run(exe, [mov, sta, "-qwarp", out])
    check("valid `<moving> <stationary> -qwarp <output>` exits 0", rc == 0, f"rc={rc}")
    if rc == 0 and os.path.exists(out):
        oi = nib.load(out); si = nib.load(sta)
        check("output is on the stationary grid (dims)", oi.shape == si.shape, f"{oi.shape} vs {si.shape}")
        check("output affine matches stationary", np.allclose(oi.affine, si.affine, atol=1e-3))
        check("output datatype is float32", oi.get_data_dtype() == np.float32, str(oi.get_data_dtype()))
    else:
        check("output written", False, "no output file")

    print("1b. end-to-end: the warp must IMPROVE similarity and stay finite/bounded")
    # A structured 3D phantom (base) and a source that is the SAME phantom sampled through a
    # smooth low-frequency displacement (a nonlinear deformation qwarp should partly undo).
    # Evaluated analytically so there is no scipy dependency and the "truth" is exact.
    m = 52
    baff = np.diag([1.0, 1.0, 1.0, 1.0]); baff[:3, 3] = -(m / 2.0)
    zc, yc, xc = np.mgrid[0:m, 0:m, 0:m].astype(np.float64)
    def phantom(x, y, z):
        v = np.zeros_like(x)
        for (cx, cy, cz, r, a) in [(26,26,26,11,100.0),(20,32,24,5,70.0),(34,22,30,4,-55.0)]:
            v += a * np.exp(-((x-cx)**2 + (y-cy)**2 + (z-cz)**2) / (2.0*r*r))
        v += 45.0 * (((x-26)**2 + (y-26)**2 + (z-26)**2) < 15.0**2)   # a sharp sphere edge
        return v
    amp = 2.5
    dx = amp * np.sin(2*np.pi*yc/m) * np.cos(2*np.pi*zc/m)
    dy = amp * np.sin(2*np.pi*zc/m)
    dz = amp * np.cos(2*np.pi*xc/m)
    base_d = phantom(xc, yc, zc).astype(np.float32)
    src_d  = phantom(xc+dx, yc+dy, zc+dz).astype(np.float32)
    save(base_d, baff, j("ph_base.nii.gz"))
    save(src_d,  baff, j("ph_src.nii.gz"))
    pout = j("ph_out.nii.gz")
    rc = run(exe, [j("ph_src.nii.gz"), j("ph_base.nii.gz"), "-qwarp", pout])
    check("phantom -qwarp exits 0", rc == 0, f"rc={rc}")
    if rc == 0 and os.path.exists(pout):
        od = np.asarray(nib.load(pout).dataobj, dtype=np.float64)
        check("output is finite (no NaN/Inf)", np.all(np.isfinite(od)))
        # WSINC5 clamps to the PADDED source range, which includes the zero-fill padding, so the
        # bound is [min(src.min,0), max(src.max,0)] not the unpadded phantom's own min/max.
        lo, hi = min(float(src_d.min()), 0.0), max(float(src_d.max()), 0.0)
        check("output bounded to padded source range (WSINC5 clamp)",
              od.min() >= lo - 1e-2 and od.max() <= hi + 1e-2, f"[{od.min():.2f},{od.max():.2f}] vs [{lo:.2f},{hi:.2f}]")
        p_before = pearson(src_d.astype(np.float64), base_d.astype(np.float64))
        p_after  = pearson(od, base_d.astype(np.float64))
        # Conservative gate (observed: before~0.948, after~0.992). Require a fixed minimum
        # improvement AND a final-correlation floor, so a near-identity result, an early optimizer
        # exit, or platform noise cannot pass. Both values are printed for drift visibility.
        MIN_GAIN, FINAL_FLOOR = 0.02, 0.97
        print(f"      Pearson(_,base): before={p_before:.4f} after={p_after:.4f} gain={p_after-p_before:+.4f}")
        check(f"warp improves similarity (gain>={MIN_GAIN} and final>={FINAL_FLOOR})",
              (p_after - p_before) >= MIN_GAIN and p_after >= FINAL_FLOOR,
              f"before={p_before:.4f} after={p_after:.4f}")
    else:
        check("phantom output written", False, "no output file")

    print("2. exclusive-mode rejections (nonzero exit, no output written)")
    def rejects(name, argv):
        o = j(f"rej_{name}.nii.gz")
        av = [a if a != out else o for a in argv]
        rc = run(exe, av)
        check(f"reject: {name} (nonzero exit)", rc != 0, f"rc={rc}")
        check(f"reject: {name} writes no output", not os.path.exists(o))
    rejects("extra-option", [mov, sta, "-qwarp", "-blur", "0", "3", out])
    rejects("missing-stationary", [mov, "-qwarp", out])
    rejects("stdin-moving", ["-", sta, "-qwarp", out])
    rejects("stdout-output", [mov, sta, "-qwarp", "-"])
    rejects("mismatched-dims", [mov, j("sta_bad.nii.gz"), "-qwarp", out])
    rejects("4d-moving", [j("mov4d.nii.gz"), sta, "-qwarp", out])
    rejects("extra-operand", [mov, sta, mov, "-qwarp", out])
    rejects("qwarp-after-output", [mov, sta, out, "-qwarp"])
    rejects("qwarp-before-stationary", [mov, "-qwarp", sta, out])

    print(f"\n{npass} passed, {nfail} failed")
    return 1 if nfail else 0

if __name__ == "__main__":
    sys.exit(main())
