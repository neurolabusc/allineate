#!/usr/bin/env python3
"""Deterministic regression suite for allineate — a real pass/fail release gate.

Unlike `benchmark/benchmark.py` (speed/quality on unshareable real scans), this suite
builds **synthetic** fixtures on the fly (no facial/identifiable data, nothing to
commit as a binary) and asserts *correctness* of the geometry/preprocessing/affine
paths. It exits **nonzero** on any failure, so CI or a release script can gate on it.

Covered (each an audit-hardened path):
  1. non-finite `-com`        — NaN/+Inf must not fold NaN into sform/qform
  2. 4D rejection             — a 4D volume must be refused (fail-closed)
  3. save/apply round-trip    — `-applymat` reproduces the fitted transform (NCC ~1)
  4. scaled-float32 deface    — pending negative scl_slope must not invert the mask fill
  5. `-symb` axis-aligned tie — the world frame wins; sform/qform codes are preserved
  6. degenerate sform+qform   — a qform-only-usable image registers (end-to-end reader repair)

Usage:  python3 test/test_regression.py [--allineate ./allineate]
Requires numpy + nibabel (a hard requirement for the gate; missing -> exit 2).
"""
import argparse, os, subprocess, sys, tempfile

try:
    import numpy as np
    import nibabel as nib
except ImportError as e:                                        # pragma: no cover
    print(f"FATAL: regression gate needs numpy+nibabel ({e})", file=sys.stderr)
    sys.exit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILURES = []


def run(exe, args):
    """Run allineate; return (returncode, stderr)."""
    p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stderr


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(f"{name}: {detail}")


def blob(shape=(24, 24, 24), center=None, radius=6.0, val=100.0):
    """A smooth spherical blob — a crude but registrable synthetic 'brain'."""
    c = center if center is not None else [s / 2 for s in shape]
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    r2 = (xx - c[2]) ** 2 + (yy - c[1]) ** 2 + (zz - c[0]) ** 2
    d = val * np.exp(-r2 / (2 * radius ** 2))
    return d.astype(np.float32)


def save(data, affine, path, code=1, sform=True, qform=True, dtype=np.float32,
         scl_slope=None, scl_inter=None):
    img = nib.Nifti1Image(data.astype(dtype), affine)
    if sform:
        img.header.set_sform(affine, code=code)
    if qform:
        img.header.set_qform(affine, code=code)
    img.set_data_dtype(dtype)
    if scl_slope is not None:
        img.header['scl_slope'] = scl_slope
        img.header['scl_inter'] = 0.0 if scl_inter is None else scl_inter
    nib.save(img, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allineate", default=os.path.join(ROOT, "allineate"))
    args = ap.parse_args()
    exe = args.allineate
    if not os.path.exists(exe):
        print(f"FATAL: allineate binary not found at {exe} (run `make` first)", file=sys.stderr)
        sys.exit(2)

    eye = np.diag([1.0, 1.0, 1.0, 1.0])

    with tempfile.TemporaryDirectory() as td:
        j = lambda n: os.path.join(td, n)

        # 1. non-finite -com: a non-finite voxel must be SKIPPED, leaving the centroid
        #    unchanged — not skew it (NaN -> whole accumulator NaN -> silent
        #    geometric-center fallback) nor poison the header (+Inf -> Inf/Inf = NaN
        #    sform). A finiteness-only check on a *centered* fixture cannot catch this:
        #    the buggy fallback lands on the geometric center, which for a centered blob
        #    ~equals the true centroid and is finite. So: use an OFF-center fixture (true
        #    centroid != geometric center), place the bad voxel on a BACKGROUND site
        #    (=0, already skipped, so the only difference from the reference is finite-vs-
        #    non-finite), and compare the folded affine to a non-finite-free reference.
        #    Split +Inf (would give a NaN header) from NaN (would give a finite-but-WRONG
        #    fallback centroid) so their failure modes don't mask each other.
        print("1. non-finite -com")
        off = blob(center=[16, 12, 9])
        off[2, 3, 4] = 0.0                     # a background voxel (skipped in the reference)
        save(off, eye, j("clean.nii.gz"))
        rc, _ = run(exe, [j("clean.nii.gz"), "-com", j("clean_out.nii.gz")])
        check("clean -com exit 0", rc == 0, f"rc={rc}")
        ref = nib.load(j("clean_out.nii.gz")).affine if rc == 0 else None
        for tag, badval in (("+Inf", np.inf), ("NaN", np.nan)):
            d = off.copy()
            d[2, 3, 4] = badval                # same image, that one voxel now non-finite
            save(d, eye, j("nf.nii.gz"))
            rc, _ = run(exe, [j("nf.nii.gz"), "-com", j("nf_out.nii.gz")])
            check(f"{tag}: exit 0", rc == 0, f"rc={rc}")
            if rc == 0:
                aff = nib.load(j("nf_out.nii.gz")).affine
                check(f"{tag}: output affine finite", bool(np.all(np.isfinite(aff))))
                if ref is not None:
                    check(f"{tag}: centroid unchanged vs reference (voxel skipped)",
                          bool(np.allclose(aff, ref, atol=1e-3)),
                          "the non-finite voxel altered the folded -com transform")

        # 2. 4D rejection: a 4D volume must be refused (fail-closed).
        print("2. 4D rejection")
        d4 = np.stack([blob(), blob()], axis=-1).astype(np.float32)
        save(d4, eye, j("vol4d.nii.gz"))
        rc, _ = run(exe, [j("vol4d.nii.gz"), "-sym", j("v4_out.nii.gz")])
        check("4D input rejected (nonzero exit)", rc != 0, f"rc={rc}")

        # 3. save/apply round-trip: -applymat reproduces the fitted transform (NCC ~1).
        print("3. save/apply round-trip")
        fixed = blob(center=[12, 12, 12])
        moving = blob(center=[14, 11, 13])            # shifted 'brain'
        save(fixed, eye, j("fix.nii.gz"))
        save(moving, eye, j("mov.nii.gz"))
        rc1, _ = run(exe, [j("mov.nii.gz"), j("fix.nii.gz"), "-cost", "ls",
                           "-savemat", j("xf.json"), j("reg.nii.gz")])
        rc2, _ = run(exe, [j("mov.nii.gz"), j("fix.nii.gz"), "-applymat", j("xf.json"),
                           j("apply.nii.gz")])
        check("register+savemat exit 0", rc1 == 0, f"rc={rc1}")
        check("applymat exit 0", rc2 == 0, f"rc={rc2}")
        if rc1 == 0 and rc2 == 0:
            a = np.asarray(nib.load(j("reg.nii.gz")).dataobj, np.float32).ravel()
            b = np.asarray(nib.load(j("apply.nii.gz")).dataobj, np.float32).ravel()
            m = np.isfinite(a) & np.isfinite(b)
            ncc = float(np.corrcoef(a[m], b[m])[0, 1])
            check("applymat reproduces registration (NCC>0.999)", ncc > 0.999, f"NCC={ncc:.5f}")

        # 4. scaled-float32 deface: pending negative slope must not invert the fill.
        print("4. scaled-float32 deface")
        tmpl = blob(center=[12, 12, 12])
        mask = (blob(center=[12, 12, 12], radius=5.0) > 20).astype(np.float32)
        save(moving, eye, j("mov_neg.nii.gz"), scl_slope=-1.0)   # pending negative scale
        save(tmpl, eye, j("tmpl.nii.gz"))
        save(mask, eye, j("mask.nii.gz"))
        rc, _ = run(exe, [j("mov_neg.nii.gz"), j("tmpl.nii.gz"),
                          "-skullstrip", j("mask.nii.gz"), j("strip.nii.gz")])
        check("scaled-float32 -skullstrip exit 0", rc == 0, f"rc={rc}")
        if rc == 0:
            out = np.asarray(nib.load(j("strip.nii.gz")).dataobj, np.float32)
            # masked-out voxels are set to the image minimum (darkest); the modal value
            # should equal the min, not the max (the negative-slope inversion bug).
            check("masked fill is the darkest value",
                  float(out.min()) < float(out.max()) and
                  (out == out.min()).mean() > 0.3,
                  f"min={out.min():.3f} max={out.max():.3f} frac@min={(out==out.min()).mean():.2f}")

        # 5. -symb axis-aligned tie: world frame wins; sform/qform codes preserved.
        print("5. -symb axis-aligned tie")
        aa = np.diag([2.0, 2.0, 2.0, 1.0])            # clean axis-aligned frame
        save(blob(), aa, j("aa.nii.gz"), code=4)      # MNI-like code 4
        rc, err = run(exe, [j("aa.nii.gz"), "-symb", j("aa_out.nii.gz")])
        check("-symb exit 0", rc == 0, f"rc={rc}")
        if rc == 0:
            check("-symb kept the world frame on the tie",
                  "chose world frame" in err, "expected 'chose world frame' in log")
            h = nib.load(j("aa_out.nii.gz"))
            check("sform_code preserved (4)", int(h.header['sform_code']) == 4,
                  f"sform_code={int(h.header['sform_code'])}")
            check("qform_code preserved (4)", int(h.header['qform_code']) == 4,
                  f"qform_code={int(h.header['qform_code'])}")

        # 6. degenerate sform + valid qform -> end-to-end reader repair -> registers.
        #    NOTE: this is NOT the engine's al_image_xform sform->qform fallback branch.
        #    nifti_sync_sform_from_qform() repairs a degenerate coded sform from a valid
        #    qform at READ time (before the engine sees it), so al_image_xform receives a
        #    synthesized-but-valid sform. This is an end-to-end CLI/reader test: a
        #    qform-only-usable image must still register. (Exercising al_image_xform's own
        #    fallback would need an in-memory API fixture that bypasses the reader sync.)
        print("6. degenerate sform + valid qform (reader repair -> registers)")
        deg = np.zeros((4, 4))                          # all-zero (degenerate) sform
        good = np.diag([1.0, 1.0, 1.0, 1.0])
        img = nib.Nifti1Image(moving, good)
        img.header.set_sform(deg, code=1)              # bogus sform...
        img.header.set_qform(good, code=1)             # ...valid qform
        img.set_data_dtype(np.float32)
        nib.save(img, j("degsform.nii.gz"))
        rc, err = run(exe, [j("degsform.nii.gz"), j("fix.nii.gz"), "-cost", "ls",
                            j("deg_out.nii.gz")])
        check("qform-only-usable image registers (exit 0)", rc == 0,
              f"rc={rc}: {err.strip()[-160:]}")

    print()
    if FAILURES:
        print(f"REGRESSION FAILED — {len(FAILURES)} check(s):", file=sys.stderr)
        for f in FAILURES:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    print("All regression checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
