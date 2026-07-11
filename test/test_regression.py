#!/usr/bin/env python3
"""Deterministic regression suite for allineate — a real pass/fail release gate.

Unlike `benchmark/benchmark.py` (speed/quality on the bundled shareable scans, descriptive
only), this suite builds **synthetic** fixtures on the fly and asserts *correctness* of the
geometry/preprocessing/affine paths. It exits **nonzero** on any failure, so CI or a release
script can gate on it. The fast-engine capture recovery (§7) is complemented by the broader
11-case template-based suite in `test/test_coreg_fast.py`, which `make test` also runs.

Covered (each an audit-hardened path):
  1. non-finite `-com`        — NaN/+Inf must not fold NaN into sform/qform
  2. 4D rejection             — a 4D volume must be refused (fail-closed)
  3. save/apply round-trip    — `-applymat` reproduces the fitted transform (NCC ~1)
  4. scaled-float32 deface    — pending negative scl_slope must not invert the mask fill
  5. `-symb` axis-aligned tie — the world frame wins; sform/qform codes are preserved
  6. degenerate sform+qform   — a qform-only-usable image registers (end-to-end reader repair)
  7. fast engine recovery     — synthetic capture + option rejection + `-cmass/-nocmass` + p1==pN
  8. cross-modal recovery     — box-weight/Hellinger guard (inverted-contrast fixture) + bit-identity
  9. revoxelized recovery     — moving resampled onto a different grid, known affine, all engines
 10. `-master` output grid    — reslice onto a finer grid == `-savemat`+`-applymat` (both engines)

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

        # 7. fast engine (-cost fastcr): synthetic capture recovery + save/apply + 4D rejection.
        #    Ground truth via the header trick: a moving image sharing the fixed
        #    voxels but with world matrix A @ S_fixed has exact FIXED->MOVING = A, so
        #    the recovered fixed_to_moving must match A (ERMS over a 100 mm sphere).
        print("7. fast engine (-cost fastcr) affine recovery")
        import json as _json
        from coreg_synth import erms as _erms  # shared scorer (also used by test_coreg_fast.py)
        # asymmetric phantom (identifiable under rotation/scale), 2 mm isotropic
        n = 44; sp = 2.0
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        def _g(cx, cy, cz, r, a):
            return a * np.exp(-(((xx-cx)**2+(yy-cy)**2+(zz-cz)**2)/(2*r*r)))
        phan = (_g(22, 22, 22, 7, 100) + _g(30, 20, 22, 3, 160) +
                _g(18, 28, 24, 4, 60) + _g(22, 22, 30, 2.5, 200)).astype(np.float32)
        Sf = np.diag([sp, sp, sp, 1.0]); Sf[:3, 3] = -(n - 1) * 0.5 * sp
        save(phan, Sf, j("cf_fix.nii.gz"))
        def _rotx(d):
            c, s = np.cos(np.radians(d)), np.sin(np.radians(d)); M = np.eye(4)
            M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c; return M
        _trans = np.eye(4); _trans[:3, 3] = [8, -6, 4]
        cases = {
            "identity": (np.eye(4), 0.1),
            "rigid":    (_trans @ _rotx(12), 1.0),
            "gscale":   (np.diag([1.15, 1.15, 1.15, 1.0]), 1.0),
        }
        for nm, (A, floor) in cases.items():
            Sm = A @ Sf
            img = nib.Nifti1Image(phan, Sm)
            img.header.set_sform(Sm, code=1); img.header.set_qform(Sm, code=0)
            img.set_data_dtype(np.float32); nib.save(img, j(f"cf_mov_{nm}.nii.gz"))
            rc, err = run(exe, [j(f"cf_mov_{nm}.nii.gz"), j("cf_fix.nii.gz"), "-cost", "fastcr",
                                "-savemat", j(f"cf_{nm}.json"), j(f"cf_out_{nm}.nii.gz")])
            ok = rc == 0 and os.path.exists(j(f"cf_{nm}.json"))
            check(f"coreg fast {nm}: exit 0 + savemat", ok, f"rc={rc}: {err.strip()[-120:]}")
            if ok:
                M = np.array(_json.load(open(j(f"cf_{nm}.json")))["fixed_to_moving"])
                e = _erms(M, A)
                check(f"coreg fast {nm}: ERMS {e:.3f} <= {floor} mm", e <= floor, f"ERMS={e:.3f}")
        # save/apply round-trip: applymat reproduces the fast fit (NCC ~ 1)
        if os.path.exists(j("cf_rigid.json")):
            rc, err = run(exe, [j("cf_mov_rigid.nii.gz"), j("cf_fix.nii.gz"),
                                "-applymat", j("cf_rigid.json"), j("cf_apply.nii.gz")])
            check("coreg fast save/apply exit 0", rc == 0, f"rc={rc}: {err.strip()[-120:]}")
            if rc == 0:
                a = np.asarray(nib.load(j("cf_out_rigid.nii.gz")).dataobj, float)
                b = np.asarray(nib.load(j("cf_apply.nii.gz")).dataobj, float)
                m = (a != 0) | (b != 0); av = a[m] - a[m].mean(); bv = b[m] - b[m].mean()
                den = np.sqrt((av*av).sum() * (bv*bv).sum())
                ncc = float((av*bv).sum()/den) if den > 0 else 0.0
                check(f"coreg fast applymat reproduces fit (NCC {ncc:.5f} > 0.999)", ncc > 0.999, f"NCC={ncc:.5f}")
        # serial vs multi-thread determinism: the parallel CR reduction uses a fixed,
        # thread-count-independent chunk layout, so -p 1 and -p N must produce a
        # BIT-IDENTICAL affine (guards against a per-thread reduction that NEWUOA would
        # amplify past the 0.05 mm agreement floor).
        det_ok = True
        mats = {}
        for nthr in (1, 4):
            env = dict(os.environ, OMP_NUM_THREADS=str(nthr))
            jp = j(f"cf_det_{nthr}.json")
            p = subprocess.run([exe, j("cf_mov_rigid.nii.gz"), j("cf_fix.nii.gz"),
                                "-cost", "fastcr", "-savemat", jp, j(f"cf_det_{nthr}.nii.gz")],
                               capture_output=True, text=True, timeout=120, env=env)
            if p.returncode != 0 or not os.path.exists(jp):
                det_ok = False; break
            mats[nthr] = np.array(_json.load(open(jp))["fixed_to_moving"], dtype=np.float64)
        check("coreg fast serial/parallel fit is bit-identical",
              det_ok and np.array_equal(mats.get(1), mats.get(4)),
              "-p 1 and -p 4 affines differ (non-deterministic CR reduction)")
        # 4D rejection
        vol4d = np.stack([phan, phan], axis=-1)
        save(vol4d, Sf, j("cf_4d.nii.gz"))
        rc, _ = run(exe, [j("cf_4d.nii.gz"), j("cf_fix.nii.gz"), "-cost", "fastcr", j("cf_4d_out.nii.gz")])
        check("coreg fast rejects 4D moving (nonzero exit)", rc != 0, f"rc={rc}")

        # no-overlap must FAIL (nonzero exit), not silently save an identity matrix.
        Sfar = Sf.copy(); Sfar[:3, 3] += 500.0     # move the head 500 mm away -> no overlap
        img = nib.Nifti1Image(phan, Sfar)
        img.header.set_sform(Sfar, code=1); img.header.set_qform(Sfar, code=0)
        img.set_data_dtype(np.float32); nib.save(img, j("cf_far.nii.gz"))
        farjson = j("cf_far.json")
        rc, _ = run(exe, [j("cf_far.nii.gz"), j("cf_fix.nii.gz"), "-cost", "fastcr",
                          "-savemat", farjson, j("cf_far_out.nii.gz")])
        check("coreg fast no-overlap fails (nonzero exit)", rc != 0, f"rc={rc}")
        check("coreg fast no-overlap writes no matrix", not os.path.exists(farjson),
              "a failed fit must not leave a -savemat artifact")

        # matrix-only mode (-savemat, no output image) succeeds and writes the matrix
        # without performing the final reslice.
        mjson = j("cf_matonly.json")
        rc, err = run(exe, [j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"),
                            "-cost", "fastcr", "-savemat", mjson])
        check("coreg fast matrix-only exit 0 + matrix", rc == 0 and os.path.exists(mjson),
              f"rc={rc}: {err.strip()[-120:]}")
        if os.path.exists(mjson):
            meta = _json.load(open(mjson))
            check("coreg fast savemat records engine=coreg_fast dof=12 cost=cr",
                  meta.get("engine") == "coreg_fast" and meta.get("dof") == 12 and meta.get("cost") == "cr",
                  f"engine={meta.get('engine')} dof={meta.get('dof')} cost={meta.get('cost')}")
        # -cost fast selects the Hellinger cost (its recovery is validated on real sharp images in
        # test_coreg_fast.py; on the smooth synthetic phantom MI is imprecise, so only check wiring).
        hjson = j("cf_hel.json")
        rc, err = run(exe, [j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"), "-cost", "fast", "-savemat", hjson])
        ok = rc == 0 and os.path.exists(hjson) and _json.load(open(hjson)).get("cost") == "hel"
        check("coreg -cost fast accepted + savemat records cost=hel", ok,
              f"rc={rc}: {err.strip()[-120:]}")

        # unsupported allineate options are rejected (not silently ignored) by the fast engine
        for opt in (["-cost", "ls"], ["-warp", "shr"], ["-interp", "cubic"],
                    ["-source_automask"]):
            rc, _ = run(exe, [j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"),
                              "-cost", "fastcr", *opt, j("cf_rej.nii.gz")])
            check(f"coreg fast rejects {opt[0]} (nonzero exit)", rc != 0, f"rc={rc}")
        # ...but -final (output interpolation) IS honored by the fast engine
        rc, _ = run(exe, [j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"),
                          "-cost", "fastcr", "-final", "nearest", j("cf_fin.nii.gz")])
        check("coreg fast accepts -final (exit 0)", rc == 0, f"rc={rc}")
        # -cmass affirms default automatic initialization selection; -nocmass forces the supplied
        # affine. Both must be accepted and still produce a valid fit on the identity phantom.
        for cm in ("-cmass", "-nocmass"):
            cmj = j(f"cf_cm{cm}.json")
            rc, err = run(exe, [j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"),
                                "-cost", "fastcr", cm, "-savemat", cmj])
            check(f"coreg fast accepts {cm} + writes savemat", rc == 0 and os.path.exists(cmj),
                  f"rc={rc}: {err.strip()[-120:]}")

        # Ambient COREG_FAST_* variables are deliberately inert: the estimator is configured
        # only through its explicit options, so an embedding process cannot accidentally alter
        # a fit. Exercise every former override and require the same resolved config + affine.
        envd = dict(os.environ, COREG_FAST_VERBOSE="1", COREG_FAST_COST="ls",
                    COREG_FAST_NOCOARSE="1", COREG_FAST_MAXDOF="6",
                    COREG_FAST_MAXLV="1", COREG_FAST_THR="0.4", COREG_FAST_SEED="hdr")
        mj = j("cf_env_ignored.json")
        p = subprocess.run([exe, j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz"),
                            "-cost", "fast", "-savemat", mj], capture_output=True, text=True,
                           timeout=120, env=envd)
        metae = _json.load(open(mj)) if os.path.exists(mj) else {}
        baseh = _json.load(open(hjson)) if os.path.exists(hjson) else {}
        same = ("fixed_to_moving" in metae and "fixed_to_moving" in baseh and
                np.array_equal(np.array(metae["fixed_to_moving"]),
                               np.array(baseh["fixed_to_moving"])))
        check("coreg fast ignores ambient COREG_FAST_* configuration",
              p.returncode == 0 and metae.get("dof") == 12 and metae.get("cost") == "hel" and same,
              f"rc={p.returncode} dof={metae.get('dof')} cost={metae.get('cost')} same={same}")

        # 8. Cross-modal recovery — the guard for the box-weight/Hellinger changes to the allineate
        #    DEFAULT (AL_NPT_MATCH_MIN, binary box weight, topclip membership, live sample clips)
        #    AND for -cost fast. A moving image with INVERTED, nonlinear contrast vs the fixed (a
        #    T1<->T2 proxy: bright<->dark; least-squares/CR would fail, Hellinger + the box weight
        #    recover it) is warped by a known affine A. The loose ERMS/det bounds catch the
        #    overlap-shrink / wrong-scale failures the box weight fixed, while tolerating MI's
        #    lower precision on smooth phantoms.
        print("8. cross-modal recovery (allineate + -cost fast) + -cost fast bit-identity")
        m = 64; msp = 2.0
        mzz, myy, mxx = np.mgrid[0:m, 0:m, 0:m]
        def _cb(cx, cy, cz, r, a):
            return a * np.exp(-(((mxx-cx)**2+(myy-cy)**2+(mzz-cz)**2)/(2.0*r*r)))
        xphan = (_cb(32,32,32,12,100)+_cb(42,26,32,6,200)+_cb(24,40,34,7,60)+_cb(32,32,42,4,220)).astype(np.float32)
        Sxf = np.diag([msp, msp, msp, 1.0]); Sxf[:3, 3] = -(m-1)*0.5*msp
        save(xphan, Sxf, j("xm_fix.nii.gz"))
        xinv = xphan.max() - xphan; xinv = (xinv*xinv/xinv.max()).astype(np.float32)  # invert + nonlinear
        Ax = np.eye(4); Ax[:3, 3] = [10.0, -8.0, 6.0]
        Ax = Ax @ _rotx(12) @ np.diag([1.1, 0.92, 1.05, 1.0])
        Sxm = Ax @ Sxf
        ximg = nib.Nifti1Image(xinv, Sxm)
        ximg.header.set_sform(Sxm, code=1); ximg.header.set_qform(Sxm, code=0)
        ximg.set_data_dtype(np.float32); nib.save(ximg, j("xm_mov.nii.gz"))
        for eng, flag, bound in (("allineate", [], 2.5), ("cost fast", ["-cost", "fast"], 3.5)):
            xj = j(f"xm_{eng.replace(' ', '_')}.json")   # per-engine path: a stale matrix from a
            xo = j(f"xm_{eng.replace(' ', '_')}.nii.gz")  # prior engine must not satisfy this check
            rc, err = run(exe, [j("xm_mov.nii.gz"), j("xm_fix.nii.gz"), *flag, "-savemat", xj, xo])
            ok = rc == 0 and os.path.exists(xj)
            check(f"cross-modal {eng}: exit 0 + savemat", ok, f"rc={rc}: {err.strip()[-120:]}")
            if ok:
                M = np.array(_json.load(open(xj))["fixed_to_moving"])
                e = _erms(M, Ax); dt = float(np.linalg.det(M[:3, :3]))
                check(f"cross-modal {eng}: ERMS {e:.2f} <= {bound} mm (no shrink)", e <= bound, f"ERMS={e:.2f}")
                check(f"cross-modal {eng}: det {dt:.2f} in [0.7,1.4]", 0.7 <= dt <= 1.4, f"det={dt:.2f}")
        # -cost fast bit-identity across thread counts (guards the chunked-reduction determinism)
        run(exe, [j("xm_mov.nii.gz"), j("xm_fix.nii.gz"), "-cost", "fast", "-p", "1", "-savemat", j("xh1.json"), j("xo.nii.gz")])
        run(exe, [j("xm_mov.nii.gz"), j("xm_fix.nii.gz"), "-cost", "fast", "-p", "4", "-savemat", j("xh4.json"), j("xo.nii.gz")])
        bid = (os.path.exists(j("xh1.json")) and os.path.exists(j("xh4.json")) and
               np.abs(np.array(_json.load(open(j("xh1.json")))["fixed_to_moving"]) -
                      np.array(_json.load(open(j("xh4.json")))["fixed_to_moving"])).max() == 0)
        check("-cost fast bit-identical p1 vs p4", bid, "matrices differ across thread counts")

        # 9. Physically-revoxelized recovery — the moving image is RESAMPLED onto a DIFFERENT grid
        #    (68^3 @ 1.7 mm, shifted origin) than the fixed (60^3 @ 2 mm) through a known world
        #    affine A. Unlike the header-trick fixtures (§7/§8, same lattice), this exercises
        #    fractional sampling / unequal resolution+FOV+origin. Asserts fixed->moving recovery ~ A
        #    for allineate / -cost fastcr / -cost fast. Trilinear is hand-rolled to keep the gate numpy+nibabel-only.
        print("9. physically-revoxelized recovery (different grid, known affine)")
        def _trilerp(vol, ci, cj, ck):
            n0, n1, n2 = vol.shape
            i0 = np.floor(ci).astype(int); j0 = np.floor(cj).astype(int); k0 = np.floor(ck).astype(int)
            di = ci - i0; dj = cj - j0; dk = ck - k0; out = np.zeros(ci.shape, np.float32)
            for oi, wi in ((0, 1-di), (1, di)):
                for oj, wj in ((0, 1-dj), (1, dj)):
                    for ok, wk in ((0, 1-dk), (1, dk)):
                        a = i0+oi; b = j0+oj; c = k0+ok
                        v = (a >= 0)&(a < n0)&(b >= 0)&(b < n1)&(c >= 0)&(c < n2)
                        out += (wi*wj*wk*v) * vol[np.clip(a,0,n0-1), np.clip(b,0,n1-1), np.clip(c,0,n2-1)]
            return out
        nrf = 60; srf = 2.0
        ri, rj, rk = np.mgrid[0:nrf, 0:nrf, 0:nrf]
        def _rb(ci, cj, ck, r, a): return a*np.exp(-(((ri-ci)**2+(rj-cj)**2+(rk-ck)**2)/(2.0*r*r)))
        rfix = (_rb(30,30,30,11,100)+_rb(40,24,30,5,200)+_rb(22,38,32,6,60)+_rb(30,30,40,4,220)).astype(np.float32)
        S1r = np.diag([srf, srf, srf, 1.0]); S1r[:3, 3] = -(nrf-1)*0.5*srf
        save(rfix, S1r, j("rv_fix.nii.gz"))
        Ar = (np.array([[1.,0,0,9.],[0,1,0,-7.],[0,0,1,5.],[0,0,0,1.]]) @ _rotx(10)
              @ np.diag([1.08, 0.95, 1.03, 1.0]))
        nrm = 68; srm = 1.7
        S2r = np.diag([srm, srm, srm, 1.0]); S2r[:3, 3] = -(nrm-1)*0.5*srm + np.array([3., -2., 4.])
        Mrs = np.linalg.inv(S1r) @ np.linalg.inv(Ar) @ S2r  # moving idx -> fixed idx
        gi, gj, gk = np.mgrid[0:nrm, 0:nrm, 0:nrm]
        fv = Mrs @ np.stack([gi.ravel(), gj.ravel(), gk.ravel(), np.ones(gi.size)])
        rmov = _trilerp(rfix, fv[0], fv[1], fv[2]).reshape(nrm, nrm, nrm)
        rimg = nib.Nifti1Image(rmov, S2r); rimg.header.set_sform(S2r, code=1); rimg.header.set_qform(S2r, code=0)
        rimg.set_data_dtype(np.float32); nib.save(rimg, j("rv_mov.nii.gz"))
        for eng, flag, bound in (("allineate", [], 2.5), ("cost fastcr", ["-cost", "fastcr"], 2.5), ("cost fast", ["-cost", "fast"], 5.0)):
            rj = j(f"rv_{eng.replace(' ', '_')}.json")    # per-engine: no stale matrix carryover
            ro = j(f"rv_{eng.replace(' ', '_')}.nii.gz")
            rc, err = run(exe, [j("rv_mov.nii.gz"), j("rv_fix.nii.gz"), *flag, "-savemat", rj, ro])
            ok = rc == 0 and os.path.exists(rj)
            check(f"revoxelized {eng}: exit 0 + savemat", ok, f"rc={rc}: {err.strip()[-120:]}")
            if ok:
                e = _erms(np.array(_json.load(open(rj))["fixed_to_moving"]), Ar)
                check(f"revoxelized {eng}: ERMS {e:.2f} <= {bound} mm", e <= bound, f"ERMS={e:.2f}")

        # 10. -master output-grid override: register at the stationary resolution but reslice
        #     the result onto a DIFFERENT grid sharing the fixed world frame (e.g. a higher-res
        #     template). The FIT must be unchanged (matrix identical to a plain run) and the
        #     output must be byte-identical to the equivalent -savemat + -applymat two-step —
        #     -master is exactly that two-step built in. Checked for both engines.
        print("10. -master output-grid override")
        save(blob(center=[12, 12, 12]), eye, j("ms_fix.nii.gz"))
        save(blob(center=[13, 11, 12]), eye, j("ms_mov.nii.gz"))
        fine = np.diag([0.5, 0.5, 0.5, 1.0])           # finer grid, same (eye) world frame
        save(np.zeros((48, 48, 48), np.float32), fine, j("ms_master.nii.gz"))
        for eng, flag in (("allineate", ["-cost", "ls"]), ("cost fastcr", ["-cost", "fastcr"])):
            sl = eng.replace(" ", "_")                     # per-engine paths: no stale carryover
            mj, pj = j(f"ms_m_{sl}.json"), j(f"ms_p_{sl}.json")
            one_p, two_p = j(f"ms_one_{sl}.nii.gz"), j(f"ms_two_{sl}.nii.gz")
            rc1, e1 = run(exe, [j("ms_mov.nii.gz"), j("ms_fix.nii.gz"), *flag,
                                "-master", j("ms_master.nii.gz"), "-savemat", mj, one_p])
            rc2, _ = run(exe, [j("ms_mov.nii.gz"), j("ms_fix.nii.gz"), *flag,
                               "-savemat", pj, j(f"ms_plain_{sl}.nii.gz")])
            rc3, _ = run(exe, [j("ms_mov.nii.gz"), j("ms_master.nii.gz"), "-applymat", pj, two_p])
            ok = rc1 == 0 and rc2 == 0 and rc3 == 0 and os.path.exists(one_p)
            check(f"-master {eng}: all exit 0", ok, f"rc={rc1},{rc2},{rc3}: {e1.strip()[-100:]}")
            if ok:
                one = nib.load(one_p)
                check(f"-master {eng}: output on master grid (48^3)", one.shape == (48, 48, 48), f"shape={one.shape}")
                A = np.array(_json.load(open(mj))["fixed_to_moving"])
                B = np.array(_json.load(open(pj))["fixed_to_moving"])
                check(f"-master {eng}: fit unchanged by output grid", bool(np.array_equal(A, B)),
                      f"max|d|={float(np.max(np.abs(A - B))):.2e}")
                oa = np.asarray(one.dataobj, np.float32)
                tw = np.asarray(nib.load(two_p).dataobj, np.float32)
                check(f"-master {eng}: byte-identical to savemat+applymat", bool(np.array_equal(oa, tw)),
                      f"max|d|={float(np.max(np.abs(oa - tw))):.2e}")

        # 11. -cmass/-nocmass BEHAVIORAL divergence (fast engine). Reuses the §7 phantom (cf_fix)
        #     but places the moving header ~42 mm off — beyond the header-start capture range, so
        #     ONLY the COM-translation seed lands in the right basin. Default and -cmass select
        #     the COM start by its initial dependence*overlap score; -nocmass (header-only) fails. An
        #     identity fixture cannot distinguish these, and would not catch an ignored use_cmass
        #     or the NULL-options dereference — this case does.
        print("11. -cmass/-nocmass behavioral divergence (fast engine)")
        Aoff = np.eye(4); Aoff[:3, 3] = [42.0, -25.0, 17.0]   # large oblique offset, COM-recoverable
        save(phan, Aoff @ Sf, j("cm_mov.nii.gz"))
        erms_cm = {}
        for tag, extra in (("default", []), ("cmass", ["-cmass"]), ("nocmass", ["-nocmass"])):
            cj = j(f"cm_{tag}.json")
            rc, _ = run(exe, [j("cm_mov.nii.gz"), j("cf_fix.nii.gz"), "-cost", "fastcr", *extra,
                              "-p", "1", "-savemat", cj, j(f"cm_out_{tag}.nii.gz")])
            erms_cm[tag] = (_erms(np.array(_json.load(open(cj))["fixed_to_moving"]), Aoff)
                            if (rc == 0 and os.path.exists(cj)) else float("inf"))
        check("cmass: default recovers the 42 mm offset (ERMS < 3 mm)", erms_cm["default"] < 3.0,
              f"ERMS={erms_cm['default']:.2f}")
        check("cmass: -cmass recovers via the COM seed (ERMS < 3 mm)", erms_cm["cmass"] < 3.0,
              f"ERMS={erms_cm['cmass']:.2f}")
        check("cmass: -nocmass fails header-only (ERMS > 20 mm)", erms_cm["nocmass"] > 20.0,
              f"ERMS={erms_cm['nocmass']:.2f}")
        check("cmass: default == -cmass (both auto-select the same initialization)",
              abs(erms_cm["default"] - erms_cm["cmass"]) < 1e-6,
              f"default={erms_cm['default']:.4f} cmass={erms_cm['cmass']:.4f}")

        # 12. C-API NULL-options contract: coreg_fast_estimate(mov, fix, NULL, &res) must resolve
        #     to defaults and successfully fit a valid identity pair. The CLI always passes
        #     non-NULL, so this is the only path that catches a raw `opts->` read. `make test`
        #     builds the harness; when absent (standalone run) the check is skipped.
        print("12. C-API NULL-options contract (coreg_fast_estimate)")
        capi = os.path.join(ROOT, "test_capi_nullopts")
        if os.path.exists(capi):
            p = subprocess.run([capi, j("cf_mov_identity.nii.gz"), j("cf_fix.nii.gz")],
                               capture_output=True, text=True, timeout=120)
            check("C-API coreg_fast_estimate(NULL opts) succeeds", p.returncode == 0,
                  f"returncode={p.returncode} (negative = killed by signal): {p.stderr.strip()[-120:]}")
        else:
            print(f"  [SKIP] C-API harness not built ({capi}); run via `make test`")

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
