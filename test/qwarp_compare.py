#!/usr/bin/env python3
"""Developer-only equivalence harness for the -qwarp port (NOT part of `make test`).

Compares a candidate warped image against AFNI's 3dQwarp reference output and against the
stationary/base, so a mutually-wrong-but-correlated result cannot pass. Reads the gitignored
qwarp/ developer validation data; exits 0 with a SKIP notice if that data is absent, so this
script is safe to keep tracked without breaking a clean clone.

Reported metrics (per pair):
  * voxelwise Pearson correlation (candidate vs reference)
  * RMSE normalized by the AFNI reference dynamic range
  * absolute-error percentiles (50/90/99/max)
  * geometry equality (dims + affine within tolerance)
  * sanity: Pearson(candidate, base) and Pearson(reference, base) — the candidate must track
    the REFERENCE more tightly than either tracks the base, else it is only trivially aligned.

Usage:
  python3 test/qwarp_compare.py                       # all three bundled reference cases
  python3 test/qwarp_compare.py CAND REF [BASE]       # a single explicit comparison
  python3 test/qwarp_compare.py --self                # gate: compare each reference to ITSELF
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
QDIR = os.path.join(REPO, "qwarp")
BASE = os.path.join(QDIR, "MNI152_T1_1mm_brain.nii.gz")
# (candidate-producing input, AFNI reference output) stems
CASES = [
    ("usaT1_chris", "usaqT1_chris"),
    ("usaT1_ds000031", "usaqT1_ds000031"),
    ("usaT1w_MICCAI2017", "usaqT1w_MICCAI2017"),
]


def _load(path):
    import numpy as np, nibabel as nib
    im = nib.load(path)
    return np.asarray(im.dataobj, dtype=np.float64), im.affine.astype(np.float64), im.shape


def _pearson(a, b):
    import numpy as np
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    return float((a * b).sum() / (da * db)) if da > 0 and db > 0 else float("nan")


# Recorded acceptance thresholds (set after the first faithful serial run).
# Enforced so this harness is a real gate, not just a report.
THRESH_PEARSON_CAND_REF = 0.97    # candidate must track the AFNI reference this tightly
THRESH_BASE_RATIO       = 0.98    # cand-vs-base must be >= this * (AFNI-ref-vs-base)
THRESH_NORM_RMSE        = 0.08    # normalized RMSE ceiling (observed: chris 0.035 / ds 0.022 / MICCAI 0.042)


def compare(cand_path, ref_path, base_path=None, require_base=True):
    import numpy as np
    cand, caff, cshape = _load(cand_path)
    ref, raff, rshape = _load(ref_path)
    geom_ok = (cshape == rshape) and np.allclose(caff, raff, atol=1e-3)
    if cand.shape != ref.shape:
        print(f"  DIM MISMATCH cand{cand.shape} vs ref{ref.shape}")
        return False
    diff = np.abs(cand - ref)
    rng = float(ref.max() - ref.min()) or 1.0
    rmse = float(np.sqrt((diff * diff).mean())); nrmse = rmse / rng
    pct = np.percentile(diff, [50, 90, 99, 100])
    r = _pearson(cand, ref)
    print(f"  Pearson(cand,ref) = {r:.6f}")
    print(f"  RMSE = {rmse:.4f}  (norm by ref range {rng:.1f} = {nrmse:.5f})")
    print(f"  |err| pct 50/90/99/max = {pct[0]:.3f} / {pct[1]:.3f} / {pct[2]:.3f} / {pct[3]:.3f}")
    print(f"  geometry (dims+affine) equal: {geom_ok}")
    ok = geom_ok
    if not (r >= THRESH_PEARSON_CAND_REF):
        print(f"  [FAIL] Pearson(cand,ref) {r:.4f} < {THRESH_PEARSON_CAND_REF}"); ok = False
    if not (nrmse <= THRESH_NORM_RMSE):
        print(f"  [FAIL] normalized RMSE {nrmse:.4f} > {THRESH_NORM_RMSE}"); ok = False
    # base comparison is REQUIRED here (a missing/mismatched base must fail, not silently skip half
    # the gate) — it is what rejects a trivial stationary-image candidate.
    if base_path and os.path.exists(base_path):
        base, _, _ = _load(base_path)
        if base.shape != ref.shape:
            print(f"  [FAIL] base shape {base.shape} != ref {ref.shape}"); ok = False
        else:
            cb, rb = _pearson(cand, base), _pearson(ref, base)
            print(f"  sanity: Pearson(cand,base)={cb:.6f}  Pearson(ref,base)={rb:.6f}")
            # anti-trivial contract: a real warp makes cand ~ ref (both aligned to base), so
            # Pearson(cand,ref) must exceed BOTH correlations-to-base; a stationary/identity
            # candidate has cand-vs-base ~ 1 > cand-vs-ref and fails here.
            if not (r > cb and r > rb):
                print(f"  [FAIL] Pearson(cand,ref) {r:.4f} does not exceed both base correlations "
                      f"(cand,base={cb:.4f} ref,base={rb:.4f})"); ok = False
            if not (cb >= THRESH_BASE_RATIO * rb):
                print(f"  [FAIL] cand-vs-base {cb:.4f} < {THRESH_BASE_RATIO}*ref-vs-base ({THRESH_BASE_RATIO*rb:.4f})"); ok = False
    elif require_base:
        print(f"  [FAIL] base image required for the anti-trivial gate but missing: {base_path}"); ok = False
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    args = [a for a in sys.argv[1:]]
    if not os.path.isdir(QDIR):
        print(f"SKIP: developer validation data {QDIR} not present.")
        return 0
    if args and args[0] == "--self":
        # Gate: comparing each reference to itself must give Pearson 1, RMSE 0.
        ok = True; compared = 0
        for _, ref in CASES:
            p = os.path.join(QDIR, ref + ".nii.gz")
            if not os.path.exists(p):
                print(f"SKIP {ref}: missing"); continue
            print(f"[self] {ref}")
            ok = compare(p, p, BASE) and ok; compared += 1
        if compared < len(CASES):
            print(f"[FAIL] --self compared {compared}/{len(CASES)} declared references"); ok = False
        return 0 if ok else 1
    if len(args) >= 2:   # explicit CAND REF [BASE]: exit nonzero if thresholds not met
        return 0 if compare(args[0], args[1], args[2] if len(args) > 2 else BASE) else 1
    # default: the advertised THREE-case gate. Since QDIR is present (checked above), ALL three
    # declared reference cases must be compared and pass — a missing reference or an un-produced
    # candidate is a failure, not a silent skip (otherwise an incomplete qwarp/ could "pass" with
    # zero comparisons). Use an explicit CAND REF invocation for ad-hoc single comparisons.
    ok = True; compared = 0
    for src, ref in CASES:
        refp = os.path.join(QDIR, ref + ".nii.gz")
        candp = os.path.join(QDIR, src + "__qwarp.nii.gz")  # produced by the dev run
        if not os.path.exists(refp):
            print(f"[FAIL] {ref}: reference missing (the default gate requires all {len(CASES)} cases)"); ok = False; continue
        if not os.path.exists(candp):
            print(f"PENDING {src}: no candidate at {os.path.basename(candp)} — run -qwarp first"); ok = False; continue
        print(f"[case] {src} -> {ref}")
        ok = compare(candp, refp, BASE) and ok; compared += 1
    if compared < len(CASES):
        print(f"\n[FAIL] compared {compared}/{len(CASES)} declared cases"); ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
