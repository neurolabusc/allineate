#!/usr/bin/env python3
"""Benchmark affine registration engines: register every moving image in inputs/ to every
stationary template in templates/ and report speed / memory / match
quality for each engine in its own table (`fast` is the default cost, so the ordinary
AFNI-style engine is selected explicitly with `-cost hel`):

  * allineate  — `allineate <mov> <fix> -cost hel out`  (AFNI-style ordinary engine)
  * fast       — `allineate <mov> <fix> out`            (DEFAULT: HEL/CR coarse competition)
  * fasthel    — `allineate <mov> <fix> -cost fasthel out` (HEL-only fast path)
  * fastx      — `allineate <mov> <fix> -cost fastx out` (explicit alias of the default)
  * fast-robust — `allineate <mov> <fix> -robustfov -com -cost fast out`
  * AFNI       — `3dAllineate -base <fix> -source <mov> -prefix out` (reference; only with --afni)

Each engine runs every pair at 1 thread and at all cores (N = os.cpu_count()), so a single table
carries both the single- and multi-thread cost plus the resulting speed-up. Match quality is a
post-hoc Hellinger-affinity metric (see benchmark/hellinger.py): the Hellinger distance of the
joint 2D intensity histogram from independence, which measures statistical DEPENDENCE between
corresponding voxels (higher = better). Unlike NCC it assumes no linear intensity relationship,
so it is valid for cross-modal pairs (T2/fMRI -> T1). It is independent of each engine's own
registration cost, so it compares engines fairly.

Outputs go to outputs/ (untracked; they overwrite freely). Markdown tables are printed to stdout
(copy-paste into README.md); progress goes to stderr.

Usage:
    python3 benchmark.py                 # allineate + fast engines
    python3 benchmark.py --multi-only    # initial all-core regression sweep
    python3 benchmark.py --engine allineate  # only the ordinary AFNI-style engine (-cost hel)
    python3 benchmark.py --engine fasthel    # HEL-only fast path
    python3 benchmark.py --engine fastx      # explicit mixed-default selector
    python3 benchmark.py --engine fast-robust  # robust preprocessing + fast engine
    python3 benchmark.py --afni          # also benchmark AFNI 3dAllineate (must be on PATH)
    python3 benchmark.py --allineate ../allineate
"""
import argparse
import os
import signal
import shutil
import subprocess
import sys
import time

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

from hellinger import hellinger_quality

# "N" (multi-thread) column: all logical cores, matching allineate's -p 0 / omp_get_max_threads.
N_THREADS = os.cpu_count() or 1
TIME_FLAG = "-l" if sys.platform == "darwin" else "-v"


def scores(out_path, stat_path, mask_path):
    """Return (hellinger_full, hellinger_masked). Output is resliced onto the stationary grid,
    so output, stationary, and mask share dimensions. The quality metric is the Hellinger
    distance of the joint 2D intensity histogram from independence (see benchmark/hellinger.py)
    — statistical DEPENDENCE between corresponding voxels (higher = better), valid cross-modal.
    full = union of non-zero voxels; masked = restricted to the template brain mask (> 0.5)."""
    a = nib.load(out_path).get_fdata(dtype=np.float32)
    b = nib.load(stat_path).get_fdata(dtype=np.float32)
    if a.shape != b.shape:
        return float("nan"), float("nan")
    hf = hellinger_quality(a, b, (a > 0) | (b > 0))
    hm = float("nan")
    if mask_path and os.path.exists(mask_path):
        m = nib.load(mask_path).get_fdata(dtype=np.float32)
        if m.shape == a.shape:
            hm = hellinger_quality(a, b, m > 0.5)
    return hf, hm


def parse_peak_rss_mb(time_stderr):
    """Peak RSS in MB from `/usr/bin/time -l` (macOS: bytes) or `-v` (Linux: kbytes)."""
    for line in time_stderr.splitlines():
        s = line.strip()
        if sys.platform == "darwin":
            if "maximum resident set size" in s:
                try:
                    return int(s.split()[0]) / (1024 * 1024)
                except (ValueError, IndexError):
                    return None
        else:
            if "Maximum resident set size" in s:
                try:
                    return int(s.split(":")[-1].strip()) / 1024
                except (ValueError, IndexError):
                    return None
    return None


def run_timed(cmd, env, timeout):
    """Run a command under /usr/bin/time; return (secs, peak_mb) or None on failure/timeout."""
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    try:
        _, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        return None
    secs = time.perf_counter() - t0
    if p.returncode != 0:
        sys.stderr.write(stderr[-500:] + "\n")
        return None
    return secs, parse_peak_rss_mb(stderr)


# Engine registry: name -> callable(moving, stationary, out, threads, allineate, timeout) that
# runs one registration under /usr/bin/time and returns (secs, peak_mb) or None.
def _run_allineate(flag):
    def run(mov, fix, out, threads, allineate, timeout, weight=None):
        if os.path.exists(out):
            os.remove(out)
        cmd = (["/usr/bin/time", TIME_FLAG, allineate, mov, fix, out, "-p", str(threads)] + flag)
        if weight:
            cmd += ["-weight", weight]
        return run_timed(cmd, os.environ.copy(), timeout)
    return run


def _run_afni(mov, fix, out, threads, allineate, timeout, weight=None):
    """AFNI 3dAllineate (reference tool, defaults)."""
    env = dict(os.environ, AFNI_DECONFLICT="OVERWRITE", OMP_NUM_THREADS=str(threads))
    if os.path.exists(out):
        os.remove(out)
    cmd = ["/usr/bin/time", TIME_FLAG, "3dAllineate",
           "-base", fix, "-source", mov, "-prefix", out]
    if weight:
        cmd += ["-weight", weight]
    return run_timed(cmd, env, timeout)


ENGINES = {
    "allineate": _run_allineate(["-cost", "hel"]),
    "fast": _run_allineate(["-cost", "fast"]),
    "fasthel": _run_allineate(["-cost", "fasthel"]),
    "fastx": _run_allineate(["-cost", "fastx"]),
    "fast-robust": _run_allineate(["-robustfov", "-com", "-cost", "fast"]),
    "afni": _run_afni,
}


def stem(path):
    n = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if n.endswith(ext):
            return n[: -len(ext)]
    return n


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n/a"


def markdown_table(rows, multi_only=False):
    """rows: (stationary, moving, t1, ram1, tn, ramn, speedup, cost, cost_masked)."""
    if multi_only:
        hdr = ["Stationary", "Moving", f"{N_THREADS} Time", f"{N_THREADS} Peak RAM",
               "Cost", "Cost Masked"]
        lines = ["| " + " | ".join(hdr) + " |",
                 "|" + "|".join(["---"] * len(hdr)) + "|"]
        for st, mv, _t1, _ram1, tn, ramn, _sp, cost, cm in rows:
            def s(x, nd=1):
                return f"{x:.{nd}f}" if x is not None and not (
                    isinstance(x, float) and np.isnan(x)) else "FAIL"
            lines.append(
                f"| {st} | {mv} | {s(tn)} | {s(ramn, 0)} | "
                f"{_fmt(cost)} | {_fmt(cm)} |")
        return "\n".join(lines)

    hdr = ["Stationary", "Moving", "1 Time", "1 Peak RAM",
           f"{N_THREADS} Time", f"{N_THREADS} Peak RAM", "Speed Up", "Cost", "Cost Masked"]
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    for st, mv, t1, ram1, tn, ramn, sp, cost, cm in rows:
        def s(x, nd=1):
            return f"{x:.{nd}f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "FAIL"
        lines.append(
            f"| {st} | {mv} | {s(t1)} | {s(ram1, 0)} | {s(tn)} | {s(ramn, 0)} | "
            f"{s(sp, 1)}x | {_fmt(cost)} | {_fmt(cm)} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allineate", default=os.path.join(REPO, "allineate"),
                    help="path to the allineate binary (default: repo root)")
    ap.add_argument("--afni", action="store_true",
                    help="also benchmark AFNI 3dAllineate (must be on PATH)")
    ap.add_argument("--engine",
                    choices=("both", "allineate", "fast", "fasthel", "fastx", "fast-robust", "afni"),
                    default="both",
                    help="engine(s) to benchmark (default: both = allineate+fast, where 'fast' "
                         "is the mixed-default selector). 'fasthel' "
                         "forces HEL-only; 'fastx' explicitly selects the mixed default; "
                         "'fast-robust' adds -robustfov -com to the fast engine; "
                         "'afni' selects AFNI 3dAllineate standalone "
                         "(or add it with --afni)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-registration timeout in seconds (default: 3600)")
    ap.add_argument("--multi-only", action="store_true",
                    help="run only the all-core pass and print compact regression tables; "
                         "omit the 1-thread pass and speed-up columns")
    ap.add_argument("--weight", metavar="WEIGHTIMG",
                    help="weighted-registration mode: pass `-weight WEIGHTIMG` to every engine "
                         "(AFNI 3dAllineate-style graded weight), register only to --base, and score "
                         "the masked cost inside the weight's brain plateau. Requires --base.")
    ap.add_argument("--base", metavar="BASEIMG",
                    help="single stationary/base image for --weight mode (e.g. the SSW1 template). "
                         "The WEIGHTIMG must already share this base's grid.")
    ap.add_argument("--mask", metavar="MASKIMG",
                    help="override the masked-cost ROI (default in --weight mode: WEIGHTIMG > 0.3*max).")
    args = ap.parse_args()

    if args.weight and not args.base:
        sys.exit("--weight requires --base (the single stationary the weight is defined on)")

    if not os.access(args.allineate, os.X_OK):
        sys.exit(f"allineate binary not found/executable at '{args.allineate}' — run `make` in {REPO}")

    engines = (["allineate", "fast"] if args.engine == "both" else [args.engine])
    if args.afni and "afni" not in engines:
        engines.append("afni")
    if "afni" in engines and shutil.which("3dAllineate") is None:
        sys.exit("an AFNI engine was requested but 3dAllineate is not on PATH")

    inputs_dir = os.path.join(HERE, "inputs")
    templates_dir = os.path.join(HERE, "templates")
    masks_dir = os.path.join(HERE, "masks")
    out_dir = os.path.join(HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    movings = sorted(os.path.join(inputs_dir, f) for f in os.listdir(inputs_dir)
                     if f.endswith((".nii", ".nii.gz")))
    stationaries = sorted(os.path.join(templates_dir, f) for f in os.listdir(templates_dir)
                          if f.endswith((".nii", ".nii.gz")))

    def plateau_mask(weight_path, tag):
        """Brain ROI = the graded weight's high plateau (> 0.3*max): the brain/deep-tissue
        core, so the masked cost excludes the attenuated scalp/background."""
        w = nib.load(weight_path).get_fdata(dtype=np.float32)
        mp = os.path.join(out_dir, f"_plateau_{tag}.nii.gz")
        nib.save(nib.Nifti1Image((w > 0.3 * float(w.max())).astype(np.uint8),
                                 nib.load(weight_path).affine), mp)
        return mp

    # A "case" is (stationary_path, weight_path_or_None, mask_path, label). The benchmark scores
    # each moving against every case; a weighted case additionally passes `-weight` to every engine
    # and scores the masked cost inside the weight plateau.
    cases = []
    if args.weight:
        # Explicit single-base weighted mode (--weight/--base): unchanged behavior.
        wm = args.mask if args.mask else plateau_mask(args.weight, stem(args.base))
        cases.append((args.base, args.weight, wm, stem(args.base) + " +w"))
    else:
        for st in stationaries:
            cases.append((st, None, os.path.join(masks_dir, f"{stem(st)}_brain_mask.nii.gz"), stem(st)))
        # Weighted templates are a DEFAULT part of the benchmark: weighted/<T>.nii.gz paired with
        # weighted/<T>_weight.nii.gz. Each input is registered to <T> with `-weight <T>_weight`,
        # and the masked cost is scored inside the weight plateau.
        wdir = os.path.join(HERE, "weighted")
        if os.path.isdir(wdir):
            for wt in sorted(f for f in os.listdir(wdir) if f.endswith((".nii", ".nii.gz"))
                             and not stem(f).endswith("_weight")):
                wpath = os.path.join(wdir, wt)
                weight = os.path.join(wdir, f"{stem(wt)}_weight.nii.gz")
                if not os.path.exists(weight):
                    sys.stderr.write(f"  (skip weighted template {stem(wt)}: no {stem(wt)}_weight image)\n")
                    continue
                cases.append((wpath, weight, plateau_mask(weight, stem(wt)), stem(wt) + " +w"))

    if not movings or not cases:
        sys.exit("no inputs/ or templates/ (or weighted/) images found")

    # engine -> list of table rows
    tables = {e: [] for e in engines}
    n_fail = 0
    for engine in engines:
        mode_label = f"{N_THREADS} threads only" if args.multi_only else f"1 and {N_THREADS} threads"
        sys.stderr.write(f"\n=== engine: {engine} ({mode_label}) ===\n")
        for st, wgt, mask, label in cases:
            for mv in movings:
                wtag = "__weight" if wgt else ""
                out = os.path.join(out_dir, f"{stem(mv)}__to__{stem(st)}__{engine}{wtag}.nii.gz")
                row = [label, stem(mv)]
                res = {}
                runs = (("n", N_THREADS),) if args.multi_only else (
                    ("1", 1), ("n", N_THREADS))
                for tag, threads in runs:
                    sys.stderr.write(f"  {stem(mv):16s} -> {label:18s} [{engine:9s} {tag:>2s}] ... ")
                    sys.stderr.flush()
                    r = ENGINES[engine](mv, st, out, threads, args.allineate, args.timeout,
                                        weight=wgt)
                    if r is None:
                        sys.stderr.write("FAIL/timeout\n")
                        n_fail += 1
                        res[tag] = (None, None)
                    else:
                        res[tag] = r
                        sys.stderr.write(f"{r[0]:.1f}s, {_fmt(r[1], 0)} MB\n")
                t1, ram1 = res.get("1", (None, None))
                tn, ramn = res["n"]
                speedup = (t1 / tn) if (t1 and tn) else None
                # Quality scored on the N-thread output (deterministic for allineate/fast).
                # Every engine pre-removes `out` before each run, but also require the N-thread run
                # to have succeeded (tn is not None) so a SIGKILLed/failed run that left a partial
                # file is never scored — its Cost stays blank next to the FAIL time.
                hel, mhel = (scores(out, st, mask)
                             if (tn is not None and os.path.exists(out)) else (None, None))
                tables[engine].append((*row, t1, ram1, tn, ramn, speedup, hel, mhel))

    # Copy-pasteable markdown to stdout.
    print(f"Benchmark on {os.uname().sysname} {os.uname().machine} "
          f"({len(movings)} moving x {len(cases)} base cases; N = {N_THREADS} threads).\n")
    if args.multi_only:
        print("**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB "
              "(peak RSS); *Cost* / *Cost Masked* = Hellinger-affinity match quality "
              "(higher = better; masked restricts to the template brain mask). "
              f"All registrations use {N_THREADS} threads.\n")
    else:
        print("**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB "
              "(peak RSS); *Speed Up* = 1-thread Time / N-thread Time; *Cost* / *Cost Masked* = "
              "Hellinger-affinity match quality (higher = better; masked restricts to the template "
              f"brain mask). The `1` columns are single-thread, the `{N_THREADS}` columns use all cores.\n")
    wsuf = " + `-weight`" if args.weight else ""
    titles = {"allineate": "allineate ordinary engine (`-cost hel`)" + wsuf,
              "fast": "fast mixed coarse (`-cost fast`, the default)" + wsuf,
              "fasthel": "fast HEL-only (`-cost fasthel`)" + wsuf,
              "fastx": "fast mixed coarse (`-cost fastx`, explicit default)" + wsuf,
              "fast-robust": "fast robust (`-robustfov -com -cost fast`)" + wsuf,
              "afni": "AFNI 3dAllineate (reference, defaults)" + wsuf}
    for engine in engines:
        print(f"### {titles[engine]}\n")
        print(markdown_table(tables[engine], multi_only=args.multi_only) + "\n")

    if n_fail:
        sys.stderr.write(f"\n{n_fail} registration(s) FAILED/timed out.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
