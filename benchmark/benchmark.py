#!/usr/bin/env python3
"""Benchmark affine registration engines: register every moving image in inputs/ to every
stationary template in templates/ and report speed / memory / match
quality for each engine in its own table (`fast` is the default cost, so the ordinary
AFNI-style engine is selected explicitly with `-cost hel`):

  * allineate  — `allineate <mov> <fix> -cost hel out`  (AFNI-style ordinary engine)
  * fast       — `allineate <mov> <fix> out`            (SPM/FLIRT-inspired path, now the DEFAULT, Hellinger cost)
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
    python3 benchmark.py --engine allineate  # only the ordinary AFNI-style engine (-cost hel)
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
    def run(mov, fix, out, threads, allineate, timeout):
        if os.path.exists(out):
            os.remove(out)
        cmd = (["/usr/bin/time", TIME_FLAG, allineate, mov, fix, out, "-p", str(threads)] + flag)
        return run_timed(cmd, os.environ.copy(), timeout)
    return run


def _run_afni(mov, fix, out, threads, allineate, timeout):
    """AFNI 3dAllineate (reference tool, defaults)."""
    env = dict(os.environ, AFNI_DECONFLICT="OVERWRITE", OMP_NUM_THREADS=str(threads))
    if os.path.exists(out):
        os.remove(out)
    cmd = ["/usr/bin/time", TIME_FLAG, "3dAllineate",
           "-base", fix, "-source", mov, "-prefix", out]
    return run_timed(cmd, env, timeout)


ENGINES = {
    "allineate": _run_allineate(["-cost", "hel"]),
    "fast": _run_allineate(["-cost", "fast"]),
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


def markdown_table(rows):
    """rows: (stationary, moving, t1, ram1, tn, ramn, speedup, cost, cost_masked)."""
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
                    choices=("both", "allineate", "fast", "fast-robust", "afni"),
                    default="both",
                    help="engine(s) to benchmark (default: both = allineate+fast). 'fast-robust' adds "
                         "-robustfov -com to the fast engine; 'afni' selects AFNI 3dAllineate "
                         "standalone (or add it with --afni)")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-registration timeout in seconds (default: 3600)")
    args = ap.parse_args()

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
    if not movings or not stationaries:
        sys.exit("no inputs/ or templates/ images found")

    # engine -> list of table rows
    tables = {e: [] for e in engines}
    n_fail = 0
    for engine in engines:
        sys.stderr.write(f"\n=== engine: {engine} (1 and {N_THREADS} threads) ===\n")
        for st in stationaries:
            mask = os.path.join(masks_dir, f"{stem(st)}_brain_mask.nii.gz")
            for mv in movings:
                out = os.path.join(out_dir, f"{stem(mv)}__to__{stem(st)}__{engine}.nii.gz")
                row = [stem(st), stem(mv)]
                res = {}
                for tag, threads in (("1", 1), ("n", N_THREADS)):
                    sys.stderr.write(f"  {stem(mv):16s} -> {stem(st):16s} [{engine:9s} {tag:>2s}] ... ")
                    sys.stderr.flush()
                    r = ENGINES[engine](mv, st, out, threads, args.allineate, args.timeout)
                    if r is None:
                        sys.stderr.write("FAIL/timeout\n")
                        n_fail += 1
                        res[tag] = (None, None)
                    else:
                        res[tag] = r
                        sys.stderr.write(f"{r[0]:.1f}s, {_fmt(r[1], 0)} MB\n")
                t1, ram1 = res["1"]
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
          f"({len(movings)} moving x {len(stationaries)} stationary; N = {N_THREADS} threads).\n")
    print("**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB "
          "(peak RSS); *Speed Up* = 1-thread Time / N-thread Time; *Cost* / *Cost Masked* = "
          "Hellinger-affinity match quality (higher = better; masked restricts to the template "
          f"brain mask). The `1` columns are single-thread, the `{N_THREADS}` columns use all cores.\n")
    titles = {"allineate": "allineate ordinary engine (`-cost hel`)",
              "fast": "fast (`-cost fast`, the default)",
              "fast-robust": "fast robust (`-robustfov -com -cost fast`)",
              "afni": "AFNI 3dAllineate (reference, defaults)"}
    for engine in engines:
        print(f"### {titles[engine]}\n")
        print(markdown_table(tables[engine]) + "\n")

    if n_fail:
        sys.stderr.write(f"\n{n_fail} registration(s) FAILED/timed out.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
