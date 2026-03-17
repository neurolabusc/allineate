#!/usr/bin/env python3
"""Validation script for allineate: runs all benchmark tests and prints
a markdown table matching the README Validation section.

Usage:
    cd allineate
    OMP_NUM_THREADS=10 python3 validate.py [--exe ./allineate]

Requires: nibabel, numpy
"""

import argparse
import os
import subprocess
import sys
import time

import nibabel as nib
import numpy as np


TESTS = [
    {
        "name": "T1 2mm ls",
        "moving": "T1_head_2mm",
        "stationary": "MNI152_T1_2mm",
        "opts": ["-cost", "ls"],
        "output": "wT1ls_2mm.nii.gz",
    },
    {
        "name": "T1 1mm ls",
        "moving": "T1_head",
        "stationary": "MNI152_T1_1mm",
        "opts": ["-cost", "ls"],
        "output": "wT1ls.nii.gz",
    },
    {
        "name": "T1 1mm hel",
        "moving": "T1_head",
        "stationary": "MNI152_T1_1mm",
        "opts": [],
        "output": "wT1.nii.gz",
    },
    {
        "name": "T1 1mm hel cmass",
        "moving": "T1_head",
        "stationary": "MNI152_T1_1mm",
        "opts": ["-cmass"],
        "output": "wT1cmas.nii.gz",
    },
    {
        "name": "fMRI lpc",
        "moving": "fmri",
        "stationary": "T1_head",
        "opts": ["-cmass", "-cost", "lpc", "-source_automask"],
        "output": "fmri2t1.nii.gz",
    },
]


def ncc(a, b):
    """Normalized cross-correlation between two float arrays (masked to nonzero union)."""
    mask = (a != 0) | (b != 0)
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    if len(a) == 0:
        return 0.0
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)



def run_test_with_ram(exe, examples_dir, test, outdir):
    """Run test using /usr/bin/time to capture peak RAM."""
    moving = os.path.join(examples_dir, test["moving"])
    stationary = os.path.join(examples_dir, test["stationary"])
    outfile = os.path.join(outdir, test["output"])
    cmd = [exe, moving, stationary] + test["opts"] + [outfile]
    cmd_display = " ".join(
        [os.path.basename(exe), test["moving"], test["stationary"]]
        + test["opts"]
    )

    print(f"  Running: {cmd_display} ...", end="", flush=True)

    # Use /usr/bin/time for peak RAM
    if sys.platform == "darwin":
        time_cmd = ["/usr/bin/time", "-l"] + cmd
    else:
        time_cmd = ["/usr/bin/time", "-v"] + cmd

    t0 = time.monotonic()
    result = subprocess.run(time_cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f" FAILED (rc={result.returncode})")
        print(result.stderr)
        return None

    # Parse peak RAM from /usr/bin/time output (in stderr)
    peak_mb = None
    for line in result.stderr.splitlines():
        line = line.strip()
        if sys.platform == "darwin":
            # macOS: "  NNN  maximum resident set size" (in bytes)
            if "maximum resident set size" in line:
                parts = line.split()
                try:
                    peak_mb = int(parts[0]) / (1024 * 1024)
                except (ValueError, IndexError):
                    pass
        else:
            # Linux: "Maximum resident set size (kbytes): NNN"
            if "Maximum resident set size" in line:
                parts = line.split(":")
                try:
                    peak_mb = int(parts[-1].strip()) / 1024
                except (ValueError, IndexError):
                    pass

    # Load output and compute NCC against stationary image
    out_img = nib.load(outfile).get_fdata(dtype=np.float32).ravel()

    stat_path = stationary
    for ext in [".nii.gz", ".nii"]:
        p = stat_path + ext
        if os.path.exists(p):
            stat_path = p
            break
    stat_img = nib.load(stat_path).get_fdata(dtype=np.float32).ravel()
    ncc_stat = ncc(out_img, stat_img) if out_img.shape == stat_img.shape else float("nan")

    ram_str = f"{peak_mb:.0f} MB" if peak_mb else "?"
    print(f" {elapsed:.1f}s, {ram_str}, NCC(stat)={ncc_stat:.3f}")

    return {
        "elapsed": elapsed,
        "peak_mb": peak_mb,
        "ncc_stat": ncc_stat,
        "cmd_display": cmd_display,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate allineate benchmarks")
    parser.add_argument("--exe", default="./allineate", help="Path to allineate binary")
    parser.add_argument("--examples", default="examples", help="Path to examples directory")
    args = parser.parse_args()

    exe = os.path.abspath(args.exe)
    examples_dir = os.path.abspath(args.examples)

    if not os.path.isfile(exe):
        print(f"Error: executable not found: {exe}")
        sys.exit(1)
    if not os.path.isdir(examples_dir):
        print(f"Error: examples directory not found: {examples_dir}")
        sys.exit(1)

    # Show version
    ver = subprocess.run([exe], capture_output=True, text=True)
    version_line = ver.stdout.splitlines()[0] if ver.stdout else ver.stderr.splitlines()[0]
    print(f"Validating: {version_line}")
    threads = os.environ.get("OMP_NUM_THREADS", "?")
    print(f"OMP_NUM_THREADS={threads}\n")

    outdir = os.path.join(examples_dir, "out")
    os.makedirs(outdir, exist_ok=True)

    results = []
    for test in TESTS:
        r = run_test_with_ram(exe, examples_dir, test, outdir)
        results.append((test, r))

    # Print markdown table
    print()
    print("| Test | Command | Time | Peak RAM | NCC (stationary) |")
    print("|------|---------|------|----------|-------------------|")
    for test, r in results:
        if r is None:
            print(f"| {test['name']} | `{test['name']}` | FAILED | — | — |")
            continue
        time_str = f"{r['elapsed']:.1f}s"
        ram_str = f"{r['peak_mb']:.0f} MB" if r['peak_mb'] else "?"
        print(f"| {test['name']} | `{r['cmd_display']}` | {time_str} | {ram_str} | {r['ncc_stat']:.3f} |")

    # Check for regressions
    print()
    all_ok = True
    for test, r in results:
        if r is None:
            print(f"FAIL: {test['name']} did not complete")
            all_ok = False
        elif r["ncc_stat"] < 0.4:
            print(f"WARN: {test['name']} NCC(stat) = {r['ncc_stat']:.4f} < 0.4")
            all_ok = False
    if all_ok:
        print("All tests passed")


if __name__ == "__main__":
    main()
