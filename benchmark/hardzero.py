#!/usr/bin/env python3
"""Benchmark the default fast strategy specifically on hard-zeroed bases.

The ordinary benchmark's whole-head templates cannot expose the coarse rigid-vs-scale
failure described in weighted.md. This script evaluates the committed SSW base plus
brain-only versions of both committed templates. The stripped bases are generated in a
temporary directory from the tracked templates and masks; no duplicate NIfTI data is kept.

For every moving image it reports masked Hellinger affinity. For T1-weighted moving
images it also reports NCC inside the base brain mask, the more sensitive same-modality
judge for the large rotational failures this benchmark targets.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from hellinger import hellinger_quality


def stem(path):
    name = os.path.basename(path)
    return name[:-7] if name.endswith(".nii.gz") else os.path.splitext(name)[0]


def load(path):
    return nib.load(path).get_fdata(dtype=np.float32)


def make_stripped(template_path, mask_path, out_path):
    template = nib.load(template_path)
    data = np.asarray(template.dataobj, dtype=np.float32)
    mask = load(mask_path) > 0.5
    out = nib.Nifti1Image(np.where(mask, data, 0.0).astype(np.float32),
                          template.affine, template.header)
    out.set_data_dtype(np.float32)
    nib.save(out, out_path)
    return out_path, mask_path


def run_case(exe, moving, base, mask_path, threads, timeout, out_path):
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([exe, moving, base, "-p", str(threads), out_path],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return None, proc.stderr.strip()[-200:]

    aligned = load(out_path)
    fixed = load(base)
    mask = load(mask_path) > 0.5
    if aligned.shape != fixed.shape or mask.shape != fixed.shape:
        return None, "grid mismatch"
    hel = hellinger_quality(aligned, fixed, mask)
    ncc = float(np.corrcoef(aligned[mask], fixed[mask])[0, 1])
    return (elapsed, hel, ncc), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allineate", default=os.path.join(REPO, "allineate"))
    parser.add_argument("-p", "--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    inputs_dir = os.path.join(HERE, "inputs")
    templates_dir = os.path.join(HERE, "templates")
    masks_dir = os.path.join(HERE, "masks")
    weighted_dir = os.path.join(HERE, "weighted")
    movings = sorted(os.path.join(inputs_dir, f) for f in os.listdir(inputs_dir)
                     if f.endswith((".nii", ".nii.gz")))
    required = [
        args.allineate,
        os.path.join(weighted_dir, "MNI152_2009_SSW.nii.gz"),
        os.path.join(weighted_dir, "MNI152_2009_SSW_weight.nii.gz"),
        os.path.join(templates_dir, "MNI152_T1_1mm.nii.gz"),
        os.path.join(masks_dir, "MNI152_T1_1mm_brain_mask.nii.gz"),
        os.path.join(templates_dir, "avg152T1.nii.gz"),
        os.path.join(masks_dir, "avg152T1_brain_mask.nii.gz"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise SystemExit("missing required fixture(s): " + ", ".join(missing))

    failures = 0
    with tempfile.TemporaryDirectory(prefix="allineate_hardzero_") as tmp:
        ssw_weight = os.path.join(weighted_dir, "MNI152_2009_SSW_weight.nii.gz")
        ssw_weight_data = load(ssw_weight)
        ssw_mask_path = os.path.join(tmp, "SSW_brain_mask.nii.gz")
        wi = nib.load(ssw_weight)
        nib.save(nib.Nifti1Image(
            (ssw_weight_data > 0.3 * float(ssw_weight_data.max())).astype(np.uint8),
            wi.affine, wi.header), ssw_mask_path)

        cases = [("MNI152_2009_SSW",
                  os.path.join(weighted_dir, "MNI152_2009_SSW.nii.gz"),
                  ssw_mask_path)]
        for template_name, mask_name, label in (
            ("MNI152_T1_1mm.nii.gz", "MNI152_T1_1mm_brain_mask.nii.gz", "MNI152_strip"),
            ("avg152T1.nii.gz", "avg152T1_brain_mask.nii.gz", "avg152_strip"),
        ):
            cases.append((label, *make_stripped(
                os.path.join(templates_dir, template_name),
                os.path.join(masks_dir, mask_name),
                os.path.join(tmp, label + ".nii.gz"))))

        print("| Moving | Base | Time | Masked Hellinger | Masked NCC |")
        print("|---|---|---:|---:|---:|")
        for label, base, mask in cases:
            for moving in movings:
                name = stem(moving)
                out = os.path.join(tmp, f"{name}__to__{label}.nii.gz")
                result, error = run_case(args.allineate, moving, base, mask,
                                         args.threads, args.timeout, out)
                if result is None:
                    failures += 1
                    print(f"| {name} | {label} | FAIL | — | — |")
                    sys.stderr.write(f"{name} -> {label}: {error}\n")
                    continue
                elapsed, hel, ncc = result
                ncc_text = f"{ncc:.4f}" if name.lower().startswith("t1") else "—"
                print(f"| {name} | {label} | {elapsed:.2f} | {hel:.4f} | {ncc_text} |")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
