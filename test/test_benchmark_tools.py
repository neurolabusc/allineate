#!/usr/bin/env python3
"""Focused checks for hard-zero benchmark self-validation and NCC handling."""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmark"))

import hardzero


def check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    if not condition:
        raise SystemExit(1)


def main():
    hard = np.zeros((4, 4, 4), dtype=np.float32)
    hard.reshape(-1)[:8] = 1.0
    check("exact-minimum fraction matches the engine gate",
          hardzero.hardzero_fraction(hard) == 0.875)
    check("valid hard-zero base is accepted",
          hardzero.require_hardzero(hard, "valid") == 0.875)

    negative = hard.copy()
    negative[0, 0, 0] = -1.0
    try:
        hardzero.require_hardzero(negative, "negative template")
    except ValueError:
        rejected = True
    else:
        rejected = False
    check("negative-minimum template that misses the gate is rejected", rejected)

    mask = np.ones((2, 2, 2), dtype=bool)
    ramp = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    check("well-defined NCC is retained",
          abs(hardzero.masked_ncc(ramp, ramp, mask) - 1.0) < 1e-12)
    check("constant aligned data yields a clean unavailable NCC",
          hardzero.masked_ncc(np.ones_like(ramp), ramp, mask) is None)
    check("non-finite NCC input is rejected",
          hardzero.masked_ncc(np.full_like(ramp, np.nan), ramp, mask) is None)
    check("unavailable same-modal NCC is a reported failure",
          hardzero.format_ncc(None, True) == ("FAIL", True))
    check("cross-modal NCC is explicitly not requested",
          hardzero.format_ncc(None, False) == ("—", False))

    print("benchmark helper tests passed")


if __name__ == "__main__":
    main()
