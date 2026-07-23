# allineate benchmark

A small, repeatable benchmark of registration quality/speed/memory on a fixed set of images. Mixed fast capture is the default, so the ordinary AFNI-style engine must be selected explicitly with `-cost hel`:

- **allineate** — `allineate <mov> <fix> -cost hel out` (AFNI-style ordinary engine, 12-DOF affine, Hellinger cost)
- **fast / fastx** — `allineate <mov> <fix> out` or `-cost fast[x]` (the **default**: mixed HEL/CR whole-head capture; full-depth multi-start for a hard-zeroed base)
- **fasthel**   — `allineate <mov> <fix> -cost fasthel out` (HEL-only fast path)
- **fast robust** — `allineate <mov> <fix> -robustfov -com -cost fast out` (crop and recenter before the fast fit)
- **AFNI**      — `3dAllineate -base <fix> -source <mov> -prefix out` (reference tool, defaults; only with `--afni`)
- **FLIRT**     — `flirt -in <mov> -ref <fix> -out out` (FSL reference tool, 12-DOF affine; optimizes FLIRT's default correlation-ratio internally, single-threaded)

## Data provenance & licensing

All bundled imaging data is cleared for redistribution.

- **Templates & masks** — `avg152T1`, `MNI152_T1_1mm`, and the brain masks are the standard FSL/MNI152 average atlases (non-subject, publicly redistributable under the FSL/MNI terms).
- **Subject moving images** — `T1w1mm`, `T1w2mm`, `T2w`, `fmri` are de-identified and cleared by the data owner for sharing. `FLAIR_MICCAI2017` and `T1w_MICCAI2017` are the two local MICCAI benchmark additions used in the expanded results below.

The `make test` correctness gate generates most fixtures on the fly. Its two fast-HEL accuracy cases derive deterministic known-transform fixtures from the bundled non-subject `avg152T1` atlas and mask; it does not gate on the subject scans or their descriptive benchmark scores.

## Inputs

| role | images |
|---|---|
| moving | `FLAIR_MICCAI2017` (1.30×1.21×3 mm FLAIR, **cross-modal**), `T1w1mm` (0.9 mm T1), `T1w2mm` (2 mm T1), `T1w_MICCAI2017` (0.98×0.98×1.5 mm T1), `T2w` (1×1×2 mm T2, **cross-modal**), `fmri` (2.5×2.5×2 mm EPI, **cross-modal**) |
| stationary | `avg152T1` (2 mm), `MNI152_T1_1mm` (1 mm) |
| weighted (default) | `weighted/<T>.nii.gz` paired with `weighted/<T>_weight.nii.gz` — each input is also registered to `<T>` with `-weight <T>_weight` and scored inside the weight plateau (currently `MNI152_2009_SSW`) |

## Running it

```bash
make                          # build the release binary first
cd benchmark
python3 benchmark.py          # allineate + fast engines (needs numpy + nibabel)
python3 benchmark.py --engine allineate  # ordinary AFNI-style engine (-cost hel) only
python3 benchmark.py --engine fasthel    # HEL-only fast path
python3 benchmark.py --engine fastx      # explicit mixed-default selector
python3 benchmark.py --engine fast-robust  # robust preprocessing + fast engine only
python3 benchmark.py --afni   # also benchmark AFNI 3dAllineate (must be on PATH)
```

Every plain template AND every `weighted/` template pair is benchmarked by default (no flag). A `weighted/<T>.nii.gz` template must ship a `weighted/<T>_weight.nii.gz` sibling; each input is registered to it with the AFNI-style graded `-weight`, and the masked cost is scored inside the weight plateau (`weight > 0.3·max`). All benchmark data lives under `benchmark/` (`inputs/`, `templates/`, `masks/`, `weighted/`); the benchmark reads nothing outside it.

The `fast` engine's hard-zeroed-base robustness (e.g. an `@SSwarper` SSW template) combines two mechanisms gated by the same full-resolution base detector (`>25%` of voxels exactly at the minimum). First, the HEL cost's joint-foreground moving-dark skip removes the moving-bin-zero independence floor. Second, the search runs rigid HEL, bounded scale-bracketed HEL, and CR-seeded HEL as independent strategies through the 2 mm 7-DOF stage, then applies 9/12-DOF polish only to the winner. This is necessary because a skull-stripped base can make the correct orientation score worse when scale is locked at 8 mm; pruning at 4 mm was empirically unsafe. Whole-head bases retain the faster mixed strategy byte-for-byte.

The default mixed strategy is the capture hedge for a hard-zeroed MOVING image against a whole-head base, where the base gate correctly leaves the HEL histogram untouched but the coarse HEL landscape can still be shallow. At 8 mm it fits HEL×affine, HEL×COM, CR×affine, and CR×COM as independent candidates; OpenMP runs up to four optimizers concurrently. It re-scores the candidates with HEL dependence×overlap and runs the 4/2 mm HEL stages once. Use `fasthel` or `fastcr` to force a single-cost path.

For the dedicated hard-zeroed matrix, run:

```bash
python3 hardzero.py              # default binary, all available threads
python3 hardzero.py -p 1         # deterministic serial comparison
```

The script uses the committed SSW base and generates brain-only MNI152/avg152 bases in a temporary directory from the committed templates and masks. It reports masked Hellinger for every modality and masked NCC for T1 inputs. No stripped-image duplicates are stored. The four decisive same-modality checks currently reproduce the imported results: ARC→SSW 0.3357, T1w1mm→MNI-strip 0.4770, ARC→avg-strip 0.4972, and MICCAI→avg-strip 0.0994 NCC. See [weighted.md](weighted.md) for the failure analysis and before/after summary.

Each engine runs every (stationary, moving) pair at **1 thread** and at **all cores** (N = `os.cpu_count()`), so one table per engine carries both the single- and multi-thread cost plus the speed-up. Match quality is a post-hoc **Hellinger-affinity** metric (`benchmark/hellinger.py`): the Hellinger distance of the joint 2D intensity histogram from independence — statistical DEPENDENCE between corresponding voxels (higher = better). Unlike NCC it assumes no linear intensity relationship, so it is valid for cross-modal pairs (T2/fMRI → T1) and comparable across engines (it is independent of each engine's own registration cost).

## Results (Apple M4 Max, Apple Clang + libomp, release build)

**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB (peak RSS); *Speed Up* = 1-thread Time / N-thread Time; *Cost* / *Cost Masked* = Hellinger-affinity match quality (higher = better; masked restricts to the template brain mask). N = 14 threads. The cross-modal `FLAIR_MICCAI2017`/`T2w`/`fmri` rows are the especially interesting ones for quality.

### allineate fast HEL-only (`-cost fasthel`) is inspired by SPM/FLIRT

This historical table is the HEL-only path now selected with `-cost fasthel`. It is ~1–3 s end-to-end at every resolution. The mixed default is compared directly below.

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 2.5 | 159 | 1.2 | 160 | 2.0x | 0.4344 | 0.2209 |
| MNI152_T1_1mm | T1w1mm | 2.8 | 297 | 1.4 | 297 | 2.1x | 0.4795 | 0.2074 |
| MNI152_T1_1mm | T1w2mm | 2.4 | 135 | 1.1 | 136 | 2.2x | 0.4746 | 0.2082 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 2.6 | 289 | 1.3 | 290 | 2.0x | 0.4394 | 0.1751 |
| MNI152_T1_1mm | T2w | 2.1 | 157 | 1.0 | 157 | 2.1x | 0.4182 | 0.1777 |
| MNI152_T1_1mm | fmri | 1.8 | 122 | 0.8 | 123 | 2.2x | 0.4035 | 0.2119 |
| avg152T1 | FLAIR_MICCAI2017 | 2.0 | 71 | 0.5 | 71 | 3.7x | 0.4793 | 0.2919 |
| avg152T1 | T1w1mm | 2.0 | 208 | 0.8 | 208 | 2.6x | 0.5095 | 0.2586 |
| avg152T1 | T1w2mm | 1.9 | 47 | 0.5 | 47 | 3.9x | 0.5170 | 0.2802 |
| avg152T1 | T1w_MICCAI2017 | 2.2 | 200 | 0.7 | 200 | 3.2x | 0.4639 | 0.2272 |
| avg152T1 | T2w | 1.5 | 68 | 0.4 | 69 | 3.5x | 0.4412 | 0.2429 |
| avg152T1 | fmri | 1.4 | 32 | 0.4 | 34 | 3.7x | 0.4281 | 0.2773 |

### mixed default (`fast`/`fastx`) versus `fasthel` (14-thread capture check)

This targeted sweep uses only the 14-thread mode and compares the post-hoc masked Hellinger score. The four mixed coarse candidates run concurrently, so the observed end-to-end overhead over `fasthel` is generally only 0.00–0.09 s here (3.7% on average). No case changed basin catastrophically: the largest masked loss was −0.0053 (`T1w2mm→avg152T1`) and the largest gain was +0.0037 (`T1w2mm→MNI152_T1_1mm`). On the motivating hard-zeroed `M2017_test_T1w→MNI152_T1_1mm` case, which is not part of the bundled sweep, the mixed default recovered masked-brain NCC 0.4045 (determinant 0.696) instead of the failed HEL basin near NCC 0.105.

| Stationary | Moving | fasthel Time | default Time | Δ Time | fasthel Masked | default Masked | Δ Masked |
|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 1.06 | 1.11 | +0.05 | 0.2209 | 0.2202 | -0.0008 |
| MNI152_T1_1mm | T1w1mm | 1.28 | 1.37 | +0.09 | 0.2074 | 0.2064 | -0.0010 |
| MNI152_T1_1mm | T1w2mm | 1.01 | 1.01 | -0.01 | 0.2082 | 0.2118 | +0.0037 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 1.20 | 1.21 | +0.01 | 0.1751 | 0.1751 | +0.0000 |
| MNI152_T1_1mm | T2w | 0.90 | 0.92 | +0.02 | 0.1777 | 0.1781 | +0.0004 |
| MNI152_T1_1mm | fmri | 0.74 | 0.79 | +0.04 | 0.2119 | 0.2119 | +0.0000 |
| avg152T1 | FLAIR_MICCAI2017 | 0.47 | 0.49 | +0.02 | 0.2919 | 0.2919 | +0.0000 |
| avg152T1 | T1w1mm | 0.63 | 0.65 | +0.03 | 0.2586 | 0.2586 | +0.0000 |
| avg152T1 | T1w2mm | 0.44 | 0.39 | -0.04 | 0.2802 | 0.2749 | -0.0053 |
| avg152T1 | T1w_MICCAI2017 | 0.63 | 0.65 | +0.02 | 0.2272 | 0.2272 | +0.0000 |
| avg152T1 | T2w | 0.38 | 0.38 | -0.01 | 0.2429 | 0.2449 | +0.0020 |
| avg152T1 | fmri | 0.31 | 0.34 | +0.03 | 0.2773 | 0.2800 | +0.0027 |
| MNI152_2009_SSW +w | FLAIR_MICCAI2017 | 1.12 | 1.17 | +0.05 | 0.1548 | 0.1548 | +0.0000 |
| MNI152_2009_SSW +w | T1w1mm | 1.39 | 1.47 | +0.08 | 0.1725 | 0.1734 | +0.0009 |
| MNI152_2009_SSW +w | T1w2mm | 1.07 | 1.15 | +0.08 | 0.2042 | 0.2041 | -0.0002 |
| MNI152_2009_SSW +w | T1w_MICCAI2017 | 1.40 | 1.43 | +0.04 | 0.1436 | 0.1436 | +0.0000 |
| MNI152_2009_SSW +w | T2w | 0.97 | 1.00 | +0.03 | 0.1530 | 0.1530 | +0.0000 |
| MNI152_2009_SSW +w | fmri | 0.80 | 0.85 | +0.05 | 0.1378 | 0.1378 | +0.0000 |

### allineate Hellinger (`-cost hel`) is very similar to AFNI 3dAllineate

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 159.5 | 277 | 22.3 | 541 | 7.1x | 0.4355 | 0.2185 |
| MNI152_T1_1mm | T1w1mm | 147.9 | 460 | 22.9 | 714 | 6.5x | 0.4806 | 0.2119 |
| MNI152_T1_1mm | T1w2mm | 98.7 | 194 | 13.4 | 436 | 7.4x | 0.4710 | 0.2202 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 251.3 | 437 | 37.9 | 743 | 6.6x | 0.4392 | 0.1722 |
| MNI152_T1_1mm | T2w | 65.0 | 279 | 13.2 | 404 | 4.9x | 0.4185 | 0.1758 |
| MNI152_T1_1mm | fmri | 20.2 | 190 | 4.5 | 255 | 4.5x | 0.4051 | 0.2114 |
| avg152T1 | FLAIR_MICCAI2017 | 45.8 | 91 | 6.1 | 202 | 7.5x | 0.4819 | 0.2868 |
| avg152T1 | T1w1mm | 38.4 | 222 | 6.8 | 285 | 5.7x | 0.5109 | 0.2627 |
| avg152T1 | T1w2mm | 34.7 | 66 | 4.6 | 180 | 7.6x | 0.5123 | 0.2833 |
| avg152T1 | T1w_MICCAI2017 | 64.2 | 201 | 9.0 | 321 | 7.1x | 0.4727 | 0.2287 |
| avg152T1 | T2w | 22.5 | 85 | 3.9 | 146 | 5.8x | 0.4592 | 0.2257 |
| avg152T1 | fmri | 18.5 | 44 | 2.6 | 111 | 7.1x | 0.4325 | 0.2700 |

### AFNI 3dAllineate (reference, defaults)

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 278.3 | 510 | 195.1 | 512 | 1.4x | 0.4346 | 0.2129 |
| MNI152_T1_1mm | T1w1mm | 315.5 | 822 | 185.0 | 822 | 1.7x | 0.4811 | 0.2114 |
| MNI152_T1_1mm | T1w2mm | 187.5 | 509 | 133.2 | 506 | 1.4x | 0.4737 | 0.2143 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 491.6 | 689 | 321.9 | 696 | 1.5x | 0.4383 | 0.1706 |
| MNI152_T1_1mm | T2w | 119.1 | 542 | 79.0 | 542 | 1.5x | 0.4252 | 0.1503 |
| MNI152_T1_1mm | fmri | 42.7 | 524 | 32.3 | 538 | 1.3x | 0.4059 | 0.2078 |
| avg152T1 | FLAIR_MICCAI2017 | 75.5 | 150 | 52.8 | 155 | 1.4x | 0.4849 | 0.2658 |
| avg152T1 | T1w1mm | 92.2 | 372 | 56.3 | 373 | 1.6x | 0.5094 | 0.2658 |
| avg152T1 | T1w2mm | 75.0 | 116 | 51.0 | 115 | 1.5x | 0.5158 | 0.2787 |
| avg152T1 | T1w_MICCAI2017 | 87.4 | 290 | 55.9 | 294 | 1.6x | 0.4721 | 0.2211 |
| avg152T1 | T2w | 30.0 | 141 | 22.4 | 144 | 1.3x | 0.4692 | 0.1966 |
| avg152T1 | fmri | 23.5 | 98 | 17.0 | 92 | 1.4x | 0.4400 | 0.2331 |


### FLIRT (FSL, reference, defaults)

`flirt -in <mov> -ref <fix> -out out` (FSL 12-DOF affine, defaults; FLIRT optimizes correlation-ratio internally). FLIRT is single-threaded, so the parallel-thread columns (`14 Time` / `14 Peak RAM` / `Speed Up`) are omitted; *Time* and *Peak RAM* are its single-thread figures. As in every table here, *Cost* / *Cost Masked* are the post-hoc **Hellinger-affinity** goodness-of-fit (not FLIRT's internal correlation-ratio), so the quality column is directly comparable across all engines.

| Stationary | Moving | Time | Peak RAM | Cost | Cost Masked |
|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 6.0 | 195 | 0.4516 | 0.2236 |
| MNI152_T1_1mm | T1w1mm | 18.1 | 322 | 0.4821 | 0.2117 |
| MNI152_T1_1mm | T1w2mm | 5.6 | 144 | 0.4785 | 0.2055 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 18.5 | 300 | 0.3911 | 0.0721 |
| MNI152_T1_1mm | T2w | 7.3 | 167 | 0.3874 | 0.1701 |
| MNI152_T1_1mm | fmri | 4.7 | 135 | 0.3916 | 0.0748 |
| avg152T1 | FLAIR_MICCAI2017 | 4.6 | 82 | 0.5152 | 0.3176 |
| avg152T1 | T1w1mm | 13.5 | 241 | 0.5307 | 0.2799 |
| avg152T1 | T1w2mm | 4.7 | 47 | 0.5207 | 0.2680 |
| avg152T1 | T1w_MICCAI2017 | 10.9 | 221 | 0.4658 | 0.1464 |
| avg152T1 | T2w | 3.8 | 73 | 0.4393 | 0.2545 |
| avg152T1 | fmri | 3.8 | 35 | 0.4046 | 0.0878 |

## Weighted registration (`-weight`, AFNI 3dAllineate style)

These tables isolate the `-weight` option: each engine registers every moving image to the **same base** — the committed whole-head `benchmark/weighted/MNI152_2009_SSW` T1 template — under the **same graded base-space weight**, `MNI152_2009_SSW_weight` (an `@SSwarper`-style weight: full brain plus attenuated, non-zero scalp). The weight is applied graded per voxel exactly as AFNI 3dAllineate does (`allineate <mov> MNI152_2009_SSW -weight MNI152_2009_SSW_weight -cost hel|fast out`). The two allineate tables come straight from the default benchmark run (weighted templates are benchmarked automatically):

```bash
python3 benchmark.py            # plain templates + the weighted/ pair, both engines
```

The base is the same for both allineate tables, so *Cost* / *Cost Masked* are directly comparable **between `-cost hel` and `-cost fasthel`** but **not** against the plain-template tables above (those use `avg152T1`/`MNI152_T1_1mm` bases; the SSW base includes the whole head, which dilutes the Hellinger score). The masked ROI here is the weight's own brain plateau (`weight > 0.3*max`), not an external brain-mask file. The AFNI + `-weight` reference table below is a **historical reference on the earlier `MNI152_2009_template_SSW1` base** (from the removed `weight/` dev folder); it was NOT re-run and is not directly comparable to the two allineate tables above it.

**Reading them:** with the **weight-at-finest-level** fix (the graded weight now steers only the 2 mm stage; the 8 mm rigid coarse and 4 mm global-scale bracket stay whole-head — see `AGENTS.md`), the HEL-only fast engine no longer falls into the scalp-shrink basin. It now **matches or beats the ordinary `-cost hel` engine on every masked score** here — same-modality T1 (`T1w1mm` 0.173 vs 0.171, `T1w2mm` 0.204 vs 0.198, `T1w_MICCAI2017` 0.144 vs 0.137) and, more notably, the cross-modal rows (`T2w` 0.153 vs 0.113, `FLAIR_MICCAI2017` 0.155 vs 0.142, `fmri` 0.138 vs 0.124) — while running ~30–60× faster (~1–3 s vs 20–130 s). The earlier collapse (`T1w2mm`/`T1w_MICCAI2017`/`T2w` masked ~0.09–0.10) is gone. All rows still score lower than the plain-template tables because the SSW base includes the whole head (which dilutes the Hellinger score).

### allineate ordinary engine (`-cost hel`) + `-weight`

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_2009_SSW +w | FLAIR_MICCAI2017 | 109.4 | 293 | 15.8 | 519 | 6.9x | 0.2961 | 0.1421 |
| MNI152_2009_SSW +w | T1w1mm | 61.2 | 414 | 15.4 | 526 | 4.0x | 0.3697 | 0.1712 |
| MNI152_2009_SSW +w | T1w2mm | 78.6 | 230 | 11.3 | 395 | 7.0x | 0.3718 | 0.1982 |
| MNI152_2009_SSW +w | T1w_MICCAI2017 | 131.3 | 390 | 20.9 | 611 | 6.3x | 0.3227 | 0.1370 |
| MNI152_2009_SSW +w | T2w | 45.6 | 266 | 10.7 | 381 | 4.3x | 0.1895 | 0.1129 |
| MNI152_2009_SSW +w | fmri | 19.5 | 186 | 4.5 | 258 | 4.4x | 0.2566 | 0.1238 |

### fast HEL-only (`-cost fasthel`) + `-weight`

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_2009_SSW +w | FLAIR_MICCAI2017 | 2.4 | 188 | 1.2 | 191 | 2.0x | 0.3259 | 0.1548 |
| MNI152_2009_SSW +w | T1w1mm | 2.6 | 325 | 1.5 | 325 | 1.8x | 0.3633 | 0.1725 |
| MNI152_2009_SSW +w | T1w2mm | 2.3 | 165 | 1.2 | 165 | 2.0x | 0.3614 | 0.2042 |
| MNI152_2009_SSW +w | T1w_MICCAI2017 | 3.0 | 318 | 1.5 | 319 | 2.0x | 0.3370 | 0.1436 |
| MNI152_2009_SSW +w | T2w | 2.0 | 187 | 1.0 | 186 | 2.0x | 0.2570 | 0.1530 |
| MNI152_2009_SSW +w | fmri | 1.7 | 148 | 0.8 | 145 | 2.1x | 0.2895 | 0.1378 |

### AFNI 3dAllineate (reference) + `-weight` — *historical, prior `SSW1` base (not re-run)*

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_2009_template_SSW1 | FLAIR_MICCAI2017 | 166.7 | 516 | 114.9 | 516 | 1.5x | 0.2316 | 0.1301 |
| MNI152_2009_template_SSW1 | T1w1mm | 94.0 | 735 | 59.6 | 736 | 1.6x | 0.3687 | 0.1725 |
| MNI152_2009_template_SSW1 | T1w2mm | 111.6 | 528 | 83.1 | 522 | 1.3x | 0.3583 | 0.2046 |
| MNI152_2009_template_SSW1 | T1w_MICCAI2017 | 187.4 | 594 | 132.0 | 593 | 1.4x | 0.3330 | 0.1421 |
| MNI152_2009_template_SSW1 | T2w | 66.5 | 519 | 49.3 | 522 | 1.3x | 0.2687 | 0.1553 |
| MNI152_2009_template_SSW1 | fmri | 31.8 | 508 | 25.9 | 517 | 1.2x | 0.2886 | 0.1378 |
