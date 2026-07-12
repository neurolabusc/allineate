# allineate benchmark

A small, repeatable benchmark of registration quality/speed/memory on a fixed set of images, comparing three engines plus a robust fast preset. **`fast` is now the default cost**, so the ordinary AFNI-style engine must be selected explicitly with `-cost hel`:

- **allineate** — `allineate <mov> <fix> -cost hel out` (AFNI-style ordinary engine, 12-DOF affine, Hellinger cost)
- **fast**      — `allineate <mov> <fix> out` (SPM/FLIRT-inspired path, now the **default**, **Hellinger** cost)
- **fast robust** — `allineate <mov> <fix> -robustfov -com -cost fast out` (crop and recenter before the fast fit)
- **AFNI**      — `3dAllineate -base <fix> -source <mov> -prefix out` (reference tool, defaults; only with `--afni`)

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

## Running it

```bash
make                          # build the release binary first
cd benchmark
python3 benchmark.py          # allineate + fast engines (needs numpy + nibabel)
python3 benchmark.py --engine allineate  # ordinary AFNI-style engine (-cost hel) only
python3 benchmark.py --engine fast-robust  # robust preprocessing + fast engine only
python3 benchmark.py --afni   # also benchmark AFNI 3dAllineate (must be on PATH)
```

Each engine runs every (stationary, moving) pair at **1 thread** and at **all cores** (N = `os.cpu_count()`), so one table per engine carries both the single- and multi-thread cost plus the speed-up. Match quality is a post-hoc **Hellinger-affinity** metric (`benchmark/hellinger.py`): the Hellinger distance of the joint 2D intensity histogram from independence — statistical DEPENDENCE between corresponding voxels (higher = better). Unlike NCC it assumes no linear intensity relationship, so it is valid for cross-modal pairs (T2/fMRI → T1) and comparable across engines (it is independent of each engine's own registration cost).

## Results (Apple M4 Max, Apple Clang + libomp, release build)

**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB (peak RSS); *Speed Up* = 1-thread Time / N-thread Time; *Cost* / *Cost Masked* = Hellinger-affinity match quality (higher = better; masked restricts to the template brain mask). N = 14 threads. The cross-modal `FLAIR_MICCAI2017`/`T2w`/`fmri` rows are the especially interesting ones for quality.

### allineate fast (`-cost fast`) is inspired by SPM/FLIRT

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 2.3 | 159 | 1.1 | 160 | 2.1x | 0.4346 | 0.2212 |
| MNI152_T1_1mm | T1w1mm | 2.8 | 296 | 1.3 | 297 | 2.2x | 0.4796 | 0.2079 |
| MNI152_T1_1mm | T1w2mm | 2.3 | 134 | 1.0 | 136 | 2.3x | 0.4747 | 0.2108 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 2.4 | 288 | 1.1 | 290 | 2.1x | 0.4389 | 0.1759 |
| MNI152_T1_1mm | T2w | 1.9 | 157 | 0.9 | 157 | 2.2x | 0.4164 | 0.1788 |
| MNI152_T1_1mm | fmri | 1.6 | 122 | 0.7 | 123 | 2.1x | 0.4034 | 0.2121 |
| avg152T1 | FLAIR_MICCAI2017 | 1.6 | 70 | 0.4 | 71 | 3.7x | 0.4784 | 0.2924 |
| avg152T1 | T1w1mm | 1.7 | 208 | 0.6 | 208 | 3.0x | 0.5092 | 0.2572 |
| avg152T1 | T1w2mm | 1.3 | 46 | 0.4 | 47 | 3.6x | 0.5170 | 0.2802 |
| avg152T1 | T1w_MICCAI2017 | 1.9 | 199 | 0.6 | 200 | 3.3x | 0.4656 | 0.2280 |
| avg152T1 | T2w | 1.3 | 68 | 0.3 | 68 | 3.8x | 0.4402 | 0.2414 |
| avg152T1 | fmri | 1.3 | 31 | 0.3 | 32 | 4.2x | 0.4296 | 0.2752 |

### allineate Hellinger (`-cost hel`) is very similar to AFNI 3dAllineate

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 160.5 | 238 | 26.3 | 509 | 6.1x | 0.4347 | 0.2223 |
| MNI152_T1_1mm | T1w1mm | 159.4 | 423 | 23.4 | 659 | 6.8x | 0.4793 | 0.2116 |
| MNI152_T1_1mm | T1w2mm | 99.6 | 190 | 14.6 | 405 | 6.8x | 0.4699 | 0.2230 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 226.0 | 398 | 32.7 | 639 | 6.9x | 0.4383 | 0.1750 |
| MNI152_T1_1mm | T2w | 67.6 | 273 | 13.3 | 389 | 5.1x | 0.4137 | 0.1850 |
| MNI152_T1_1mm | fmri | 21.8 | 183 | 4.8 | 240 | 4.5x | 0.4045 | 0.2172 |
| avg152T1 | FLAIR_MICCAI2017 | 49.2 | 90 | 5.9 | 200 | 8.3x | 0.4780 | 0.2917 |
| avg152T1 | T1w1mm | 58.8 | 221 | 8.6 | 335 | 6.8x | 0.5076 | 0.2643 |
| avg152T1 | T1w2mm | 41.3 | 65 | 5.1 | 172 | 8.1x | 0.5084 | 0.2798 |
| avg152T1 | T1w_MICCAI2017 | 79.5 | 197 | 9.1 | 318 | 8.7x | 0.4673 | 0.2304 |
| avg152T1 | T2w | 22.5 | 84 | 3.9 | 146 | 5.7x | 0.4544 | 0.2312 |
| avg152T1 | fmri | 20.6 | 49 | 3.7 | 98 | 5.6x | 0.4297 | 0.2788 |

### allineate fast robust (`-robustfov -com -cost fast`)

This is the recommended robust fast preset: crop excess inferior FOV, recenter the
retained brightness mass, then estimate the fast affine. The current CLI uses
`-cost fast`; it replaces the older `-coreg fast` spelling. The complete suite
was run with:

```bash
python3 benchmark.py --engine fast-robust
```

The tables report all 12 moving/template pairs. Hellinger-affinity quality was
scored from the deterministic 14-thread output; the single-thread table therefore
reports only its directly measured time and peak RSS.

#### fast robust: single thread (`-p 1`)

| Stationary | Moving | Time (s) | Peak RAM (MB) |
|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 2.4 | 151 |
| MNI152_T1_1mm | T1w1mm | 2.3 | 263 |
| MNI152_T1_1mm | T1w2mm | 2.2 | 128 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 2.4 | 254 |
| MNI152_T1_1mm | T2w | 1.9 | 169 |
| MNI152_T1_1mm | fmri | 1.5 | 121 |
| avg152T1 | FLAIR_MICCAI2017 | 1.3 | 64 |
| avg152T1 | T1w1mm | 1.5 | 202 |
| avg152T1 | T1w2mm | 1.2 | 40 |
| avg152T1 | T1w_MICCAI2017 | 1.5 | 193 |
| avg152T1 | T2w | 1.2 | 78 |
| avg152T1 | fmri | 1.1 | 35 |

#### fast robust: 14 threads (`-p 14`)

| Stationary | Moving | Time (s) | Peak RAM (MB) | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|
| MNI152_T1_1mm | FLAIR_MICCAI2017 | 0.9 | 151 | 2.6x | 0.4349 | 0.2216 |
| MNI152_T1_1mm | T1w1mm | 1.1 | 264 | 2.1x | 0.4795 | 0.2071 |
| MNI152_T1_1mm | T1w2mm | 0.9 | 128 | 2.3x | 0.4714 | 0.2049 |
| MNI152_T1_1mm | T1w_MICCAI2017 | 1.0 | 254 | 2.4x | 0.4393 | 0.1748 |
| MNI152_T1_1mm | T2w | 0.8 | 170 | 2.2x | 0.4177 | 0.1804 |
| MNI152_T1_1mm | fmri | 0.7 | 122 | 2.2x | 0.4037 | 0.2092 |
| avg152T1 | FLAIR_MICCAI2017 | 0.3 | 65 | 3.8x | 0.4875 | 0.2910 |
| avg152T1 | T1w1mm | 0.5 | 203 | 3.2x | 0.5068 | 0.2606 |
| avg152T1 | T1w2mm | 0.3 | 41 | 3.7x | 0.5152 | 0.2739 |
| avg152T1 | T1w_MICCAI2017 | 0.4 | 194 | 3.4x | 0.4647 | 0.2275 |
| avg152T1 | T2w | 0.3 | 78 | 3.6x | 0.4394 | 0.2491 |
| avg152T1 | fmri | 0.3 | 33 | 3.9x | 0.4290 | 0.2762 |

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
