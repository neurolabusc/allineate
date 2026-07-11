# allineate benchmark

A small, repeatable benchmark of registration quality/speed/memory on a fixed set of images,
comparing three engines with **default settings**, each in its own table:

- **allineate** — `allineate <mov> <fix> out` (this project's default engine, 12-DOF affine, Hellinger cost)
- **fast**      — `allineate <mov> <fix> -cost fast out` (SPM/FLIRT-inspired path, **Hellinger** cost — the default fast cost)
- **AFNI**      — `3dAllineate -base <fix> -source <mov> -prefix out` (reference tool, defaults; only with `--afni`)

## Data provenance & licensing

All bundled imaging data is cleared for redistribution.

- **Templates & masks** — `avg152T1`, `MNI152_T1_1mm`, and the brain masks are the standard
  FSL/MNI152 average atlases (non-subject, publicly redistributable under the FSL/MNI terms).
- **Subject moving images** — `T1w1mm`, `T1w2mm`, `T2w`, `fmri` are de-identified and cleared by the
  data owner for sharing.

The `make test` correctness gate is 100% synthetic and does not depend on this data.

## Inputs

| role | images |
|---|---|
| moving | `T1w1mm` (0.9 mm T1), `T1w2mm` (2 mm T1), `T2w` (1×1×2 mm T2, **cross-modal**), `fmri` (2.5×2.5×2 mm EPI, **cross-modal**) |
| stationary | `avg152T1` (2 mm), `MNI152_T1_1mm` (1 mm) |

## Running it

```bash
make                          # build the release binary first
cd benchmark
python3 benchmark.py          # allineate + fast engines (needs numpy + nibabel)
python3 benchmark.py --afni   # also benchmark AFNI 3dAllineate (must be on PATH)
```

Each engine runs every (stationary, moving) pair at **1 thread** and at **all cores** (N =
`os.cpu_count()`), so one table per engine carries both the single- and multi-thread cost plus
the speed-up. Match quality is a post-hoc **Hellinger-affinity** metric (`benchmark/hellinger.py`):
the Hellinger distance of the joint 2D intensity histogram from independence — statistical
DEPENDENCE between corresponding voxels (higher = better). Unlike NCC it assumes no linear
intensity relationship, so it is valid for cross-modal pairs (T2/fMRI → T1) and comparable across
engines (it is independent of each engine's own registration cost).

## Results (Apple M4 Max, Apple Clang + libomp, release build)

**Legend** — *Time* in seconds (end-to-end, includes read/write); *Peak RAM* in MB (peak RSS);
*Speed Up* = 1-thread Time / N-thread Time; *Cost* / *Cost Masked* = Hellinger-affinity match
quality (higher = better; masked restricts to the template brain mask). N = 14 threads. The
cross-modal `T2w`/`fmri` rows are the interesting ones for quality.

### allineate (default engine)

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | T1w1mm | 62.0 | 408 | 14.7 | 527 | 4.2x | 0.4806 | 0.2116 |
| MNI152_T1_1mm | T1w2mm | 34.8 | 233 | 8.7 | 320 | 4.0x | 0.4707 | 0.2218 |
| MNI152_T1_1mm | T2w | 50.8 | 270 | 10.7 | 383 | 4.8x | 0.4145 | 0.1838 |
| MNI152_T1_1mm | fmri | 17.5 | 186 | 4.5 | 237 | 3.8x | 0.4047 | 0.2164 |
| avg152T1 | T1w1mm | 21.3 | 240 | 4.6 | 293 | 4.7x | 0.5079 | 0.2650 |
| avg152T1 | T1w2mm | 17.1 | 63 | 2.9 | 116 | 5.9x | 0.5108 | 0.2807 |
| avg152T1 | T2w | 18.0 | 84 | 3.2 | 141 | 5.6x | 0.4560 | 0.2302 |
| avg152T1 | fmri | 12.4 | 46 | 2.5 | 82 | 5.0x | 0.4302 | 0.2772 |

### fast (`-cost fast`)

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | T1w1mm | 2.5 | 296 | 1.3 | 296 | 1.9x | 0.4790 | 0.2066 |
| MNI152_T1_1mm | T1w2mm | 2.2 | 134 | 1.1 | 135 | 2.1x | 0.4748 | 0.2140 |
| MNI152_T1_1mm | T2w | 2.3 | 157 | 1.1 | 157 | 2.2x | 0.4289 | 0.1522 |
| MNI152_T1_1mm | fmri | 1.9 | 122 | 0.8 | 123 | 2.2x | 0.4115 | 0.2115 |
| avg152T1 | T1w1mm | 2.0 | 207 | 0.7 | 208 | 2.9x | 0.5093 | 0.2620 |
| avg152T1 | T1w2mm | 1.5 | 45 | 0.4 | 47 | 3.3x | 0.5168 | 0.2752 |
| avg152T1 | T2w | 1.4 | 67 | 0.4 | 68 | 3.4x | 0.4587 | 0.2021 |
| avg152T1 | fmri | 1.1 | 33 | 0.3 | 32 | 3.5x | 0.4370 | 0.2616 |

### AFNI 3dAllineate (reference, defaults)

| Stationary | Moving | 1 Time | 1 Peak RAM | 14 Time | 14 Peak RAM | Speed Up | Cost | Cost Masked |
|---|---|---|---|---|---|---|---|---|
| MNI152_T1_1mm | T1w1mm | 327.2 | 822 | 203.1 | 823 | 1.6x | 0.4811 | 0.2114 |
| MNI152_T1_1mm | T1w2mm | 196.2 | 504 | 150.0 | 510 | 1.3x | 0.4737 | 0.2143 |
| MNI152_T1_1mm | T2w | 115.8 | 540 | 88.2 | 535 | 1.3x | 0.4252 | 0.1503 |
| MNI152_T1_1mm | fmri | 40.7 | 525 | 35.4 | 523 | 1.2x | 0.4059 | 0.2078 |
| avg152T1 | T1w1mm | 102.8 | 370 | 62.3 | 374 | 1.6x | 0.5094 | 0.2658 |
| avg152T1 | T1w2mm | 77.7 | 118 | 57.2 | 115 | 1.4x | 0.5158 | 0.2787 |
| avg152T1 | T2w | 30.9 | 140 | 26.1 | 145 | 1.2x | 0.4692 | 0.1966 |
| avg152T1 | fmri | 24.1 | 91 | 20.5 | 86 | 1.2x | 0.4400 | 0.2331 |

## Observations

- **Speed.** `fast` is by far the quickest — **~20–130× faster than AFNI single-thread** (e.g.
  MNI/T1w1mm: 2.5 s vs 327 s) and **~8–25× faster than `allineate`** (2.5 s vs 62 s), at matched
  quality. `allineate` is ~1.7–6× faster than AFNI single-thread.
- **Threading.** `allineate` scales best (~3.8–6×). `fast` gains less (~1.9–3.5×) because it is already
  dominated by fixed serial setup at ~1 s. **AFNI barely parallelizes (~1.2–1.6×)** — most of its
  wall time is serial, so its multi-thread numbers stay high.
- **Memory.** `fast` uses the least RAM on every pair; AFNI uses the most (up to ~820 MB).
- **Quality is close across all three engines** — full Hellinger is within ~0.01–0.02 almost
  everywhere, confirming all three reach the same basin on these pairs. On same-modality T1, `fast`
  (Hellinger) tracks `allineate` closely on full Hellinger (within ~0.006, ahead on 3 of 4 pairs) and sits marginally below on the brain-masked metric. On the cross-modal **`T2w` masked** metric, `fast`
  (0.152 / 0.202) and AFNI (0.150 / 0.197) both sit a little below `allineate` (0.184 / 0.230) — i.e.
  `allineate` still gives the tightest brain-interior T2w fit, while `fast` matches AFNI. For the
  correlation-ratio variant (`-cost fastcr`), see the note in the top-level README: it is meant for
  same-modality / synthetic use and is weaker cross-modal, so it is not benchmarked here.
- This benchmark is **descriptive** (speed/quality/RAM), not a pass/fail gate — that is `make test`.
