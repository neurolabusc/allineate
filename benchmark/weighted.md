# Hard-zeroed bases: weighted and stripped benchmark

This benchmark isolates the registrations that ordinary whole-head templates do not
exercise: a stationary/base image whose air, face, or non-brain tissue has been set
exactly to zero. This is increasingly common with skull stripping and defacing.

## Fixtures

| source | base | exact-minimum fraction |
|---|---|---:|
| `templates/` | `avg152T1` (whole head) | 0.0001 |
| `templates/` | `MNI152_T1_1mm` (whole head) | 0.061 |
| `weighted/` | `MNI152_2009_SSW` | 0.622 |
| generated | `MNI152_T1_1mm` × brain mask | 0.747 |
| generated | `avg152T1` × brain mask | 0.739 |

The generated bases are created in a temporary directory by `hardzero.py`; they are
not tracked duplicates. The observed classes are separated by an order of magnitude,
so the engine uses a conservative 0.25 exact-minimum fraction to detect a hard-zeroed
base.

## Why rigid coarse fails

The problem is not simply a large shared zero bin. At the correct M2017→SSW pose,
the engine's Hellinger cost is better than at the failed pose at every pyramid level.
The search does not reach it because orientation and scale are coupled on a
brain-only silhouette:

| 8 mm pose | minimized cost |
|---|---:|
| correct orientation + anisotropic scale | **0.8937** |
| correct orientation + isotropic scale | 0.9224 |
| wrong ~30° pitch | 0.9241 |
| correct orientation, scale locked to 1 | **0.9433** |

The historical rigid 8 mm rule is correct for whole-head images, where it prevents
spurious shrink, but the correct stripped-base orientation is genuinely worse until
scale is present. Unconditionally freeing scale is also unsafe: it collapsed
T1w1mm→stripped NCC from about 0.48 to 0.10.

## Adaptive strategy

Whole-head bases retain the fast mixed strategy byte-for-byte. Hard-zeroed bases run:

1. Historical rigid HEL coarse.
2. Bounded HEL scale×z-ratio coarse search with angular non-maximum suppression.
3. CR coarse handed to the ordinary HEL fine stages.

All three reach 2 mm and 7 DOF. The best fine-level strategy alone receives the
9/12-DOF polish. Selecting at 4 mm is unsafe: it disagreed with the 2 mm winner in
25/40 development cases. The early whole-head `fastx` selector is also excluded from
the hard-zero arbitration; when included it won at 7 DOF but selected two poor
post-polish fits.

## Before versus adaptive result

The table below records the decisive same-modality cases from the original 21-case
comparison. NCC is measured inside the stationary template brain mask.

| moving → hard-zeroed base | before NCC | adaptive NCC |
|---|---:|---:|
| T1w_ARC2017 → SSW | 0.2778 | **0.3357** |
| T1w1mm → MNI152-strip | 0.1340 | **0.4770** |
| T1w_ARC2017 → MNI152-strip | 0.1168 | **0.4291** |
| T1w_MICCAI2017 → MNI152-strip | −0.0286 | **0.0965** |
| T1w_ARC2017 → avg152-strip | 0.1079 | **0.4972** |
| T1w_MICCAI2017 → avg152-strip | −0.0373 | **0.0994** |

Across all 21 hard-zeroed pairs, masked Hellinger improved beyond noise in 8,
regressed in 0, and tied in 13. Among the 12 same-modality T1 rows, masked NCC
improved beyond noise in 6, regressed in 0, and tied in 6. Whole-head output for the
motivating ARC2017→MNI152 case remains byte-identical to the pre-multi-start `fastx`
output.

## Reproduce

From the repository root:

```bash
python3 benchmark/hardzero.py -p 14
```

The script reports masked Hellinger for all modalities and masked NCC for T1 inputs.
The correctness gate separately generates MNI152-strip and requires NCC ≥ 0.40 plus
bit-identical `-p 1`/`-p 4` output.
