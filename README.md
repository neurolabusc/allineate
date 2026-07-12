# allineate

Standalone affine (12 DOF) image registration for NIfTI files. Adapted from AFNI's 3dAllineate (public domain).

## Usage

```bash
allineate <moving> <stationary> [opts] <output>
```

Use `-` for `<moving>` to read the moving image from **stdin**, and `-` for `<output>` to write the result to **stdout** (see [Piping](#piping)).

Options:
- `-cost fast` / `-cost fastcr` — use the **fast** SPM/FLIRT-inspired affine engine instead of the default allineate engine (default: allineate). An independently implemented multiresolution 12-DOF path (8→4→2 mm pyramid) tuned for 3D adult brains. The coarse (8 mm) level searches orientation+translation as a **rigid** body with global scale locked at identity; global scale is introduced at 4 mm (with a discrete scale bracket so a true envelope-edge scale is still recovered) and anisotropic scale/shear at 2 mm (rigid→7→9→12 DOF). Locking scale at the coarsest level is deliberate: freeing it there lets the cost fall into a spurious isotropic-shrink basin on wide-FOV / short-axis cross-modal inputs. Returns and `-savemat`s the same world-mm `fixed_to_moving` transform. Not combinable with `-skullstrip`/`-sym`/`-zoom`. Default/`-cmass` scores the supplied-affine and COM-recentered starts once at 8 mm using `(1-cost)×overlap`, then runs only the winning descent; `-com` forces the recentered frame and `-nocmass` forces the supplied affine. The translation guardrail is ±128 mm, rotation capture is approximately ±30°, and global scale capture is approximately 0.75–1.33 (wide-scale/infant cases stay on the allineate `-zoom` path). Two cost selectors:
  - `-cost fast` (**default fast cost**, **Hellinger**) — robust for **cross-modal** registration (e.g. FLAIR/T2w → T1 template), fixing the shrink/roll that correlation-ratio suffers cross-modal. It remains much faster than the ordinary engine, while the ordinary engine's AFNI-style far-origin search now gives essentially the same MICCAI quality against the 1 mm MNI template. Same-modality T1 full Hellinger remains within ~0.009 of allineate. On the original T2w pairs no engine wins every metric: fast is slightly ahead on MNI full Hellinger and avg152T1 masked Hellinger, while allineate leads the complementary two scores. See `benchmark/README.md`.
  - `-cost fastcr` (**correlation-ratio**) — the CR cost is **unstable when modalities differ** (spurious isotropic shrink/roll on cross-modal); prefer `-cost fast` for any cross-modal work. Best kept for same-modality inputs, where it matches allineate quality. (CR is also what the synthetic capture suites use, because Hellinger/MI is imprecise on smooth/low-detail volumes and can't sharply recover the synthetic test phantoms.)
- `-cost XX` — cost function (allineate engine): `hel` (Hellinger, default), `lpc` (local Pearson, cross-modal), `lpa` (abs local Pearson, cross-modal), `ls` (Pearson/least-squares)
- `-cmass` / `-nocmass` — center-of-mass initialization
- `-source_automask` — fill outside source automask with noise (recommended with lpc/lpa)
- `-warp XX` — degrees of freedom: `sho` (shift, 3), `shr` (shift+rotate, 6), `srs` (+scale, 9), `aff` (full affine, 12, default)
- `-interp XX` — fine-pass matching interpolation: `NN`, `linear` (default), `cubic`
- `-final XX` — output interpolation: `NN`, `linear`, `cubic` (default; `linear` for `-skullstrip`)
- `-nearest` / `-linear` / `-cubic` — shortcuts for `-final`
- `-p <threads>` — maximum number of parallel threads (`0` = use all available); OpenMP builds only. Options may be given on either side of the output filename
- `-skullstrip XX` — brain mask (in moving/template space); output is the stationary image with non-brain voxels set to its darkest value
- `-robustfov [mm]` — crop the moving image to a robust field of view (default 170 mm) from the top of the head down, removing lower head/neck (emulates FSL robustfov); adjusts dimensions and any valid sform/qform so kept voxels retain their world coordinates. A useful preprocessing step before registering scans with lots of neck to templates that have little. **With the allineate engine, pair it with `-cmass`** (`-robustfov -cmass`): cropping removes the inferior anatomy the header-started coarse search uses as an anchor, so without a center-of-mass initialization the fit can drift into a wrong (inflated-scale) basin. The fast engine (`-cost fast`/`-cost fastcr`) already scores a COM seed and needs no help
- `-savemat out.json` — save the fitted registration affine to a self-describing JSON file: the world-space 4×4 `fixed_to_moving` transform (maps stationary/FIXED world-mm → moving/source world-mm, i.e. the pull/resampling transform) and its inverse `moving_to_fixed`, plus `cost`, `dof`, and the input filenames. Requires a stationary image (it is a registration result); not valid with `-skullstrip`. **The output image filename may be omitted to save only the matrix** (`allineate moving fixed -cost ls -savemat out.json`). If `-com`/`-sym`/`-sagseed` pre-steps ran, the matrix is relative to the seeded moving header (so `-applymat` reproduces a *plain* registration exactly; a pre-stepped one needs the same pre-steps applied to the input first)
- `-applymat in.json <target>` — apply a saved matrix (no registration): reslice the moving image onto the `<target>` grid using `fixed_to_moving`. Because the matrix is world-mm→world-mm, the target may have **any resolution / FOV / origin** sharing the fixed world frame (e.g. save at 2 mm, apply onto a 1 mm grid). Usage: `allineate <input> <target> -applymat in.json <output>`. Use `-final NN` for label/atlas volumes
- `-master grid.nii` — **register at the stationary resolution, output at a different one.** A coarse (e.g. 2 mm) template is usually enough for robust linear registration, but downstream processing often wants the result on a finer grid. `-master` reslices the registered moving image onto `grid.nii` — which must **share the stationary world frame** (typically a higher-res version of the same template) — instead of the stationary grid. The fit itself is unchanged (the saved `-savemat` matrix is identical with or without `-master`); only the output sampling differs, and the result is byte-identical to a `-savemat` + `-applymat`-onto-`grid` two-step. Works with both engines. `-final` sets the reslice interpolation (`-final NN` for label/atlas volumes). Example: `allineate T2w.nii.gz avg152T1_2mm.nii.gz -cost fast -master avg152T1_1mm.nii.gz out.nii.gz`. Not combinable with `-applymat`/`-skullstrip`
- `-zoom` — flag that **size may be abnormal** (e.g. an infant brain vs an adult template). Two effects: (1) adds a **global isotropic scale** DOF to the `-sagseed` seed fit (needs `-sym` + a template), so the bulk of an extreme size mismatch is corrected *before* the main fit; and (2) **relaxes the main affine's scale range** to [0.5, 2.0] (from the default 0.711–1.406), since the user is explicitly signalling abnormal size. Only the scale limits relax — all other regularization, and the default (adult) behavior, are unchanged, so it doesn't risk overfitting normal brains. The seed scale is isotropic (single factor, no shape distortion). May be counter-productive for modalities with little scalp signal vs a T1 template (EPI fMRI, DWI)
- `-dark_automask` — drop matched voxel pairs from the cost where **either** image is at its darkest value (background / zero-pad / NaN-filled-to-zero). Uses each image's own minimum (not a hardcoded 0), so it is safe for signed data such as CT Hounsfield units. Folds into the existing per-point cost weight (no extra pass); falls back to unmasked weights if too few pairs survive. Applies to the main registration and to `-sym`/`-symd`/`-sagseed`. Off by default. Complements `-source_automask` (which noise-fills the source background instead of excluding it)
- `-symd` — like `-sym` but first **de-obliques** the frame (snaps the sform to axis-aligned, treating the voxel grid as the anatomical frame). For scans deliberately acquired oblique-to-world (AC-PC angling, tilted infant/kyphotic positioning) where the voxel axes *are* the anatomical axes, this stops `-sym` from rolling an already-grid-symmetric head onto the oblique world frame
- `-symb` — **auto-compete**: run both `-sym` and `-symd` and keep the de-obliqued frame **only if its correction rotation is smaller by more than 0.5°**; otherwise keep the original world frame (ties and sub-0.5° differences favor the world frame, so an already axis-aligned image is left untouched — no needless header rewrite). Picks de-oblique for a clearly oblique grid-symmetric head, the world frame when the sform genuinely encodes anatomical axes. **Limitation:** the decision compares only correction *rotation*, not fit *cost*, so it is a heuristic — it does not guarantee the anatomically correct frame in ambiguous cases
- `-com` — set the origin to the brightness center of mass: computes the intensity-weighted centroid, maps it to world coordinates, and folds a pure translation into the header so the centroid sits at world (0, 0, 0). Header-only (no resampling); a cheap centered starting point for symmetric images. Runs early in the chain (after `-robustfov`, before `-sym`)
- `-sym` — template-free midsagittal-plane (MSP) alignment. Registers the image to its own world-X mirror, takes the half of the recovered rigid transform, and applies it as a re-centering correction. **Standalone** (no stationary image): reslices the data symmetric about world X = 0. **As a pre-step** (with a stationary image): folds the correction into the moving header as an initial estimate before registering to the template
- `-nosagseed` — disable the **`-sagseed`** in-MSP rigid seed, which is **on by default whenever `-sym` runs with a stationary template**. `-sym` centers the MSP on world X = 0 but is blind to the three MSP-preserving rigid DOF (P-A shift, I-S shift, and pitch); `-sagseed` then runs a 3-DOF-constrained fit of the moving image to the (MSP-aligned) template freeing only those {y-shift, z-shift, pitch}, and folds the result into the moving header — completing a full-rigid seed before the main unconstrained fit. Uses the same `-cost`/`-interp`/`-source_automask`/`-cmass` as the main registration. No standalone mode (it needs a template); it does nothing without `-sym`

The stationary image is optional: with only a moving image and a preprocessing option, `allineate` runs the preprocessing and writes the result (no registration):

Registration uses the sform when `sform_code >= qform_code`, otherwise the qform; if the preferred form is unset or degenerate the other is tried. An input with neither usable form receives the same pixdim-centered fallback frame used by niimath.

```bash
allineate neck.nii.gz -robustfov out.nii.gz                    # crop neck only
allineate neck.nii.gz MNI152_T1_1mm -robustfov -cost ls out    # crop, then register
allineate off.nii.gz -com out.nii.gz                           # center origin on brightness mass
allineate off.nii.gz MNI152_T1_1mm -com -cost ls out           # center, then register
allineate T1.nii.gz MNI152_T1_1mm -cost ls -savemat xf.json out # register + save affine as JSON
allineate T1.nii.gz MNI152_T1_1mm -cost fast out         # fast SPM/FLIRT-inspired affine, Hellinger (~8–25× faster single-thread)
allineate tilted.nii.gz -sym out.nii.gz                        # midsagittal reslice
allineate tilted.nii.gz MNI152_T1_1mm -sym -cost ls out        # MSP + sagseed full-rigid seed, then register
allineate tilted.nii.gz MNI152_T1_1mm -sym -nosagseed out      # MSP seed only (disable sagseed)
```

## Piping

The moving image and the output support UNIX pipes via `-`:

```bash
# uncompressed NIfTI-1 on stdin; result to stdout, then compress
cat moving_2mm.nii | allineate - template_2mm.nii.gz -cost ls - | gzip > out.nii.gz
```

Piped input uses raw NIfTI-1, matching the shared niimath I/O layer. Only the moving image can come from stdin — the stationary image and the `-skullstrip` mask must be files. Piped **output** is uncompressed NIfTI-1 (pipe through `gzip` for compression). Piped I/O supports NIfTI-1 only.

## Build

```bash
make              # optimized release build (default) -> ./allineate
make test         # build, then run the deterministic regression gate (exits nonzero on failure)
make sanitize     # AddressSanitizer build (heap-overflow / use-after-free)
make profile      # release build with per-stage timing (-DAL_PROFILE)
make clean

# `make test` requires python3 + numpy + nibabel (pip install numpy nibabel); a missing
# package makes the gate exit 2, distinct from a product regression (exit 1).
```

On Apple Silicon macOS LeakSanitizer is unavailable, so `make sanitize` does not catch leaks there; use `MallocNanoZone=0 leaks --atExit -- ./allineate …` for leak detection.

Or compile manually (note the `miniCoreFLT.c` source):

```bash
# macOS Apple Clang + libomp (fastest on ARM; requires: brew install libomp)
clang -O3 -ffast-math -fno-finite-math-only -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c coreg_fast.c -lz -lm -o allineate

# macOS Homebrew GCC + OpenMP
gcc-15 -O3 -ffast-math -fno-finite-math-only -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c coreg_fast.c -lz -lm -o allineate

# Linux with GCC + OpenMP
gcc -O3 -ffast-math -fno-finite-math-only -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c coreg_fast.c -lz -lm -o allineate
```

## Examples

The benchmark harness runs on the bundled datasets under `benchmark/inputs/`, `benchmark/templates/`, and `benchmark/masks/` (see `benchmark/README.md`), or substitute your own NIfTI volumes:

```bash
./allineate moving.nii.gz template.nii.gz registered.nii.gz
```

The `benchmark/` folder also has a script for evaluating speed/quality across all three engines, `benchmark/benchmark.py` (see `benchmark/README.md`).
