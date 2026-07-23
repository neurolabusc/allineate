# allineate

Standalone affine (12 DOF) image registration for NIfTI files. Adapted from AFNI's 3dAllineate (public domain).

## Usage

```bash
allineate <moving> <stationary> [opts] <output>
```

Use `-` for `<moving>` to read the moving image from **stdin**, and `-` for `<output>` to write the result to **stdout** (see [Piping](#piping)).

Options:
- `-cost fast` / `fastx` / `fasthel` / `fastcr` — the **fast** SPM/FLIRT-inspired affine engine. **`fast` is the default** and is an alias of `fastx`; a bare `allineate <mov> <fix> out` therefore runs the adaptive fast strategy. Use `-cost hel` (or `nmi`/`lpc`/`lpa`/`ls`) to select the ordinary AFNI-style allineate engine instead. The fast engine is an independently implemented multiresolution 12-DOF path (8→4→2 mm pyramid) tuned for 3D adult brains. The translation guardrail is ±128 mm, rotation capture is approximately ±30°, and global scale capture is approximately 0.75–1.33.
  - `-cost fast` or `-cost fastx` (**default**) — on a whole-head base, independently fits the 8 mm rigid stage under Hellinger and correlation ratio from both the supplied-affine and COM frames (up to four candidates). OpenMP runs those independent optimizers concurrently; the HEL dependence×overlap winner alone continues through 4/2 mm. A hard-zeroed/skull-stripped base automatically uses a broader three-strategy search: rigid HEL, bounded scale-bracketed HEL, and CR-seeded HEL all reach the 2 mm 7-DOF stage, then only the winner receives 9/12-DOF polish. This addresses masked/defaced inputs without charging whole-head registrations for the extra search.
  - `-cost fasthel` — forces the historical Hellinger-only fast path. Hellinger is robust cross-modal on real, sharp-edged images, but can have a shallow coarse landscape when large regions have been hard-zeroed.
  - `-cost fastcr` — forces correlation ratio. It is useful for same-modality inputs and smooth synthetic phantoms, but is not recommended as a general cross-modal default because it can enter spurious shrink/roll basins.
  - Default/`-cmass` permits supplied-affine and COM initialization; `-com` forces the recentered frame and `-nocmass` forces the supplied affine. The fast engine is not combinable with `-skullstrip`/`-zoom`; the `-com`/`-sym`/`-symd`/`-symb` header seeds remain supported.
- `-weight <img>` — **AFNI 3dAllineate-style per-voxel registration weight** (both engines). A **base(stationary)-space** weight image that must share the stationary grid (its dimensions AND world frame are checked: the 8 stationary-index corners are mapped through the weight's own sform/qform and the fit is rejected if any diverges more than ~1/20 of a voxel from the stationary frame, so a matching-dims image in a different orientation is caught rather than silently misaligned). It is **normalized to [0,1]** (divided by its own max) and used **graded, per voxel**, exactly as `3dAllineate -weight`: on the ordinary engine it **replaces the automatic weight** (and, unlike the autoweight, is not binarized for the box-mode costs — it stays graded); on the fast engine it steers the **finest (2 mm) pyramid stage only** — the coarse rigid capture and the 4 mm global-scale stage stay unweighted so a brain-concentrated weight cannot re-select a shrunken global scale (the FLIRT Fig-1 basin) — weighting each fixed sample there. A voxel weighted 0 is excluded and one weighted ~1 dominates. Equivalent to AFNI: `allineate T1.nii MNI_SSW1.nii -cost hel -weight MNI_SSW2.nii out.nii` reproduces `3dAllineate -base MNI_SSW1 -weight MNI_SSW2 -source T1 -prefix out` (Pearson ~0.98 on the MNI 2009 sample). **Keep the out-of-ROI head attenuated (nonzero), not zeroed:** a weight that fully zeroes everything outside the brain removes the scale anchor, and the cost can then shrink the source to drag its scalp into the ROI (the FLIRT Fig-1 degeneracy). **Caveat (fast engine):** the fast cost is more sensitive to this than the ordinary/AFNI-equivalent engine and can still fall into the scalp-shrink basin on hard cross-modal pairs even with an attenuated weight — for reliable weighted registration prefer `-cost hel`. Rejected with `-applymat`, `-skullstrip`, and preprocessing-only (no stationary image).
- `-cost XX` — cost function for the ordinary AFNI-style allineate engine (selecting any of these switches off the default `fast` engine): `hel` (Hellinger, the ordinary engine's default), `nmi` (normalized mutual information), `lpc` (local Pearson, cross-modal), `lpa` (abs local Pearson, cross-modal), `ls` (Pearson/least-squares)
- `-cmass` / `-nocmass` — center-of-mass initialization
- `-source_automask` — fill outside source automask with noise (recommended with lpc/lpa)
- `-warp XX` — degrees of freedom: `sho` (shift, 3), `shr` (shift+rotate, 6), `srs` (+scale, 9), `aff` (full affine, 12, default)
- `-interp XX` — fine-pass matching interpolation: `NN`, `linear` (default), `cubic`
- `-final XX` — output interpolation: `NN`, `linear`, `cubic` (default; `linear` for `-skullstrip`)
- `-nearest` / `-linear` / `-cubic` — shortcuts for `-final`
- `-fill XX` — out-of-FOV fill value for voxels of the output reslice that fall outside the moving image: `auto` (default), `zero`, or `nan`. `auto` fills with 0 for positive-only images (e.g. MRI with a 0 background — output stays byte-identical to the historical behavior), but for an image whose darkest voxel is negative (CT/X-ray Hounsfield units, where air ≈ -1000) it fills with that darkest value so out-of-FOV reads as air rather than soft tissue; `zero` always fills 0; `nan` fills NaN. Honored by both engines (the ordinary engine applies it internally; the fast, `-applymat`, and `-master` reslices apply it in `main.c`)
- `-p <threads>` — maximum number of parallel threads (`0` = use all available); OpenMP builds only. Options may be given on either side of the output filename
- `-skullstrip XX` — brain mask (in moving/template space); output is the stationary image with non-brain voxels set to its darkest value
- `-robustfov [mm]` — crop the moving image to a robust field of view (default 170 mm) from the top of the head down, removing lower head/neck (emulates FSL robustfov); adjusts dimensions and any valid sform/qform so kept voxels retain their world coordinates. A useful preprocessing step before registering scans with lots of neck to templates that have little. **With the allineate engine, pair it with `-cmass`** (`-robustfov -cmass`): cropping removes the inferior anatomy the header-started coarse search uses as an anchor, so without a center-of-mass initialization the fit can drift into a wrong (inflated-scale) basin. The fast engine already considers a COM frame by default and needs no help
- `-savemat out.json` — save the fitted registration affine to a self-describing JSON file: the world-space 4×4 `fixed_to_moving` transform (maps stationary/FIXED world-mm → moving/source world-mm, i.e. the pull/resampling transform) and its inverse `moving_to_fixed`, plus `cost`, `dof`, and the input filenames. Requires a stationary image (it is a registration result); not valid with `-skullstrip`. **The output image filename may be omitted to save only the matrix** (`allineate moving fixed -cost ls -savemat out.json`). If `-com`/`-sym`/`-sagseed` pre-steps ran, the matrix is relative to the seeded moving header (so `-applymat` reproduces a *plain* registration exactly; a pre-stepped one needs the same pre-steps applied to the input first)
- `-applymat in.json <target>` — apply a saved matrix (no registration): reslice the moving image onto the `<target>` grid using `fixed_to_moving`. Because the matrix is world-mm→world-mm, the target may have **any resolution / FOV / origin** sharing the fixed world frame (e.g. save at 2 mm, apply onto a 1 mm grid). Usage: `allineate <input> <target> -applymat in.json <output>`. Use `-final NN` for label/atlas volumes
- `-master grid.nii` — **register at the stationary resolution, output at a different one.** A coarse (e.g. 2 mm) template is usually enough for robust linear registration, but downstream processing often wants the result on a finer grid. `-master` reslices the registered moving image onto `grid.nii` — which must **share the stationary world frame** (typically a higher-res version of the same template) — instead of the stationary grid. The fit itself is unchanged (the saved `-savemat` matrix is identical with or without `-master`); only the output sampling differs, and the result is byte-identical to a `-savemat` + `-applymat`-onto-`grid` two-step. Works with both engines. `-final` sets the reslice interpolation (`-final NN` for label/atlas volumes). Example: `allineate T2w.nii.gz avg152T1_2mm.nii.gz -cost fast -master avg152T1_1mm.nii.gz out.nii.gz`. Not combinable with `-applymat`/`-skullstrip`
- `-qwarp` — **nonlinear (deformable) registration**, an exclusive mode: the only accepted form is `allineate <moving> <stationary> -qwarp <output>`. A faithful, attributed public-domain port of AFNI's `3dQwarp -blur 0 3 -source <moving> -base <stationary> -prefix <output>`. This is **only the nonlinear stage** — the moving image must ALREADY be unifized (intensity-inhomogeneity corrected), skull-stripped, affine-aligned, and sampled on the stationary grid (the two inputs must share equal dimensions and an equivalent voxel→world transform; any mismatch, extra option, missing operand, stdin/stdout operand, 4D input, or unusable geometry is a hard error). It writes the warped moving image on the stationary grid. Internally: data-dependent zero-padding, a FWHM-3-voxel source blur, an auto weight from the base, a hierarchical patch-shrinking optimizer (global patch → down to 25-voxel patches) driven by clipped-Pearson similarity with a bulk/shear deformation penalty, and a final WSINC5 reslice. **Equivalent, not bit-identical, to AFNI** (Gaussian/median substitutions for AFNI's GPL routines) — validated at Pearson 0.98–0.996 vs AFNI's reference outputs on 1 mm MNI-space brains. Runtime ≈3.5 min per 1 mm case (multi-threaded); run cases one at a time. Deferred (not yet supported): running/consuming an affine, transform composition, `-iniwarp`, and emitting the `_WARP` displacement dataset. Set `QW_VERB=1` for a per-level progress trace.
- `-reface <shell>` — **anonymize (reface)** a subject scan, emulating AFNI `afni_refacer2.csh -mode_reface`. Form: `allineate <subject> <template> -reface <shell> <output>`. Registers the subject (moving) to the `<template>` (stationary), back-projects the template-space **shell** onto the ORIGINAL subject grid, and composites an anonymized image: shell voxels `>0` overwrite the subject with an artificial face (brightness-matched to the subject and edge-blended), `==0` keep the subject, `<0` are zeroed. The output stays on the subject's own grid. The template is the *stationary* image so its registration weight can be supplied with `-weight` (as AFNI does). Works with both engines (fast default, or `-cost hel`/`lpa`/…); `-robustfov` is fast-engine only. The shell is a 3D template-space image (the AFNI 4D `afni_refacer_shell_sym_2.1` subbrick `[1]` = the reface/face shell). This is a **clean-room** reimplementation of the AFNI recipe (the edge blend uses niimath's own Gaussian, not AFNI's `3dBlurInMask`). Not combinable with `-skullstrip`/`-applymat`/`-master`/`-savemat`/`-qwarp`/`-com`/`-sym`; the shell resampling is fixed nearest-neighbor so `-final`/`-interp`/`-fill` do not apply. Example: `allineate T1.nii.gz MNI152_2009_SSW.nii.gz -weight MNI152_2009_SSW_weight.nii.gz -reface refacer_shell_face.nii.gz refaced.nii.gz`
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
allineate T1.nii.gz MNI152_T1_1mm -cost fast out         # default adaptive fast capture
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

Piped input uses raw NIfTI-1, matching the shared niimath I/O layer. Only the moving image can come from stdin — every secondary input (the stationary image and the `-skullstrip` mask, `-master` grid, and `-weight` image) must be a file. Piped **output** is uncompressed NIfTI-1 (pipe through `gzip` for compression). Piped I/O supports NIfTI-1 only.

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
