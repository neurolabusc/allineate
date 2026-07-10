Guidance for AI coding agents working in this repository.

## Project Overview

Standalone C library + CLI for affine (12-DOF) NIfTI registration, adapted from AFNI's
3dAllineate (public domain). This repo is a **minimal demo/test-bed for the allineate functions
in `niimath`** (`/Users/chris/src/niimath`). CLI-first, serial-only.

## Upstream sync with niimath (IMPORTANT)

Five files are shared with `niimath/src/`. Divergence status:

| File | vs niimath |
|------|-----------|
| `allineate.c` / `allineate.h` | **DIVERGED** — auditor fixes + all demo features (`-sym`/`-symd`/`-symb`/`-sagseed`/`-com`/`-savemat`/`-applymat`/`-dark_automask`/`-zoom`) |
| `powell_newuoa.c` | **DIVERGED** — workspace teardown/allocation hardening |
| `nifti_io.c` / `nifti_io.h` | **EXTERNAL** (see policy) — demo piping + `nifti_image_write_status` |
| `main.c`, `miniCoreFLT.c/h` | demo-only (not in niimath) |

**niimath compatibility is verified**: dropping the five shared files into `niimath/src` builds
cleanly under niimath's flags and runs. The public API is unchanged — `nii_allineate`/`nii_deface`/
`nii_reslice_affine` keep their signatures (demo-only options live in `al_opts` fields that
`nii_allineate` ignores; only `main.c` reads them), and `nifti_image_write()` stays a `void`
wrapper over `nifti_image_write_status()`.

### POLICY: `nifti_io.c` / `nifti_io.h` are an EXTERNAL library — do not invest in it
Maintained by another team that **rejects our defensive/hardening changes**. Treat as a vendored
dependency: **do not harden or refactor**; when an audit flags something there, **note it and move
on**. The minimal fixes already applied (piping, `nifti_image_write_status`, the `nbyper*nvox`
overflow guard, PIGZ/ZSTD guards) stay — no further changes. Spend quality effort on the
demo-owned code (`allineate.c`, `main.c`, `miniCoreFLT.c/h`, `allineate.h`).

Known-and-owned-elsewhere (note-and-move-on): PIGZ `popen` shell-injection (`#ifdef PIGZ`, never in
default build, now filename-guarded); `nifti_io.c` reads srow/quaternion **without finiteness
validation** — the source of the NaN/Inf matrices that reach `-robustfov`/registration; the demo
defends via `al_mat44_usable()` / `mc_dmat44_usable()`.

### Restoring parity
`allineate.c`/`powell_newuoa.c`/`nifti_io.c/.h` carry hardening fixes not yet upstream. Before a
future `cp` re-sync from niimath, **port these fixes upstream first** so the re-sync doesn't revert
them. A recorded upstream commit + per-file hashes (`make parity`) would replace the drift-prone
prose above — a good future addition (the `parity` target does **not** exist yet, despite older prose
implying it).

### Keep the build tree self-contained
`main.c` `#include`s `miniCoreFLT.h`, so a clean clone must carry all five sources
(`main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c` + their headers) plus `Makefile` and
this `AGENTS.md`. Keep them all tracked; `benchmark/` stays **ignored** (see Test / Validate). Verify a
clean-clone `make` before publishing.

## Persistent decisions (do not re-litigate)
- **`-ffast-math -fno-finite-math-only`** is required (niimath's `AFASTMATH`; now our Makefile
  `CFLAGS` + all documented commands). Plain `-ffast-math` implies `-ffinite-math-only` → the
  compiler assumes no NaN/Inf, which **breaks the non-finite filtering** and warns on `INFINITY`.
  This is why finiteness is checked with magnitude guards (`al_finitef`: `-FLT_MAX ≤ v ≤ FLT_MAX`),
  not `isfinite()`.
- **DECLINED** unifying the sform/qform selector into miniCoreFLT: `al_image_xform()` is float and
  fail-closed (error return); `nifti_robustfov()` threads a double `nifti_dmat44` and wants a pixdim
  fallback. A shared helper would force a double↔float round-trip or cross the self-contained
  miniCoreFLT TU boundary to save ~4 lines. Kept a local `mc_dmat44_usable()` mirror.
- **DEFERRED** splitting `al_opts`: the CLI-workflow fields (`skullstrip`/`robustfov`/`sym`/
  `sagseed`/`zoom`/`savemat`/`applymat`/…) are interpreted by `main.c`, not `nii_allineate()`; a
  direct API caller setting them gets a silent no-op. Grouped under a "NOT interpreted by the
  engine" banner in `allineate.h` rather than split (churn for a CLI-first demo).
- **DEFERRED** making the source automask lazy: `al_automask` runs unconditionally in `al_register`
  even when `-source_automask` is off (feeds only the coverage log in the default build) — a
  bounded, freed, ms-scale O(nvox) transient. Making it lazy means distinguishing "intentionally
  absent" from OOM across the coarse-pass 2× downsample save/restore (4 sites); not worth the
  regression risk. A future optimization, not a bug.
- **`-unifize`** (bias-field correction) DEFERRED: within-modality, intensity bias is roughly
  symmetric, so it barely shifts the `-com` centroid or the `ls`/symmetry costs. Only helps
  cross-modal; not planned unless a concrete need appears.
- **DEFERRED** carrying the mirror-fit **cost** out of `al_register` for `-symb` selection: the
  selector uses correction *rotation* only; a weak near-identity minimum could beat a better
  larger-correction fit. Low-probability tail case; threading cost out of `al_register` (whose
  callers all discard it) is more plumbing/regression risk than warranted. Tie-preference for the
  world frame (above) covers the observed harm.
- **ACCEPTED (note-and-move-on)** two low CLI footguns in `main.c`: (1) `read_affine_json`'s number
  scanner skips `nan`/`inf` tokens rather than parsing them, but `nii_apply_affine` re-validates with
  `al_mat44_usable` and cleanly rejects a non-finite matrix — robustness note only; (2) `-robustfov`
  with an optional numeric arg swallows a **purely numeric** output filename (`200`, no extension) as
  the FOV; any name with an extension (`200.nii`) is safe. Not fixed — changing the lookahead risks
  the common case for a contrived input.
- **Regression gate now exists** (`test/test_regression.py`, `make test`, committed): a deterministic
  pass/fail suite that builds **synthetic** (non-facial) fixtures on the fly and asserts the hardened
  paths — non-finite `-com` → finite header, 4D rejection, save/apply round-trip (NCC>0.999),
  scaled-float32 deface fill, `-symb` axis-aligned tie (world frame + code preserved), and a
  qform-only-usable reader-repair case. **Exits nonzero on any failure** (2 = missing python deps);
  `benchmark/benchmark.py` likewise exits nonzero on hard fails/timeouts. Extend it when adding a
  feature. Still-open: a **`make parity`** hash check vs the niimath upstream (needs a recorded
  upstream commit ref) and broader coverage (e.g. `-robustfov`, `-zoom`).

When reviewing/auditing, **re-verify every claim against the live source before acting** — treat
findings as hypotheses, not facts; an independent verification agent is a good tool for this.

### Shareability: `examples/` history is fine; only `benchmark/` is sensitive
The old `examples/` images (`T1_head*`, `fmri`, MNI152 templates) were removed as cleanup (the repo
pivoted to the local `benchmark/` harness), **not** for privacy — they are fine to share, so their
blobs remaining in Git history are harmless and no history rewrite is needed. The **only**
non-shareable data is the **`benchmark/`** tree — some edge-case inputs contain facial information, so
it is git-ignored: do NOT `git add -f` it or ever commit it, and do not rely on it for a clean-clone
build/CI. A shareable subset (de-faced/synthetic) is a future addition.

## Build

`make` = optimized release (default, always rebuilds — whole-program, ~2 s). `make sanitize` = ASan
(heap-overflow/use-after-free — **NOT leaks** on Apple Silicon, LeakSanitizer is unavailable).
`make profile` = `-DAL_PROFILE` per-stage timing. `make clean`. Options: `CNAME=gcc-15`, `OMP=0`,
`ZSTD=1` (experimental). Dependencies: zlib (required); libomp for OpenMP with Apple Clang
(`brew install libomp`); zstd optional.

**Leak testing (Apple Silicon):** ASan does NOT catch leaks — use the plain/release binary:
`MallocNanoZone=0 leaks --atExit -- ./allineate <mov> <fix> out.nii.gz`.

Manual compile (Apple Clang; swap the compiler for Homebrew LLVM/GCC or Linux GCC; add `-DAL_PROFILE`
for the profiling build):
```bash
clang -O3 -ffast-math -fno-finite-math-only -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
  -L/opt/homebrew/opt/libomp/lib -lomp -DHAVE_ZLIB \
  main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c -lz -lm -o allineate
```
Apple Clang + libomp is ~20% faster than GCC-15 on ARM. Always `-O3` unless debugging.

## Test / Validate

**Correctness gate (committed, shareable):** `make test` → `test/test_regression.py` builds synthetic
(non-facial) fixtures and asserts the geometry/preprocessing/affine paths, **exiting nonzero on any
failure**. Run this before a release. The benchmark below is a separate speed/quality measurement.

> **NOT SHAREABLE YET — the entire `benchmark/` tree is git-ignored (`.gitignore`: `benchmark/`).**
> Some edge-case inputs contain **non-shareable facial information**, so the whole folder is kept
> local for now. **Do not `git add -f` it.** The plan is to add a **shareable subset** (de-faced /
> synthetic inputs + templates + masks) in the future so the benchmark can ship with the repo. Until
> then, the benchmark exists only on the maintainer's machine and cannot be part of a clean-clone
> build or CI. Do not forget this before publishing.

`benchmark/benchmark.py` registers every moving image in `benchmark/inputs/` to every template in
`benchmark/templates/` with **default settings** and reports time / peak-RAM / NCC / masked-NCC
(brain-mask-restricted; masks in `benchmark/masks/`) as copy-pasteable markdown — see
`benchmark/README.md` for the method and the recorded baseline. Outputs go to `benchmark/outputs/`
(also ignored). It measures speed/quality, not correctness regressions — a committed regression suite
(with shareable fixtures) is still a worthwhile future addition.

Benchmark with `-p 10` (or `OMP_NUM_THREADS=10`); `-p 0` = all cores, `-p 1` = serial. Options may
appear on **either side** of the output filename. The `Registration completed` line reports the
coarse/fine split (always measured, independent of `-DAL_PROFILE`).

## Architecture

- **allineate.c** (~4000 lines) — registration engine: BLOK local correlation (TOHD), cost
  functions (Hellinger/Pearson/lpc/lpa), coarse→fine twopass optimization, affine warp + interp
  (NN/trilinear/tricubic). OpenMP with thread-local workspaces. `-DAL_PROFILE` for per-stage timing.
- **allineate.h** — public API + `al_opts` struct + inline enum-name parsers.
- **nifti_io.c/h** (~2100 lines) — NIfTI-1/2/Analyze reader/writer, gzip/zstd, mat44/quaternion
  math. Demo adds stdin/stdout piping and inherits niimath's `nifti_sync_sform_from_qform()`.
- **powell_newuoa.c** (~2400 lines) — Powell NEWUOA derivative-free optimizer (f2c translation).
- **miniCoreFLT.c/h** — minimal port of selected niimath `coreFLT.c` operators (`nifti_robustfov`,
  `nii_ensure_float32`). Add further coreFLT operators here rather than cloning niimath.
- **main.c** — CLI: `<moving> [stationary] [opts] <output>` (stationary optional; omit for
  preprocessing-only). Opts parse on either side of the output.

## Key Design Details

- **Affine selection (`al_image_xform()`):** NIfTI precedence — prefer sform when `sform_code >=
  qform_code`, else qform; if the preferred form is unset/**bogus** (all-zero/degenerate, via the
  scale-invariant `|det|/‖columns‖` test in `al_mat44_usable()`), fall back to the other usable
  form. `al_register` **errors out** ("no valid sform or qform") when neither is usable — no silent
  pixdim-frame synthesis. `-sym`/`-com` keep a documented pixdim-centered fallback (template-free,
  so a hard error is inappropriate). Interacts with `nifti_sync_sform_from_qform()` (qform-sync
  below), which may already have filled a degenerate sform from a valid qform at read time.
- **Cost functions:** `ls` (Pearson — fast, same-modality), `hel` (Hellinger — **default**,
  cross-modal), `lpc`/`lpa` (local Pearson — cross-modal, use with `-source_automask`). Hellinger/MI
  bin intensities to ~64 levels. `-DAL_LPC_MICHO` = lpc+ZZ combined variant.
- **Optimization:** twopass coarse→fine (Powell/NEWUOA). Coarse pass **downsamples the source 2×**
  (cache coherency; the downsampled source usually fits in cache — so it dominates runtime but is
  compute-bound), then restores full res for refinement. OpenMP parallelizes grid search + candidate
  refinement; the fine final Powell is sequential. Under `-DAL_PROFILE` the fitter reports a
  **warp(interp) vs cost split** (~50/50) — the data to reason about interpolation cost.
- **Sparse sampling (AFNI-matching):** when `nvox_src < nmask`, fine-pass match count `ntask =
  sqrt(nvox_src·nmask)` sampled at 47%; else `ntask = nmask`. Avoids oversampling for cross-res
  registration. `-DAL_MATCH_ALL` uses all mask points (slower).
- **Interp knobs (two separate stages):** `-interp` = fine-pass **matching** interp (default
  linear, matching AFNI — already trilinear, not cubic). `-final` = **output warp** interp (default
  cubic for registration, linear for `-skullstrip`); the "Applying final warp with cubic" line is a
  one-time step (~ms), controllable via `-final linear`/`-linear`/`-nearest`.
- **`-warp`:** `sho`(3)/`shr`(6)/`srs`(9)/`aff`(12, default). Params beyond the DOF are fixed
  (`fixed=2`). `al_register` validates `warp_dof ∈ [1,12]` when `free_mask==NULL` (a negative value
  would write OOB before `params[0]`).
- **Autoweight:** base-image weights — clip outliers at 3× median of nonzero (AFNI), Gaussian blur,
  threshold at 5% of max, normalize to [0,1]. Applied to all costs.
- **Source is modified in-place** during registration (data replaced, dims → base grid).
- **`-skullstrip <mask>`:** via `nii_deface()` — registers the stationary **to** the template (base
  = template, the well-posed direction), **inverts** the transform, warps the template-space mask
  onto the stationary grid, zeros non-brain to the image minimum. (Registering template→stationary
  mislocates the mask — this direction is niimath's fix.) Consumes the mask's data; caller frees the
  `nifti_image`.
- **Plain-float32 conversion (`al_ensure_float32`):** `nii_deface()` and
  `nii_apply_deface_mask()` now convert via this local static helper instead of an open-coded
  `datatype != DT_FLOAT32` block. It uses the **plain-float32 predicate** (`datatype==FLOAT32` AND no
  pending `scl_slope`/`scl_inter`, matching `aim_alias`/`nii_reslice_affine`): a float32 image with a
  **pending scale** is still converted (via `nii_to_float`, which applies the scale), so the mask
  fill reads *physical* intensities — a negative `scl_slope` can no longer invert min↔max and paint
  masked voxels bright. It also sets **`swapsize`** and clears `cal_min/cal_max` (the old blocks left
  `swapsize` stale). Kept **local to allineate.c** (not `miniCoreFLT.c`'s `nii_ensure_float32`) so
  the file stays a clean niimath drop-in — miniCoreFLT is demo-only and must not be `#include`d here.
  `al_adopt_geometry()` (data freshly built, not converted) got the same `swapsize`/`cal_*` fix inline.
- **Header-fold helpers (shared):** `al_fold_correction_into_header(nim, C, S)` writes `newS = C·S`
  to the sform (always) and to the qform **only when the input carried a *usable* coded qform**
  (`qform_code>0` AND `al_mat44_usable`; else the coded qform is dropped). `al_write_sform` also
  syncs `pixdim` to the frame's column norms — no-op for rigid folds, correct for a `-zoom` scale.
  Used by `-com`/`-sym`/`-sagseed`.
- **`-com`** (`nii_center_of_mass()`): header-only origin reset — intensity-weighted brightness
  centroid (same estimator as `-cmass`) → world via the selected `S` → fold `translate(−centroid)`
  so the centroid lands at world (0,0,0). Runs after `-robustfov`, before `-sym`. The estimator
  (`al_center_of_mass`) **filters non-finite voxels** (`al_finitef`, not `isfinite` — `-ffast-math`):
  NaN/±Inf reach the engine unvalidated and, since `NaN <= 0` is false under fast-math, a single NaN
  would poison the accumulator and fold NaN into the sform/qform/pixdim. It keeps the positivity
  (`v > 0`) brightness-centroid filter and the geometric-center fallback when no finite positive mass
  remains. (Feeds `-cmass` and `-sym` too — fix once here, not per caller.)
- **`-sym`** (`nii_symmetry()`, midsagittal alignment): registers the image to its world-X **mirror**
  (`ls`, 6-DOF, cmass, shift clamped to ¼ FOV via the transient `al_shift_max_override`), extracts
  the pure world transform `T`, forms the half `H = T¹/²` (quaternion half-angle), correction `C =
  H⁻¹`. **Direction proven empirically, not on paper** (a symmetric image shifted +10 mm in world X
  → −10 mm correction). Standalone reslices data symmetric about X=0; pre-step folds `C` into the
  header. Degenerate half → identity+warn; fit at the ±30° guardrail → warn but apply.
  - **Frame variants** (`deoblique` param): **`-symd`** first `al_deoblique_frame()`s (snaps the
    sform to the nearest signed-permutation × pixdim, grid-center fixed) — treats the **voxel grid
    as anatomical**. The snap finds the **nearest orientation-preserving signed permutation** by
    scoring all six permutations with the **handedness constraint folded into the score** (do NOT
    revert to "pick max unsigned then flip" — that is not globally nearest, since a strong unsigned
    match with the wrong handedness can beat a slightly weaker already-proper one): a permutation
    whose natural per-column signs give the wrong determinant sign must flip one column to stay
    proper, costing `2×` that column's weakest alignment confidence, so its penalized score is
    `unsigned − 2·weakest_conf`; the global max over penalized scores wins.
    **`-symb`** runs both frames and keeps the **smaller correction rotation** (the frame whose X=0
    is already closer to the MSP = the native anatomical one; costs are near-tied so rotation is the
    discriminator), but **prefers the ORIGINAL world frame on ties** — de-oblique must beat the world
    frame by more than `AL_SYMB_ROT_TOL_DEG` (0.5°) to be chosen. Without the tolerance an already
    axis-aligned image (identical rotations) switches to de-oblique and needlessly rewrites the header
    (e.g. MNI `sform_code`/`qform_code` 4 → scanner-anatomical 1) for zero geometric gain.
    Rolls the header back on the losing/failing frame. **Known limitation (accepted):**
    the selector still compares only correction *rotation*, not fit *cost* — a weak near-identity
    minimum could in principle win over a materially better larger-correction fit. Carrying the fit
    cost out of `al_register` is deferred (more plumbing/regression risk than the tail-case warrants).
  - **Gotcha (infant scans):** an oblique sform makes a *voxel*-symmetric head ~12° tilted in
    *world* space, so plain `-sym` correctly aligns the MSP to world-X and thereby **rolls** it
    ~10°. It is the oblique sform, not the cost or padding — `-symd`/`-symb` fix it by working in
    the voxel-anatomical frame.
- **Expose the affine** — world-space, grid-independent:
  - **`-savemat out.json`**: records the fitted **world-mm FIXED(base)→MOVING(source)** affine into
    a process-global (via `nii_last_affine()`, so `nii_allineate`'s signature/niimath drop-in are
    unchanged); `main.c` writes JSON (`fixed_to_moving` + inverse `moving_to_fixed` + cost/dof/
    filenames). Requires a stationary image; rejected with `-skullstrip`. **The output image
    filename may be omitted** to save only the matrix.
  - **`-applymat in.json <target>`** (`nii_apply_affine()`): reslice the moving image onto `target`'s
    grid via `gam = S_input⁻¹ · fixed_to_moving · S_target`. Since the matrix is world-mm, `target`
    may have **any** resolution/FOV/origin sharing the fixed frame. Standalone mode (no registration).
  - **Caveat (both):** if `-com`/`-sym`/`-sagseed`/`-zoom` folded a correction into the moving
    header, the matrix is relative to that *seeded* header — so `-applymat` reproduces a **plain**
    registration exactly (NCC 1.0) but not a pre-stepped one applied to the original input.
- **`-sagseed`** (`nii_sagseed()`, in-MSP rigid seed): the complement of `-sym`, **on by default
  when `-sym` runs with a template** (`-nosagseed` disables). Runs after the `-sym` pre-step: a
  3-DOF-**constrained** `al_register(moving→template)` freeing **only** the MSP-preserving isometries
  {`wpar[1]` y-shift, `wpar[2]` z-shift, `wpar[4]` pitch} — the rigid DOF `-sym` is blind to. Uses
  the new **`al_register` `free_mask[12]`** param (NULL → legacy contiguous-prefix `warp_dof`;
  non-NULL → arbitrary free subset). Correction `C = P⁻¹` (**full inverse — no half**, it's a direct
  fit, not the mirror trick) folded into the header. DOF-split proof: on an MSP-aligned phantom a
  known {Y,Z,pitch} is recovered exactly while {X,roll,yaw} yields ~0.
- **`-zoom`** (abnormal size, e.g. infant vs adult template — the default 12-DOF scale range is only
  0.711..1.406, too narrow for ~0.69×). Two opt-in effects (ordinary fits byte-identical): **(1)**
  the `-sagseed` seed adds param 6 as a **global isotropic** zoom (the transient `al_zoom_isotropic`
  ties y,z-scale to x-scale in `GA_setup_affine`; range widened to [0.5,2.0]), pre-sizing the moving
  image; **(2)** the main fit gets `relax_scale=1` → scale range 6,7,8 widened to [0.5,2.0]
  (rotation/shear/shift regularization unchanged). Caveat: may hurt modalities with little scalp
  signal vs a T1 template (EPI/DWI).
- **`-dark_automask`:** drops a matched pair from the cost when the base OR warped-source value is at
  that image's **darkest value** (per-image finite minimum — a hardcoded 0 would be wrong for signed
  CT). In `GA_scalar_fitter`: a per-eval thread-local `tl_weff[]` zeroes those pairs' weight (the
  cost functions already respect per-point weight — no new loop); falls back to unmasked weights when
  `< npt/4` (capped 64) survive. **Caveat:** pure exclusion has a "shrinking-overlap" failure mode
  (max correlation over an ever-smaller overlap) — mitigated by the autoweight + survivor floor;
  AFNI's `-source_automask` instead noise-fills. Off by default.
- **Piping (`-`, demo-only):** moving image from stdin / output to stdout. Stdin slurped once into a
  seekable in-memory buffer (auto-inflates gzip), bounded by `NII_STDIN_MAX` (16 GiB). **Only the
  moving image** may be stdin (else the cache is reused, reading the same image twice). Output to
  stdout is uncompressed NIfTI-1.
- **qform-sync (from niimath):** `nifti_sync_sform_from_qform()` runs on every read, filling an
  absent/degenerate sform from a valid qform (so a qform-only image emits a synthesized sform on
  write). Never overwrites a valid sform.
- **Thread safety:** the engine uses process-global (`gstup`, `aff_before/after`,
  `al_shift_max_override`, `al_zoom_isotropic`, `g_last_*`) and thread-local workspaces. **Serial-
  only** — do not call registration concurrently from multiple host threads.

## miniCoreFLT — porting niimath coreFLT.c operators

Clone niimath `coreFLT.c` operators here rather than pulling in full niimath. Standard adaptations:
`flt`→`float`, `DT_CALC`→`DT_FLOAT32`, `staticx`→non-static (declared in `miniCoreFLT.h`),
niimath's `xform()`→pixdim-scaled identity fallback, `printfx`→`fprintf(stderr,…)`.

- **`-robustfov [mm]`** (default 170): crops the moving image top-of-head-down (FSL robustfov
  clean-room port). Shifts a **usable** coded sform/qform (`code>0` AND `mc_dmat44_usable()`) so kept
  voxels retain world coords; a coded-but-unusable (NaN/singular) form is **dropped** (`code=0`)
  rather than shifted into a NaN header. Axis selection mirrors `al_image_xform` precedence.
  `main.c` applies it to the moving image **before** registration (chained steps read the updated
  geometry). Requires float32 (`nii_ensure_float32()` first). `-DROBUSTFOV_VERBOSE` for diagnostics.
