Guidance for AI coding agents in this repository.

Standalone C library + CLI for affine (12-DOF) NIfTI registration, adapted from AFNI's
3dAllineate (public domain). A **demo/test-bed for the allineate functions in `niimath`**
(`/Users/chris/src/niimath`). CLI-first, serial-only. When auditing, **re-verify every claim
against the live source** — treat these notes as hypotheses.

## Upstream sync with niimath

Shared with `niimath/src/`: `allineate.c/.h` and `powell_newuoa.c` (DIVERGED — demo features +
hardening), `nifti_io.c/.h` (EXTERNAL). `main.c`, `miniCoreFLT.c/.h`, `coreg_fast.c/.h` are demo-only.
The shared files drop into `niimath/src` and build; the public API is unchanged (`nii_allineate`/
`nii_deface`/`nii_reslice_affine`; demo-only options live in `al_opts` fields that `nii_allineate`
ignores and only `main.c` reads). **Before a `cp` re-sync from niimath, port the hardening fixes
upstream first** so they aren't reverted.

**POLICY — `nifti_io.c/.h` is EXTERNAL: do not harden or refactor it.** Another team rejects our
defensive changes; note audit findings there and move on. Known-and-owned-elsewhere: `nifti_io.c`
reads srow/quaternion without finiteness validation (the source of NaN/Inf matrices — the demo
defends via `al_mat44_usable`/`mc_dmat44_usable`); PIGZ `popen` (never in the default build).

## Build / test

- `make` = optimized release (`-O3 -ffast-math -fno-finite-math-only`, always rebuilds).
  `make sanitize` = ASan (heap/UAF). `make profile` = `-DAL_PROFILE` per-stage timing.
  Knobs: `CNAME=gcc-15`, `OMP=0`, `ZSTD=1`. Deps: zlib; libomp for OpenMP with Apple Clang.
- **`-ffast-math -fno-finite-math-only` is REQUIRED** (niimath's AFASTMATH). Plain `-ffast-math`
  implies `-ffinite-math-only` → the compiler assumes no NaN/Inf → breaks non-finite filtering. So
  finiteness uses magnitude guards (`al_finitef`: `-FLT_MAX ≤ v ≤ FLT_MAX`), NOT `isfinite()`.
- **Apple Silicon: ASan can't detect leaks and DEADLOCKS with libomp** (see `~/.claude/CLAUDE.md`).
  `make sanitize` builds serial (no libomp) on purpose. Leak-test the RELEASE binary:
  `MallocNanoZone=0 leaks --atExit -- ./allineate <mov> <fix> out.nii.gz`.
- **`make test`** is the correctness gate: `test/test_regression.py` (synthetic geometry/preprocessing/
  affine fixtures + fast-engine capture, cross-modal recovery, `-cmass`/`-nocmass` behavior, `-master`
  equivalence, and a C-API NULL-opts crash test) + `test/test_coreg_fast.py` (generated-phantom
  `-cost fastcr` capture recovery). Exits nonzero on any failure; run before a release and extend it
  when adding a feature. Results must stay bit-identical at `-p 1` vs `-p N`.
- **Self-contained**: a clean clone needs the six sources (`main.c allineate.c nifti_io.c
  powell_newuoa.c miniCoreFLT.c coreg_fast.c` + headers), tests, `Makefile`, and this file. The gate
  generates its own fixtures.
- `benchmark/benchmark.py` runs the `allineate` and `fast` engines (and AFNI `3dAllineate` with
  `--afni`) at 1 and N threads, scoring quality with a cross-modal-valid **Hellinger-affinity** metric.
  Descriptive (speed/quality only), not a correctness gate. See `benchmark/README.md`.

## Fast engine (`-cost fast` / `-cost fastcr`) — `coreg_fast.c/.h`

Second, explicitly-selected engine; the default stays allineate. Independent SPM/FLIRT-**inspired**
clean-room implementation (never derived from `niimath/src/GPL/`). Hierarchical 8→4→2 mm pyramid,
cheap overlap-aware affine-vs-COM initialization selection, then one coarse search. **Costs
(`CF_COST_*`):** default `HEL` (clean-room
Hellinger-affinity — Bhattacharyya coefficient of the joint 2D histogram vs its marginals,
deterministic chunked reduction so `-p1==-pN` bit-identical); plus `CR` (correlation ratio) and `LS`
(Pearson). HEL is far better than CR on real cross-modal images and is the default (CLI `-cost fast` =
HEL, `-cost fastcr` = CR; `main.c` sets `al_opts.fast` to `AL_ENGINE_FAST_HEL`/`_CR`). **Caveat:** HEL
is imprecise on smooth synthetic phantoms (near-flat MI landscape), so the synthetic capture suites use
CR; HEL is for real sharp-edged images. Returns world-mm `fixed_to_moving`; does not mutate inputs;
serial-only (`g_cf` global). `make profile` prints per-level timing (zero release overhead).

**Do-not-regress gotchas (each was a real bug or hard-won result):**
- **Thread-count-independent reduction**: samples split into a FIXED `CF_CR_NCHUNK` (64) chunks
  combined in fixed order → `-p 1`==`-p N` bit-identical. Do NOT use per-thread-count blocks (NEWUOA
  amplifies FP regrouping past the 0.05 mm floor).
- **Hierarchical pyramid**: each coarser level is built from the next-finer one (quadrature-added
  blur), not re-blurred from full-res.
- **Level dims cover the source FOV** (`ceil(n·vox/sep)`, not round). Bound the world-FOV product
  before the `int` cast (a finite-but-extreme sform/pixdim would be UB), and reject a level > 8× the
  source voxel count (a coarse-pixdim source up-sampled to isotropic can be ~8 GB while < INT_MAX).
- **Partial-FOV overlap policy**: HEL/CR statistics use only fixed-foreground samples that map
  inside the moving FOV. Treating an acquisition boundary as moving background biases a cropped
  scan toward scale/shear transforms that pull unavailable anatomy into the grid (the FLIRT Fig-1
  problem). A 10%/16-sample overlap floor still rejects degenerate fits. LS retains its historical
  background fill. Keep the fixed 64-chunk reduction and skip out-of-FOV samples inside each chunk
  so `-p1==-pN` remains bit-identical.
- **Initialization selection is cheap and overlap-aware**: default mode scores the supplied-affine
  and exact `-com` starts once at 8 mm, maximizing `(1 - cost) * covered_fixed_foreground`, then runs
  the coarse orientation search and full descent for ONLY the winner. Do not multiply the minimized
  cost by overlap (that rewards low overlap). `-com` is a strict already-recentered override and
  `-nocmass` is a strict supplied-affine override; neither auto-selects. The initial MICCAI coverage
  was FLAIR 74.0% affine vs 97.6% COM and T1 59.7% vs 96.2%; after coarse refinement, overlap alone
  would incorrectly prefer the visually bad T1 affine branch (99.982% vs 99.954%), so selection must
  retain the statistical-dependence term rather than maximize overlap alone.
- **Translation guardrail is ±128 mm, not ±64 mm**: oblique/reoriented MICCAI FLAIR/T1 headers need
  about 77–78 mm of rigid-equivalent translation and produce COM seeds around 85 mm. A ±64 mm bound
  rejects the good seed before scoring, after which HEL can manufacture a low-overlap wrong rotation.
  The coarse optimizer remains local around its seeds, so the wider fail-closed guardrail does not
  expand the discrete search; `trans_large_y` regression-guards an 80 mm capture.
- **Rigid coarse, scale later**: the 8 mm level fits RIGID (scale locked at identity); freeing scale
  there let CR collapse into a spurious ~1.2× isotropic-shrink basin on wide-FOV cross-modal inputs.
  Global scale enters at 4 mm via a discrete bracket ({0.78,0.9,1.0,1.12,1.3}) scored at the rigid
  pose — required so a true envelope-edge scale (e.g. 0.75) is recovered (a continuous refine from
  identity can't reach a 25% downscale). Do NOT re-add a coarse scale scan or drop the 4 mm bracket.
- **Audit-hardened invariants** (regression-guarded): (1) failure is atomic — `*result` written only
  after all levels + optimizer-ok + valid cost + affine validity; a degenerate fit returns nonzero and
  writes no `-savemat`. (2) rejects options it can't honor (`-cost/-warp/-interp/-source_automask/
  -dark_automask`); `-final` is allowed and initialization overrides are honored (`-com` arrives
  already recentered with auto-selection disabled; `-nocmass` disables it in the supplied frame).
  (3) `-savemat` serializes the REAL resolved config, never
  hardcoded defaults. (4) matrix-only (`-savemat`, no output) skips the reslice. (5) `cf_dims_ok`
  validates single-volume 3D on entry. (6) `cf_extract_float` scaling matches `nii_to_float`.
  (7) `max_dof` honored at every stage.
- Configured **only** through `coreg_fast_opts`; ambient `COREG_FAST_*` env vars are intentionally
  ignored so embedding applications and saved matrices are reproducible.

## Architecture (file map)

- **allineate.c** (~4000 lines) — the allineate engine: BLOK/TOHD local correlation, costs (Hellinger
  default / Pearson / lpc / lpa), coarse→fine twopass NEWUOA, affine warp + interp, OpenMP thread-local
  workspaces. The coarse driver is a byte-faithful AFNI port — notably the matching-point floor
  `AL_NPT_MATCH_MIN` MUST stay at AFNI's **98765** (fewer coarse samples make the Hellinger surface too
  noisy and a cross-modal case lands in a wrong basin from the header start).
  **Hardening — histogram-OOM must not score as a win:** `build_2Dhist` is `void`; a malloc failure
  leaves an empty histogram that `al_helmicra` maps to cost `0.0` (below `AL_BIGVAL`), which the
  fail-closed abort would accept as success under sustained OOM. A thread-local `al_hist_oom` flag (set
  ONLY at the two malloc-fail returns, cleared at entry, consumed in `GA_scalar_costfun` → `AL_BIGVAL`)
  routes OOM to the abort. Do NOT set it on the legitimate empty-histogram returns
  (`n<=9`/constant/`ngood==0`) — those must keep returning `0.0` or the default trajectory (and p1==pN
  bit-identity) changes.
- **allineate.h** — public API + `al_opts` + inline enum parsers. Engine selector uses named
  `AL_ENGINE_*` constants (0=allineate, 1=fast/CR, 2=fast/HEL).
- **nifti_io.c/.h** — NIfTI-1/2/Analyze I/O, gzip/zstd, mat44/quaternion. EXTERNAL. Adds demo
  stdin/stdout piping + `nifti_sync_sform_from_qform()`.
- **powell_newuoa.c** — NEWUOA derivative-free optimizer (f2c).
- **miniCoreFLT.c/.h** — ported niimath `coreFLT.c` operators (`nifti_robustfov`, `nii_ensure_float32`,
  `mcf_smooth_gauss` blur). Add coreFLT operators here, not full niimath.
- **coreg_fast.c/.h** — the fast engine (above).
- **main.c** — CLI `<moving> [stationary] [opts] <output>` (stationary optional → preprocessing-only);
  opts parse on either side of the output.

## Key design gotchas

- **Cross-modal weight/clip semantics (the T2w→T1 shrink fix).** On a whole-head cross-modal source the
  allineate engine used to shrink the moving brain (det>1, foreground pushed out of the source FOV)
  where AFNI contracts to fill — the FLIRT Fig-1 overlap degeneracy. The current design, matching AFNI:
  - **Autoweight is BINARY for box-mode costs** (Hellinger/MI/NMI/CR), like AFNI `-autobox`: binarize
    `wght` (→1.0 in-mask) so every in-mask voxel weighs equally. Only `ls/lpc/lpa` use the graded
    intensity weight. A graded weight down-weights the cortical periphery, so pushing it out of FOV
    costs little and the overlap shrinks unpenalized — this was the dominant divergence. Do NOT make the
    weight graded for box costs.
  - **Histogram membership uses `mri_topclip`**, stored separately from the clipate edge-bin clips
    (`aj_topclip`/`bs_topclip`, threaded through `al_helmicra`→`build_2Dhist`). Membership DROPS pairs
    above each image's topclip (bright non-brain a T1 template lacks); edge-binning still uses the
    clipate clips. Do NOT use the lower clipate `ctop` for membership (over-drops, breaks same-modality).
  - **Edge-bin clips are sample-based** (AFNI `set_2Dhist_xyclip`): a per-stage `need_hist_setup` flag,
    consumed by the stage's FIRST **sequential** cost eval which caches the clip; parallel evals read
    the cache. EVERY stage that arms the flag needs a sequential warm-up eval before its parallel
    region, or the shared write races and breaks p1==pN bit-identity.
  - These are gated on a `hist_cost` bool so `ls`/`lpc`/`lpa` skip the extra quantile/warm-up; the
    default Hellinger path is byte-for-byte unchanged. `AL_VERB` prints per-stage clips + pose (0
    release overhead). **Validate any change here on T2w AND T1 AND fmri AND `-skullstrip` + p1==pN
    bit-identity + `make test`.**
- **`al_image_xform()`** (sform/qform selection): prefer sform when `sform_code >= qform_code`, else
  qform; skip a bogus (all-zero/degenerate) preferred form and fall back. Errors out if neither is
  usable (no silent pixdim synthesis) — except `-sym`/`-com`, which keep a pixdim-centered fallback.
- **Serial-only**: process-globals (`gstup`, `g_last_*`, `g_cf`, ...). Never call registration
  concurrently across host threads.
- **Source is modified in-place** during registration (data replaced, dims → base grid).
- **`-com` / centroid**: filters non-finite voxels with `al_finitef` (NOT `isfinite` — under fast-math
  one NaN would poison the accumulator → NaN header). Positivity brightness filter + geometric-center
  fallback.
- **qform-sync**: every read fills an absent/degenerate sform from a valid qform; never overwrites a
  valid sform.
- **`-interp` vs `-final`**: `-interp` = fine-pass matching interp (default linear); `-final` = output
  warp interp (default cubic; linear for `-skullstrip`). Resolved once via `resolve_output_interp()`.
- **`-robustfov`** (FSL clean-room port): crops top-of-head-down; shifts a usable coded sform/qform.
  **Pair with `-cmass` on the allineate engine** — cropping removes the inferior anchor the
  header-started search needs, so without a COM init the fit drifts into an inflated-scale basin. This
  is a search-INIT effect, NOT a geometry bug — do NOT "fix" the crop math or re-add auto-cmass. The
  fast engine is immune (always scores a COM seed).
- **`-skullstrip`** (`nii_deface`): registers stationary→template (the well-posed direction), inverts,
  warps the template-space mask onto the stationary grid, zeros non-brain to the image minimum.
- **Plain-float32 conversion** (`al_ensure_float32`, local to `allineate.c`): a float32 image with a
  pending `scl_slope`/`scl_inter` is still converted via `nii_to_float` so mask fill reads physical
  intensities (a negative slope can't then paint masked voxels bright).
- **`-sym`/`-symd`/`-symb`** (MSP alignment via a world-X mirror fit + half-transform): `-symd` snaps
  the frame to the nearest orientation-preserving signed permutation (handedness folded into the
  score). `-symb` prefers the world frame on ties (de-oblique must beat it by > 0.5°). Compares
  correction rotation only, not fit cost (accepted tail-case limitation).
- **`-sagseed`**: 3-DOF constrained fit (y-shift/z-shift/pitch); on by default when `-sym` runs with a
  template.
- **`-savemat` / `-applymat`**: world-mm `fixed_to_moving`; `-applymat` reslices onto any target sharing
  the fixed frame. Caveat: if `-com/-sym/-sagseed/-zoom` seeded the moving header, the matrix is
  relative to that seeded header (so `-applymat` reproduces a plain fit exactly, not a pre-stepped one).
- **`-master grid`**: register at the stationary resolution, reslice the result onto `grid` (must share
  the stationary world frame) instead of the stationary grid. The fit is unchanged (same matrix);
  output is byte-identical to `-savemat`+`-applymat`-onto-`grid`. The fast engine reslices its unmutated
  `moving` straight onto the grid; the allineate engine mutates `moving` in place, so main.c
  **deep-copies** (`nim_deep_copy`) the moving image after any pre-steps and re-reslices the copy.
  Rejected with `-applymat`/`-skullstrip`.
- **`-dark_automask`**: drops matched pairs at either image's darkest value (per-image min, safe for
  signed CT). Off by default.
- **`-zoom`** (infant/abnormal size): widens the seed + main-fit scale range to [0.5, 2.0]; ordinary
  fits stay byte-identical.
- **Piping** (`-`, demo-only): moving from stdin (slurped to a seekable buffer, gzip auto-inflated,
  ≤ 16 GiB); output to stdout (uncompressed NIfTI-1). Only the moving image may be stdin.

## Clean-room provenance (licensing)

The fast path is designed only from published descriptions (SPM `spm_coreg`; FLIRT, Jenkinson & Smith
2001 PMID 11516708; FreeSurfer, Reuter et al. 2010 PMID 20637289) — **never from `niimath/src/GPL/`**
(hard rule). The only ported code is BSD/public-domain: `mcf_smooth_gauss`/`mcf_blur1d`/
`mcf_transposeXY/XZ` port niimath `coreFLT.c`'s blur/transpose operators to a raw float buffer (niimath
rev `167031ef1d6f2e6c3beb164dbf8e2c23857947b2`).

## Deferred / do-not-re-litigate

Judged not worth the churn/regression risk: splitting `al_opts` (its CLI-only fields are read by
`main.c`, not `nii_allineate`); making the source automask lazy; `-unifize` bias correction; a
`make parity` hash check vs upstream; pass/fail baselines in `benchmark/benchmark.py` (descriptive by
design — the gate is `make test`). Accepted CLI footguns: `read_affine_json` skips `nan`/`inf` tokens
(re-validated by `al_mat44_usable`); `-robustfov` swallows a purely-numeric extensionless output name.
