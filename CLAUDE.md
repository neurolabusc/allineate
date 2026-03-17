# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Standalone C library and CLI for affine (12 DOF) medical image registration of NIfTI files. Adapted from AFNI's 3dAllineate (public domain).

## Build

No build system yet. Compile manually:

```bash
# macOS Apple Clang + libomp (fastest on ARM, requires: brew install libomp)
clang -O3 -ffast-math -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# macOS Homebrew LLVM Clang + OpenMP
/opt/homebrew/opt/llvm/bin/clang -O3 -ffast-math -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# macOS Homebrew GCC + OpenMP
gcc-15 -O3 -ffast-math -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# Linux with GCC + OpenMP
gcc -O3 -ffast-math -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# Profiling build (adds -DAL_PROFILE for per-stage timing to stderr; works with any compiler above)
# Example with Apple Clang:
clang -O3 -ffast-math -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DHAVE_ZLIB -DAL_PROFILE main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate
```

Dependencies: zlib (required), zstd (optional, for .zst files), libomp (for OpenMP with Apple Clang: `brew install libomp`).

On Apple Silicon, Apple Clang 17 + libomp is ~20% faster than GCC 15 (e.g., 25.6s vs 32.3s on the 1mm ls test). Homebrew Clang 22 is comparable. All produce correct results.

Always use `-O2` or `-O3` unless debugging — compile time is negligible compared to validation runtime.

## Test / Validate

Always run benchmarks with `OMP_NUM_THREADS=10`. Run from the `examples/` directory:

```bash
export OMP_NUM_THREADS=10
./allineate T1_head_2mm MNI152_T1_2mm -cost ls ./out/wT1ls_2mm        # fast, use for iteration
./allineate T1_head MNI152_T1_1mm -cost ls ./out/wT1ls
./allineate T1_head MNI152_T1_1mm ./out/wT1
./allineate T1_head MNI152_T1_1mm -cmass ./out/wT1cmas
./allineate fmri T1_head -cmass -cost lpc -source_automask ./out/fmri2t1
```

The `T1_head_2mm` least-squares test is the fastest; use it for rapid iteration. The larger tests are slow.

## Architecture

Four source files, no internal build system:

- **allineate.c** (~3700 lines) — Core registration engine: BLOK-based local correlation (TOHD), cost functions (Hellinger, Pearson, lpc, lpa), coarse-to-fine twopass optimization, affine warp + interpolation (NN/trilinear/tricubic). Uses OpenMP with thread-local workspace buffers. Compile with `-DAL_PROFILE` for per-stage timing breakdown.
- **allineate.h** — Public API: `nii_allineate()` (register), `nii_deface()` (anonymize), `al_opts` struct, CLI argument parsing helpers.
- **nifti_io.c/h** (~2100 lines) — NIfTI-1/NIfTI-2/Analyze 7.5 reader/writer, gzip/zstd support, mat44 math, quaternion↔affine conversions.
- **powell_newuoa.c** (~2400 lines) — Powell's NEWUOA derivative-free optimizer (f2c translation). Used for the non-smooth cost function minimization.
- **main.c** — CLI entry point. Parses `<moving> <stationary> [opts] <output>`, calls `nii_allineate()`, writes gzipped NIfTI output.

## Key Design Details

- Cost functions: `ls` (Pearson, fast, same-modality), `hel` (Hellinger, default, cross-modal), `lpc`/`lpa` (local Pearson, cross-modal, use with `-source_automask`). Compile with `-DAL_LPC_MICHO` for lpc+ZZ combined cost variant.
- Optimization: twopass coarse→fine via Powell/NEWUOA; optional center-of-mass (`-cmass`) initialization. Coarse pass downsamples the source image 2x for the grid search (better cache coherency), then restores full resolution for refinement. Grid search, candidate refinement, and refinement rounds are all parallelized via OpenMP. Fine pass parallelizes candidate evaluation across threads, then runs a final sequential Powell optimization.
- Sparse sampling (matching AFNI): when the source has fewer voxels than the base mask (`nvox_src < nmask`), the fine-pass matching point count is scaled by `ntask = sqrt(nvox_source * nmask)` then sampled at 47%. When `nvox_src >= nmask`, `ntask = nmask` (no scaling). This avoids redundant oversampling for cross-resolution registration (e.g., 2.4mm fMRI to 1mm T1). Compile with `-DAL_MATCH_ALL` to use all mask points instead (slower, marginally more precise for same-resolution images).
- `-interp` controls fine-pass interpolation (default: linear, matching AFNI). `-final` controls output interpolation (default: cubic, matching AFNI). Use `-final linear` for faster output with slight blurring, or `-final NN` for integer-valued data (e.g. atlases). The `-nearest`/`-linear`/`-cubic` flags are shortcuts for `-final`.
- Tricubic warp uses an interior/border split: scanlines are divided into a bounds-checked border region and a fast interior path that skips all CLIP/floorf overhead.
- The source image is modified in-place during registration (data replaced, dims updated to match base grid)
- `nii_deface` registers template→input then warps a mask to zero facial voxels in native space
