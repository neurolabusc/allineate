# allineate

Standalone affine (12 DOF) image registration for NIfTI files. Adapted from AFNI's 3dAllineate (public domain).

## Usage

```bash
allineate <moving> <stationary> [opts] <output>
```

Options:
- `-cost XX` — cost function: `hel` (Hellinger, default), `lpc` (local Pearson, cross-modal), `lpa` (abs local Pearson, cross-modal), `ls` (Pearson/least-squares)
- `-cmass` / `-nocmass` — center-of-mass initialization
- `-source_automask` — fill outside source automask with noise (recommended with lpc/lpa)
- `-interp XX` — fine-pass matching interpolation: `NN`, `linear` (default), `cubic`
- `-final XX` — output interpolation: `NN`, `linear`, `cubic` (default)
- `-nearest` / `-linear` / `-cubic` — shortcuts for `-final`

## Build

```bash
# macOS Apple Clang + libomp (fastest on ARM; requires: brew install libomp)
clang -O3 -ffast-math -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# macOS Homebrew GCC + OpenMP
gcc-15 -O3 -ffast-math -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate

# Linux with GCC + OpenMP
gcc -O3 -ffast-math -fopenmp -DHAVE_ZLIB main.c allineate.c nifti_io.c powell_newuoa.c -lz -lm -o allineate
```

## Validation

Benchmark on Apple M4 Max, compiled with Apple Clang 17 + libomp.

### 10 threads (`OMP_NUM_THREADS=10`)

| Test | Command | Time | Peak RAM | NCC (stationary) |
|------|---------|------|----------|-------------------|
| T1 2mm ls | `allineate T1_head_2mm MNI152_T1_2mm -cost ls` | 0.8s | 56 MB | 0.903 |
| T1 1mm ls | `allineate T1_head MNI152_T1_1mm -cost ls` | 7.2s | 367 MB | 0.876 |
| T1 1mm hel | `allineate T1_head MNI152_T1_1mm` | 16.4s | 441 MB | 0.876 |
| T1 1mm hel cmass | `allineate T1_head MNI152_T1_1mm -cmass` | 17.1s | 437 MB | 0.876 |
| fMRI lpc | `allineate fmri T1_head -cmass -cost lpc -source_automask` | 5.6s | 250 MB | 0.438 |

### Single thread (`OMP_NUM_THREADS=1`)

| Test | Command | Time | Peak RAM | Speedup (10T) |
|------|---------|------|----------|---------------|
| T1 2mm ls | `allineate T1_head_2mm MNI152_T1_2mm -cost ls` | 2.9s | 38 MB | 3.6x |
| T1 1mm ls | `allineate T1_head MNI152_T1_1mm -cost ls` | 22.4s | 307 MB | 3.1x |
| T1 1mm hel | `allineate T1_head MNI152_T1_1mm` | 57.4s | 351 MB | 3.5x |
| T1 1mm hel cmass | `allineate T1_head MNI152_T1_1mm -cmass` | 56.7s | 350 MB | 3.3x |
| fMRI lpc | `allineate fmri T1_head -cmass -cost lpc -source_automask` | 10.1s | 147 MB | 1.8x |

Speedups of 1.8–3.6x from 10 threads reflect Amdahl's law: the sequential Powell/NEWUOA optimization limits parallel scaling, while the parallelized grid search and candidate refinement benefit most. Peak RAM is lower single-threaded since per-thread workspace buffers are not allocated.

### Notes

- **NCC (stationary)**: Normalized cross-correlation between output and the stationary/target image. Measures registration quality.
- The fMRI-to-T1 NCC is lower because fMRI and T1 are different modalities with inherently different contrast.
- Default uses `-interp linear` for the fine pass and `-final cubic` for output (matching AFNI). Use `-final linear` for faster output with slight blurring.
- Beware that [AFNI will strip obliquity from an image](https://github.com/afni/afni/blob/72b28ec05a23a50ce8fab70afe0a4a45d1d499cf/src/3dAllineate.c#L6561) while this project retains the NIfTI transform of the stationary image. This can be seen in the example where the fmri image is warped to match the oblique T1_head images. Tools like MRIcroGL and FSL will show the AFNI images as mis-aligned as they read the [NIfTI](https://brainder.org/2012/09/23/the-nifti-file-format/) header s-form.

## Examples

```bash
export OMP_NUM_THREADS=10
allineate T1_head_2mm MNI152_T1_2mm -cost ls ./out/wT1ls_2mm
allineate T1_head MNI152_T1_1mm -cost ls ./out/wT1ls
allineate T1_head MNI152_T1_1mm ./out/wT1
allineate T1_head MNI152_T1_1mm -cmass ./out/wT1cmas
allineate fmri T1_head -cmass -cost lpc -source_automask ./out/fmri2t1
```

Run `python3 validate.py` from the project root to regenerate the benchmark table.
