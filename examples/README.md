## About

This is a minimal implementation of the AFNI [3dAllineate](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dAllineate.html) tool. It removes AFNI dependencies, removes rarely used functions and provides optimizations.

 - The least-squares cost function (`-cost ls`) is the fastest, but requires the moving and stationary images are the same modality.
 - The default [Hellinger](https://en.wikipedia.org/wiki/Hellinger_distance) cost function works across modalities, but it is much slower.
 - The lpc function developed by [Saad et al.](https://pubmed.ncbi.nlm.nih.gov/18976717/) is useful for [aligning fMRI data to T1 scans](https://pubmed.ncbi.nlm.nih.gov/30361428/).
 - AFNI suggests `-source_automask` for the lpc and lpa cost functions.
 - The `-cmass` argument uses the center of mass of the image to start a search. This can aid situations where the input image has a very different origin than the template, though it can hurt if the image includes excess neck and shoulders.
 - The `-skullstrip` option warps a brain mask from moving space to stationary space, then sets non-brain voxels to the darkest value in the stationary image.

```bash
export OMP_NUM_THREADS=10
allineate T1_head_2mm MNI152_T1_2mm -cost ls ./out/wT1ls_2mm
3dAllineate -base ./MNI152_T1_2mm.nii.gz  -input ./T1_head_2mm.nii.gz -prefix ./afni/wT1ls_2mm.nii.gz  -cost ls
allineate T1_head MNI152_T1_1mm -cost ls ./out/wT1ls
3dAllineate -base ./MNI152_T1_1mm.nii.gz  -input ./T1_head.nii.gz -prefix ./afni/wT1ls.nii.gz  -cost ls
allineate T1_head MNI152_T1_1mm ./out/wT1
3dAllineate -base ./MNI152_T1_1mm.nii.gz  -input ./T1_head.nii.gz  -prefix ./afni/wT1.nii.gz
allineate T1_head MNI152_T1_1mm -cmass ./out/wT1cmas
3dAllineate -base ./MNI152_T1_1mm.nii.gz  -input ./T1_head.nii.gz  -prefix ./afni/wT1cmas.nii.gz -cmass
allineate fmri T1_head -cmass -cost lpc -source_automask ./out/fmri2t1
3dAllineate -base ./T1_head.nii.gz  -input ./fmri.nii.gz  -prefix ./afni/fmri2t1.nii.gz -cmass  -cost lpc -source_automask 
allineate MNI152_T1_2mm T1_head_2mm -cost ls -skullstrip mniMask.nii.gz ./out/T1ls_2mm_mask
```

Benchmark on Apple M4 Max, compiled with Apple Clang 17 + libomp.

### 10 threads (`OMP_NUM_THREADS=10`)

| Test             |  Time   |  AFNI  |
|------------------|---------|--------|
| T1 2mm ls        |    0.8s |  12.9s |
| T1 1mm ls        |    7.2s |  45.3s |
| T1 1mm hel       |   16.4s | 119.1s |
| T1 1mm hel cmass |   17.1s | 111.1s |
| fMRI lpc         |    5.6s |  15.6s |

### Single thread (`OMP_NUM_THREADS=1`)

| Test             |  Time   |  AFNI  |
|------------------|---------|--------|
| T1 2mm ls        |    2.9s |  17.9s |
| T1 1mm ls        |   22.4s |  80.2s |
| T1 1mm hel       |   57.4s | 160.3s |
| T1 1mm hel cmass |   56.7s | 184.0s |
| fMRI lpc         |   10.1s |  21.2s |
