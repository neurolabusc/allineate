/* miniCoreFLT — minimal port of selected niimath coreFLT.c preprocessing
 * operators for the allineate test bed. See miniCoreFLT.h for the contract.
 *
 * nifti_robustfov is a clean-room port of niimath's function of the same name
 * (adapted from BSD/public-domain niimath; the SI-axis detection, variance-based
 * head-top search, and geometry shift are unchanged in intent). Adaptations for
 * this project: niimath's calc type `flt` -> `float`, `DT_CALC` -> `DT_FLOAT32`,
 * niimath's `xform()` fallback -> a pixdim-scaled identity, `printfx` -> stderr.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <inttypes.h>
#include "miniCoreFLT.h"

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* Multiply size_t a*b into *out, returning 1 on overflow (0 on success). Used to
   bound every allocation/index total against a malformed header before use. */
static int mc_mul(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) return 1;
    *out = a * b;
    return 0;
}

/* Nonzero if a NIfTI index->world matrix is usable: every entry finite and the
 * upper-left 3x3 non-singular (scale-invariant |det|/column-norms test). A coded
 * transform with a NaN/Inf row would otherwise sail through the axis-selection
 * below and leave the left-right axis unresolved (lrax == -1), causing an
 * out-of-bounds dim[]/stride[] access. Mirrors allineate.c's al_mat44_usable but
 * kept local (double precision; miniCoreFLT is a self-contained faithful port). */
static int mc_dmat44_usable(nifti_dmat44 M) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            if (!(M.m[i][j] >= -DBL_MAX && M.m[i][j] <= DBL_MAX)) return 0;
    double c[3];
    for (int a = 0; a < 3; a++) {
        c[a] = sqrt(M.m[0][a]*M.m[0][a] + M.m[1][a]*M.m[1][a] + M.m[2][a]*M.m[2][a]);
        if (!(c[a] > 0.0)) return 0;
    }
    double det =
        M.m[0][0]*(M.m[1][1]*M.m[2][2] - M.m[1][2]*M.m[2][1])
      - M.m[0][1]*(M.m[1][0]*M.m[2][2] - M.m[1][2]*M.m[2][0])
      + M.m[0][2]*(M.m[1][0]*M.m[2][1] - M.m[1][1]*M.m[2][0]);
    return fabs(det) / (c[0]*c[1]*c[2]) > 1e-4;
}

/* ---- datatype conversion ------------------------------------------------- */

int nii_ensure_float32(nifti_image *nim) {
    if (nim == NULL || nim->data == NULL) return 1;
    if (nim->datatype == DT_FLOAT32 &&
        (nim->scl_slope == 0.0 || (nim->scl_slope == 1.0 && nim->scl_inter == 0.0)))
        return 0; /* already plain float32 */

    int64_t n = nim->nvox;
    if (n < 1) return 1;
    size_t nbytes;
    if (mc_mul((size_t)n, sizeof(float), &nbytes)) {
        fprintf(stderr, "nii_ensure_float32: image too large (%" PRId64 " voxels)\n", n);
        return 1;
    }
    float *out = (float *)malloc(nbytes);
    if (!out) return 1;

    /* Apply scaling ONLY when scl_slope != 0 and not the identity (slope 1,
     * inter 0). Per the NIfTI spec a zero slope means "no scaling" — the
     * intercept must NOT be added either. Matches cf_extract_float()/nii_to_float(). */
    int do_scale = (nim->scl_slope != 0.0 &&
                    !(nim->scl_slope == 1.0 && nim->scl_inter == 0.0));
    double slope = do_scale ? nim->scl_slope : 1.0;
    double inter = do_scale ? nim->scl_inter : 0.0;
    void *in = nim->data;

    #define CONV(CTYPE)                                                    \
        do { const CTYPE *p = (const CTYPE *)in;                           \
             for (int64_t i = 0; i < n; i++)                               \
                 out[i] = (float)(p[i] * slope + inter); } while (0)

    switch (nim->datatype) {
        case DT_UINT8:   CONV(unsigned char);      break;
        case DT_INT8:    CONV(signed char);        break;
        case DT_INT16:   CONV(short);              break;
        case DT_UINT16:  CONV(unsigned short);     break;
        case DT_INT32:   CONV(int);                break;
        case DT_UINT32:  CONV(unsigned int);       break;
        case DT_INT64:   CONV(int64_t);            break;
        case DT_UINT64:  CONV(uint64_t);           break;
        case DT_FLOAT32: CONV(float);              break;
        case DT_FLOAT64: CONV(double);             break;
        default:
            free(out);
            fprintf(stderr, "nii_ensure_float32: unsupported datatype %d\n", nim->datatype);
            return 1;
    }
    #undef CONV

    free(nim->data);
    nim->data = out;
    nim->datatype = DT_FLOAT32;
    nim->nbyper = 4;
    nim->swapsize = 4;
    nim->scl_slope = 1.0;
    nim->scl_inter = 0.0;
    nim->cal_min = 0.0;
    nim->cal_max = 0.0;
    return 0;
}

/* ---- robust field of view ------------------------------------------------ */

int nifti_robustfov(nifti_image *nim, double fovmm) {
    if (nim == NULL || nim->data == NULL) return 1;
    if (nim->datatype != DT_FLOAT32)
        return 1; /* caller must nii_ensure_float32 first */
    if (!isfinite(fovmm) || fovmm <= 0.0)
        return 1; /* reject NaN/Inf/non-positive (the CLI passes a validated value) */
    /* crop indexing below uses int; reject headers whose spatial dims don't fit */
    if ((nim->nx > INT_MAX) || (nim->ny > INT_MAX) || (MAX(nim->nz, 1) > INT_MAX) ||
        (nim->nx < 1) || (nim->ny < 1))
        return 1;
    int dim[3] = {(int)nim->nx, (int)nim->ny, (int)MAX(nim->nz, 1)};
    double pix[3] = {nim->dx, nim->dy, nim->dz};
    size_t nvox3D, tmp3D;
    if (mc_mul((size_t)dim[0], (size_t)dim[1], &tmp3D) ||
        mc_mul(tmp3D, (size_t)dim[2], &nvox3D))
        return 1;
    if (nvox3D < 1)
        return 1;
    size_t nvoltmp = (size_t)nim->nvox / nvox3D;
    if ((nvoltmp < 1) || (nvox3D * nvoltmp != (size_t)nim->nvox) || (nvoltmp > INT_MAX))
        return 1; /* refuse ragged/overflowing geometry rather than under-allocate */
    int nvol = (int)nvoltmp;

    /* choose transform: evaluate each coded form's usability independently (finite
       and non-singular), prefer the sform when its code >= the qform's, but fall
       back to the *other* usable form before the pixdim scale — mirroring
       al_image_xform() so a bad higher-code qform can't shadow a valid lower-code
       sform. Validating here is essential — a NaN/Inf coded matrix must not drive
       the axis selection below. s_ok/q_ok are reused at the header-shift step so an
       unusable-but-coded form is dropped rather than shifted into the output. */
    int s_ok = (nim->sform_code > 0) && mc_dmat44_usable(nim->sto_xyz);
    int q_ok = (nim->qform_code > 0) && mc_dmat44_usable(nim->qto_xyz);
    nifti_dmat44 M;
    if (s_ok && q_ok)
        M = (nim->sform_code >= nim->qform_code) ? nim->sto_xyz : nim->qto_xyz;
    else if (s_ok) M = nim->sto_xyz;
    else if (q_ok) M = nim->qto_xyz;
    else {
        memset(&M, 0, sizeof(M));
        M.m[0][0] = (pix[0] > 0.0) ? pix[0] : 1.0;
        M.m[1][1] = (pix[1] > 0.0) ? pix[1] : 1.0;
        M.m[2][2] = (pix[2] > 0.0) ? pix[2] : 1.0;
        M.m[3][3] = 1.0;
    }
    /* Compare direction cosines, not raw columns (columns carry voxel scale): on
       anisotropic/oblique data a large voxel size could otherwise dominate. */
    double cn[3];
    for (int a = 0; a < 3; a++) {
        cn[a] = sqrt(M.m[0][a] * M.m[0][a] + M.m[1][a] * M.m[1][a] + M.m[2][a] * M.m[2][a]);
        if (cn[a] <= 0.0) cn[a] = 1.0;
    }
    /* voxel axis whose direction is closest to world Z (superior+) is head-foot;
       among the rest, the one closest to world X (R) is left-right. */
    int siax = 0;
    double bz = fabs(M.m[2][0]) / cn[0];
    for (int a = 1; a < 3; a++)
        if (fabs(M.m[2][a]) / cn[a] > bz) { bz = fabs(M.m[2][a]) / cn[a]; siax = a; }
    int lrax = -1;
    double bx = -1.0;
    for (int a = 0; a < 3; a++)
        if ((a != siax) && (fabs(M.m[0][a]) / cn[a] > bx)) { bx = fabs(M.m[0][a]) / cn[a]; lrax = a; }
    if (lrax < 0) return 1;   /* unreachable once M is validated; guards dim[lrax] indexing */
    int oax = 3 - siax - lrax; /* remaining (anterior-posterior) axis */
    double pol = (M.m[2][siax] >= 0.0) ? 1.0 : -1.0; /* +: increasing index -> superior */
    int nSI = dim[siax];
    double pixSI = (pix[siax] > 0.0) ? pix[siax] : 1.0;
    size_t stride[3] = {1, (size_t)dim[0], (size_t)dim[0] * dim[1]};
    /* central 40% of left-right extent */
    int lo = (int)round(dim[lrax] * 0.30), hi = (int)round(dim[lrax] * 0.70);
    if (hi <= lo) { lo = 0; hi = dim[lrax]; }
    /* per-slice variance of central sagittal slab (volume 0) */
    float *img = (float *)nim->data;
    double *var = (double *)calloc(nSI, sizeof(double));
    if (!var)
        return 1;
    for (int s = 0; s < nSI; s++) {
        double sum = 0.0, sum2 = 0.0;
        long cnt = 0;
        for (int l = lo; l < hi; l++)
            for (int o = 0; o < dim[oax]; o++) {
                double val = img[s * stride[siax] + (size_t)l * stride[lrax] + (size_t)o * stride[oax]];
                sum += val;
                sum2 += val * val;
                cnt++;
            }
        var[s] = (cnt > 0) ? (sum2 / cnt - (sum / cnt) * (sum / cnt)) : 0.0;
    }
    /* Z_top = most superior slice whose variance exceeds 1% of the peak, plus a
       ~2 mm superior buffer to keep the scalp. */
    double mx = 0.0;
    for (int s = 0; s < nSI; s++)
        if (var[s] > mx) mx = var[s];
    double thresh = 0.01 * mx;
    int ztop = -1;
    if (pol > 0) {
        for (int s = nSI - 1; s >= 0; s--) if (var[s] > thresh) { ztop = s; break; }
    } else {
        for (int s = 0; s < nSI; s++) if (var[s] > thresh) { ztop = s; break; }
    }
    free(var);
    if (ztop < 0)
        return 0; /* no signal found; leave image unchanged */
    int buf = (int)round(2.0 / pixSI);
    ztop += (pol > 0) ? buf : -buf;
    if (ztop < 0) ztop = 0;
    if (ztop > nSI - 1) ztop = nSI - 1;
    double nkd = round(fovmm / pixSI);
    if (!isfinite(nkd) || nkd >= (double)nSI)
        return 0; /* requested FOV spans the whole image; nothing to crop */
    if (nkd < 1.0) nkd = 1.0;
    int nkeep = (int)nkd; /* safe: 1 <= nkd < nSI */
    int start, end;
    if (pol > 0) { end = ztop; start = end - nkeep + 1; }
    else { start = ztop; end = start + nkeep - 1; }
    if (start < 0) start = 0;
    if (end > nSI - 1) end = nSI - 1;
    nkeep = end - start + 1;
    if (nkeep >= nSI)
        return 0;
#ifdef ROBUSTFOV_VERBOSE /* compile with -DROBUSTFOV_VERBOSE for diagnostics */
    fprintf(stderr, "robustfov: head-foot axis %d (polarity %s), keeping %d of %d slices (%g mm)\n",
            siax, (pol > 0) ? "+" : "-", nkeep, nSI, nkeep * pixSI);
#endif
    /* crop along siax keeping [start..end] for every volume */
    int nd[3] = {dim[0], dim[1], dim[2]};
    nd[siax] = nkeep;
    size_t nnew3D, tmpNew, nnewTot, nnewBytes;
    if (mc_mul((size_t)nd[0], (size_t)nd[1], &tmpNew) ||
        mc_mul(tmpNew, (size_t)nd[2], &nnew3D) ||
        mc_mul(nnew3D, (size_t)nvol, &nnewTot) ||
        mc_mul(nnewTot, sizeof(float), &nnewBytes))
        return 1;
    size_t ns[3] = {1, (size_t)nd[0], (size_t)nd[0] * nd[1]};
    float *out = (float *)malloc(nnewBytes);
    if (!out)
        return 1; /* allocation failed; input image left unchanged */
    for (int v = 0; v < nvol; v++)
        for (int z = 0; z < nd[2]; z++)
            for (int y = 0; y < nd[1]; y++)
                for (int x = 0; x < nd[0]; x++) {
                    int ic[3] = {x, y, z};
                    ic[siax] += start;
                    size_t ii = v * nvox3D + (size_t)ic[0] * stride[0] + (size_t)ic[1] * stride[1] + (size_t)ic[2] * stride[2];
                    size_t oo = v * nnew3D + (size_t)x * ns[0] + (size_t)y * ns[1] + (size_t)z * ns[2];
                    out[oo] = img[ii];
                }
    /* Shift the translation of each USABLE coded transform so kept voxels keep
       their world coordinates. A coded-but-unusable (NaN/singular) form was never
       used for axis selection; shifting+inverting it would manufacture a NaN
       transform, so drop its code instead. Images with no usable xform keep none
       (the crop still updates dims). */
    if (nim->sform_code > 0) {
        if (s_ok) {
            for (int i = 0; i < 3; i++)
                nim->sto_xyz.m[i][3] += start * nim->sto_xyz.m[i][siax];
            nim->sto_ijk = nifti_dmat44_inverse(nim->sto_xyz);
        } else {
            nim->sform_code = 0;   /* unusable coded sform: drop rather than corrupt */
        }
    }
    if (nim->qform_code > 0) {
        if (q_ok) {
            for (int i = 0; i < 3; i++)
                nim->qto_xyz.m[i][3] += start * nim->qto_xyz.m[i][siax];
            /* the qform is written from qoffset_*, not qto_xyz, so update those too */
            nim->qoffset_x = nim->qto_xyz.m[0][3];
            nim->qoffset_y = nim->qto_xyz.m[1][3];
            nim->qoffset_z = nim->qto_xyz.m[2][3];
            nim->qto_ijk = nifti_dmat44_inverse(nim->qto_xyz);
        } else {
            nim->qform_code = 0;   /* unusable coded qform: drop rather than corrupt */
        }
    }
    /* slice-timing indices no longer reference valid slices after cropping */
    nim->slice_start = 0;
    nim->slice_end = 0;
    nim->nx = nd[0];
    nim->ny = nd[1];
    nim->nz = nd[2];
    nim->dim[1] = nd[0];
    nim->dim[2] = nd[1];
    nim->dim[3] = nd[2];
    nim->nvox = (int64_t)nnewTot;
    free(nim->data);
    nim->data = out;
    return 0;
}

/*==========================================================================*/
/* Gaussian blur — clean-room-permitted port of niimath coreFLT.c            */
/* (nifti_smooth_gauss / blurS / blurP / transposeXY / transposeXZ), BSD/    */
/* public-domain (NOT src/GPL/). Adapted to operate on a raw float buffer +  */
/* explicit dims/pixdim instead of a nifti_image; flt->float; printfx->      */
/* stderr; sigma passed in mm (per axis). Clean-room BSD port, see AGENTS.md.            */
/*==========================================================================*/

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* transpose X<->Y (rows<->columns) of an nx*ny*nz volume into out */
static inline void mcf_transposeXY(float *in, float *out, int *nxp, int *nyp, int nz) {
    int nx = *nxp, ny = *nyp;
    size_t vi = 0;
    for (int z = 0; z < nz; z++) {
        int zo = z * nx * ny;
        for (int y = 0; y < ny; y++) {
            int xo = 0;
            for (int x = 0; x < nx; x++) { out[zo + xo + y] = in[vi]; xo += ny; vi++; }
        }
    }
    *nxp = ny; *nyp = nx;
}

/* transpose X<->Z (slices<->columns) of an nx*ny*nz volume into out */
static inline void mcf_transposeXZ(float *in, float *out, int *nxp, int ny, int *nzp) {
    int nx = *nxp, nz = *nzp, nyz = ny * nz;
    size_t vi = 0;
    for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
            int yo = y * nz, zo = 0;
            for (int x = 0; x < nx; x++) { out[z + yo + zo] = in[vi]; zo += nyz; vi++; }
        }
    }
    *nxp = nz; *nzp = nx;
}

/* Blur the leading dimension (length nx) of ny contiguous rows. sigma in mm.
   Returns 0 on success, 1 on allocation failure. */
static int mcf_blur1d(float *img, int nx, int ny, float xmm, float sigma_mm,
                      float kernelWid, int parallel) {
    if (xmm == 0.0f || nx < 2 || ny < 1 || sigma_mm <= 0.0f) return 0;
    float sigma = sigma_mm / xmm;   /* mm -> voxels */
    /* Samples beyond the row ends can never contribute. Capping before conversion
       avoids oversized kernels and keeps extreme-but-finite voxel sizes out of an
       overflowing float-to-int conversion. */
    double radius = fabs((double)kernelWid * (double)sigma);
    int cutoffvox = (!(radius < nx - 1)) ? nx - 1
                    : (kernelWid < 0) ? (int)lround(radius) : (int)ceil(radius);
    if (cutoffvox < 1) cutoffvox = 1;
    float *k = (float *)malloc((size_t)(cutoffvox + 1) * sizeof(float));
    int *kStart = (int *)malloc((size_t)nx * sizeof(int));
    int *kEnd   = (int *)malloc((size_t)nx * sizeof(int));
    float *kWeight = (float *)malloc((size_t)nx * sizeof(float));
    if (!k || !kStart || !kEnd || !kWeight) {
        free(k); free(kStart); free(kEnd); free(kWeight); return 1;
    }
    float expd = 2.0f * sigma * sigma;
    for (int i = 0; i <= cutoffvox; i++) k[i] = expf(-1.0f * (i * i) / expd);
    for (int i = 0; i < nx; i++) {
        kStart[i] = MAX(-cutoffvox, -i);
        kEnd[i]   = MIN(cutoffvox, nx - i - 1);
        if (i > 0 && kStart[i] == kStart[i-1] && kEnd[i] == kEnd[i-1]) {
            kWeight[i] = kWeight[i-1]; continue;
        }
        float wt = 0.0f;
        for (int j = kStart[i]; j <= kEnd[i]; j++) wt += k[abs(j)];
        kWeight[i] = 1.0f / wt;
    }
    int failed = 0;
#if defined(_OPENMP)
    if (parallel) {
        /* One scratch row per thread (not per row): removes allocator
           contention, and — because the failure flag is resolved BEFORE the
           parallel region — removes the unsynchronized write to `failed` from
           inside the loop. Blur math and per-row write order are unchanged, so
           the output is bit-identical at any thread count. */
        int nthreads = omp_get_max_threads();
        if (nthreads > ny) nthreads = ny;
        float **scratch = (float **)calloc((size_t)nthreads, sizeof(float *));
        if (!scratch) failed = 1;
        else {
            for (int t = 0; t < nthreads; t++) {
                scratch[t] = (float *)malloc((size_t)nx * sizeof(float));
                if (!scratch[t]) { failed = 1; break; }
            }
            if (!failed) {
                #pragma omp parallel for num_threads(nthreads)
                for (int y = 0; y < ny; y++) {
                    float *tmp = scratch[omp_get_thread_num()];
                    float *row = img + (size_t)nx * y;
                    memcpy(tmp, row, (size_t)nx * sizeof(float));
                    for (int x = 0; x < nx; x++) {
                        float sum = 0.0f;
                        for (int i = kStart[x]; i <= kEnd[x]; i++) sum += tmp[x + i] * k[abs(i)];
                        row[x] = sum * kWeight[x];
                    }
                }
            }
            for (int t = 0; t < nthreads; t++) free(scratch[t]);
            free(scratch);
        }
    } else
#else
    (void)parallel;
#endif
    {
        float *tmp = (float *)malloc((size_t)nx * sizeof(float));
        if (!tmp) failed = 1;
        else {
            float *row = img;
            for (int y = 0; y < ny; y++) {
                memcpy(tmp, row, (size_t)nx * sizeof(float));
                for (int x = 0; x < nx; x++) {
                    float sum = 0.0f;
                    for (int i = kStart[x]; i <= kEnd[x]; i++) sum += tmp[x + i] * k[abs(i)];
                    row[x] = sum * kWeight[x];
                }
                row += nx;
            }
            free(tmp);
        }
    }
    free(k); free(kStart); free(kEnd); free(kWeight);
    return failed;
}

int mcf_smooth_gauss(float *data, int nx, int ny, int nz,
                     float dx, float dy, float dz,
                     float sigX_mm, float sigY_mm, float sigZ_mm, float kernelWid) {
    if (!data || nx < 1 || ny < 1 || nz < 1) return 1;
    size_t nvox3D;
    if (mc_mul((size_t)nx * ny, (size_t)nz, &nvox3D)) return 1;
    if (nvox3D < 2) return 1;
    /* Precondition: the spatial product nx*ny*nz must fit in a signed int. The
       blur/transpose helpers (mcf_blur1d's row count, mcf_transposeXY/XZ's
       `z*nx*ny` offset) use int row products, so a larger-but-valid volume would
       overflow them. One boundary check here; no per-loop checks below. */
    if (nvox3D > (size_t)INT_MAX) {
        fprintf(stderr, "mcf_smooth_gauss: nx*ny*nz (%zu) exceeds INT_MAX\n", nvox3D);
        return 1;
    }
    if (sigX_mm <= 0.0f && sigY_mm <= 0.0f && sigZ_mm <= 0.0f) return 0;
#if defined(_OPENMP)
    int parallel = (omp_get_max_threads() > 1);
#else
    int parallel = 0;
#endif
    /* Blur X: rows are already contiguous. */
    if (sigX_mm > 0.0f && nx >= 2) {
        size_t nRow = (size_t)ny * nz;
        if (mcf_blur1d(data, nx, (int)nRow, dx, sigX_mm, kernelWid, parallel)) return 1;
    }
    /* Blur Y: transpose XY, blur leading (now Y) dim, transpose back. */
    if (sigY_mm > 0.0f && ny >= 2) {
        float *t = (float *)malloc(nvox3D * sizeof(float));
        if (!t) return 1;
        int tx = nx, ty = ny;
        mcf_transposeXY(data, t, &tx, &ty, nz);          /* t is ny*nx*nz */
        int rc = mcf_blur1d(t, ny, nx * nz, dy, sigY_mm, kernelWid, parallel);
        int bx = ny, by = nx;
        mcf_transposeXY(t, data, &bx, &by, nz);          /* back to nx*ny*nz */
        free(t);
        if (rc) return 1;
    }
    /* Blur Z: transpose XZ, blur leading (now Z) dim, transpose back. */
    if (sigZ_mm > 0.0f && nz >= 2) {
        float *t = (float *)malloc(nvox3D * sizeof(float));
        if (!t) return 1;
        int tx = nx, tz = nz;
        mcf_transposeXZ(data, t, &tx, ny, &tz);          /* t is nz*ny*nx */
        int rc = mcf_blur1d(t, nz, nx * ny, dz, sigZ_mm, kernelWid, parallel);
        int bx = nz, bz = nx;
        mcf_transposeXZ(t, data, &bx, ny, &bz);          /* back to nx*ny*nz */
        free(t);
        if (rc) return 1;
    }
    return 0;
}
