/* Direct C-API regression for qwarp_run() — locks the public boundary the Python CLI gate
 * cannot reach, because nifti_image_read() SYNCHRONIZES sform/qform on load, so a CLI test can
 * never present disagreeing/coded-differently transforms or oversize header metadata.
 *
 * Asserts:
 *   (1) the ATOMIC-FAILURE contract: on any invalid input (oversize dims, too-small grid,
 *       mismatched dims, no usable transform, both transforms unusable) qwarp_run() returns
 *       nonzero AND leaves *result == NULL — never a partial output;
 *   (2) the dual-coded transform selection / two-form fallback: a coded-but-SINGULAR sform with
 *       a coded VALID qform must be accepted (fall back to qform), i.e. geometry is NOT rejected.
 *
 * Built by the Makefile `test` target without main.c:
 *   cc <cflags> test/test_qwarp_capi.c allineate.c nifti_io.c powell_newuoa.c \
 *      miniCoreFLT.c coreg_fast.c qwarp.c <zlib> -lm -o test_qwarp_capi
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nifti_io.h"
#include "qwarp.h"

extern void powell_set_mfac(float mm, float aa);
extern void powell_get_mfac(float *mm, float *aa);

static int npass = 0, nfail = 0;
static void check(const char *name, int cond) {
    if (cond) { npass++; printf("  [PASS] %s\n", name); }
    else      { nfail++; printf("  [FAIL] %s\n", name); }
}

/* Build a minimal single-volume float32 nifti_image with a structured blob (so weightize finds
 * foreground). sform/qform codes and singularity are set explicitly. Caller frees. */
static nifti_image *mk(int nx, int ny, int nz, int sform_code, int sform_singular,
                       int qform_code, int qform_singular) {
    nifti_image *n = (nifti_image *)calloc(1, sizeof *n);
    if (!n) return NULL;
    n->ndim = 3; n->dim[0] = 3;
    n->nx = n->dim[1] = nx; n->ny = n->dim[2] = ny; n->nz = n->dim[3] = nz;
    n->nt = n->nu = n->nv = n->nw = 1; n->dim[4] = n->dim[5] = n->dim[6] = n->dim[7] = 1;
    n->nvox = (int64_t)nx * ny * nz;
    n->datatype = DT_FLOAT32; n->nbyper = 4;
    n->dx = n->dy = n->dz = 1.0f; n->pixdim[1] = n->pixdim[2] = n->pixdim[3] = 1.0f;
    n->scl_slope = 0.0; n->scl_inter = 0.0;
    /* For oversize tests nvox is huge; guard the allocation and fill so we don't try to alloc TB.
     * The oversize dimension is rejected BEFORE the data is read, so a 1-voxel dummy suffices. */
    int64_t alloc_n = (n->nvox > 0 && n->nvox < (int64_t)4000000) ? n->nvox : 1;
    float *d = (float *)calloc((size_t)alloc_n, sizeof(float));
    n->data = d;
    if (d && alloc_n == n->nvox) {
        int nxy = nx * ny;
        for (int k = 0; k < nz; k++) for (int j = 0; j < ny; j++) for (int i = 0; i < nx; i++) {
            double r2 = (i-nx/2.0)*(i-nx/2.0) + (j-ny/2.0)*(j-ny/2.0) + (k-nz/2.0)*(k-nz/2.0);
            d[i + j*nx + k*nxy] = (float)(100.0 * exp(-r2 / (2.0 * (nx/5.0) * (nx/5.0))));
        }
    }
    /* identity sform (or a singular all-zero 3x3 when requested) */
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) n->sto_xyz.m[a][b] = 0.0;
    for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) n->qto_xyz.m[a][b] = 0.0;
    n->sto_xyz.m[3][3] = 1.0; n->qto_xyz.m[3][3] = 1.0;
    if (!sform_singular) { n->sto_xyz.m[0][0] = n->sto_xyz.m[1][1] = n->sto_xyz.m[2][2] = 1.0; }
    if (!qform_singular) { n->qto_xyz.m[0][0] = n->qto_xyz.m[1][1] = n->qto_xyz.m[2][2] = 1.0; }
    n->sform_code = sform_code; n->qform_code = qform_code;
    return n;
}

/* Run qwarp_run(mov,sta) and require: rc nonzero AND *result == NULL (atomic failure). */
static void expect_reject(const char *name, nifti_image *mov, nifti_image *sta) {
    nifti_image *out = (nifti_image *)0x1;   /* poison: qwarp_run must set it to NULL */
    int rc = qwarp_run(mov, sta, &out);
    check(name, rc != 0 && out == NULL);
    if (out && out != (nifti_image *)0x1) nifti_image_free(out);
}

/* Run qwarp_run(mov,sta) and require success: rc==0 AND *result != NULL. */
static void expect_accept(const char *name, nifti_image *mov, nifti_image *sta) {
    nifti_image *out = NULL;
    int rc = qwarp_run(mov, sta, &out);
    check(name, rc == 0 && out != NULL);
    if (out) nifti_image_free(out);
}

int main(void) {
    printf("qwarp C-API boundary tests\n");

    /* --- atomic rejection paths (fast; reject before the engine) --- */
    { nifti_image *a = mk(30000, 8, 8, 1,0, 1,0), *b = mk(8,8,8, 1,0, 1,0);
      expect_reject("oversize dimension -> reject, *result NULL", a, b);
      nifti_image_free(a); nifti_image_free(b); }
    { nifti_image *a = mk(3,3,3, 1,0, 1,0), *b = mk(3,3,3, 1,0, 1,0);
      expect_reject("too-small grid (<5) -> reject, *result NULL", a, b);
      nifti_image_free(a); nifti_image_free(b); }
    { nifti_image *a = mk(20,20,20, 1,0, 1,0), *b = mk(20,20,24, 1,0, 1,0);
      expect_reject("mismatched dims -> reject, *result NULL", a, b);
      nifti_image_free(a); nifti_image_free(b); }
    { nifti_image *a = mk(20,20,20, 0,0, 0,0), *b = mk(20,20,20, 0,0, 0,0);  /* neither coded */
      expect_reject("no coded transform -> reject, *result NULL", a, b);
      nifti_image_free(a); nifti_image_free(b); }
    { nifti_image *a = mk(20,20,20, 1,1, 1,1), *b = mk(20,20,20, 1,1, 1,1);  /* both singular */
      expect_reject("both transforms unusable -> reject, *result NULL", a, b);
      nifti_image_free(a); nifti_image_free(b); }

    /* --- dual-coded selection: coded-but-singular sform + coded valid qform must be ACCEPTED
     * (two-form fallback). A tiny 24^3 structured pair keeps the engine run fast. --- */
    { nifti_image *a = mk(24,24,24, 1,1, 1,0), *b = mk(24,24,24, 1,1, 1,0);
      expect_accept("singular sform + valid qform -> accepted (qform fallback)", a, b);
      nifti_image_free(a); nifti_image_free(b); }

    /* Higher-coded sform must win over qform. The qforms deliberately disagree; selecting them
     * would reject the pair before optimization. */
    { nifti_image *a = mk(24,24,24, 2,0, 1,0), *b = mk(24,24,24, 2,0, 1,0);
      b->qto_xyz.m[0][3] = 10.0f;
      expect_accept("higher-coded sform preferred over disagreeing qform", a, b);
      nifti_image_free(a); nifti_image_free(b); }

    /* qwarp's AFNI sampling policy is private to the call; preserve the embedding caller's
     * thread-local NEWUOA configuration. NOTE: this exercises restoration on the SUCCESSFUL
     * optimize path only; the fatal (-4) and zero-iteration skip paths also restore mfac/g_qw
     * (verified by inspection in qw_improve_warp) but are not fault-injected here. */
    { float mm = 0.0f, aa = 0.0f;
      powell_set_mfac(1.5f, 4.5f);
      nifti_image *a = mk(24,24,24, 1,0, 1,0), *b = mk(24,24,24, 1,0, 1,0);
      expect_accept("qwarp succeeds with caller NEWUOA configuration", a, b);
      powell_get_mfac(&mm, &aa);
      check("qwarp restores caller NEWUOA configuration", mm == 1.5f && aa == 4.5f);
      powell_set_mfac(0.0f, 0.0f);
      nifti_image_free(a); nifti_image_free(b); }

    printf("%d passed, %d failed\n", npass, nfail);
    return nfail ? 1 : 0;
}
