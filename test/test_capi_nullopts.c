/* Direct C-API regression: coreg_fast_estimate(moving, fixed, NULL, &res).
 *
 * The estimator documents a default-options contract — `opts == NULL` means "use
 * coreg_fast_opts_default()" (coreg_fast.c: `coreg_fast_opts O = opts ? *opts : ...`).
 * A regression once read the raw `opts->use_cmass` after pyramid construction, which
 * segfaults on the NULL path. The CLI always passes a non-NULL opts, so only a direct
 * C-API call can catch this. This harness exercises the NULL path end-to-end.
 *
 * Reads two NIfTI files (the test generates them), calls the estimator with NULL opts,
 * and requires a successful fit. A segfault/abort or a clean-but-incorrect failure both
 * produce a nonzero exit that fails `make test`.
 *
 * Built by the Makefile `test` target without main.c:
 *   cc <cflags> test/test_capi_nullopts.c allineate.c nifti_io.c powell_newuoa.c \
 *      miniCoreFLT.c coreg_fast.c <zlib> -lm -o test_capi_nullopts
 */
#include <stdio.h>
#include <string.h>
#include "nifti_io.h"
#include "allineate.h"
#include "coreg_fast.h"
#include "miniCoreFLT.h"
#include "reface.h"

#ifdef COREG_FAST_TEST_ALLOC
extern void coreg_fast_test_fail_samples_on_call(int call);
extern int coreg_fast_test_sample_call_count(void);
#endif

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <moving.nii> <fixed.nii>\n", argv[0]);
        return 2;
    }
    nifti_image *mov = nifti_image_read(argv[1], 1);
    nifti_image *fix = nifti_image_read(argv[2], 1);
    if (!mov || !fix) {
        fprintf(stderr, "test_capi_nullopts: failed to read inputs\n");
        if (mov) nifti_image_free(mov);
        if (fix) nifti_image_free(fix);
        return 2;
    }

#ifdef AL_TEST_32BIT_SIZE
    /* Simulate wasm32's SIZE_MAX in this native harness. The product fits int,
       but not a 32-bit float buffer, so every public boundary must reject it
       before reading the one-voxel dummy or touching caller output state. */
    float dummy = 0.0f;
    const int hx = 1024, hy = 1024, hz = 1025;
    const int64_t hnv = (int64_t)hx * hy * hz;
    nifti_image huge = *mov;
    huge.nx = huge.dim[1] = hx;
    huge.ny = huge.dim[2] = hy;
    huge.nz = huge.dim[3] = hz;
    huge.nt = huge.nu = huge.nv = huge.nw = 1;
    huge.nvox = hnv;
    huge.data = &dummy;

    coreg_fast_result size_result, size_before;
    memset(&size_result, 0x5A, sizeof size_result);
    memcpy(&size_before, &size_result, sizeof size_before);
    if (coreg_fast_estimate(&huge, &huge, NULL, &size_result) == 0 ||
        memcmp(&size_result, &size_before, sizeof size_result) != 0) {
        fprintf(stderr, "coreg accepted a non-addressable 32-bit float volume\n");
        nifti_image_free(mov); nifti_image_free(fix); return 1;
    }

    mat44 ident = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}}};
    nifti_image reslice_source = huge;
    if (nii_reslice_affine(&reslice_source, &huge, ident, AL_INTERP_LINEAR, 0.0f) == 0 ||
        reslice_source.data != &dummy) {
        fprintf(stderr, "allineate accepted a non-addressable 32-bit float volume\n");
        nifti_image_free(mov); nifti_image_free(fix); return 1;
    }

    nifti_image *refaced = (nifti_image *)0x1;
    if (reface_apply(&huge, &huge, &refaced) == 0 || refaced != NULL) {
        fprintf(stderr, "reface accepted a non-addressable 32-bit float volume\n");
        nifti_image_free(mov); nifti_image_free(fix); return 1;
    }
#endif

    coreg_fast_result res;
    /* The load-bearing call: NULL opts must resolve to defaults, not dereference NULL. */
    int rc = coreg_fast_estimate(mov, fix, NULL, &res);
    fprintf(stderr, "coreg_fast_estimate(NULL opts) returned %d\n", rc);
    if (rc) {
        nifti_image_free(mov);
        nifti_image_free(fix);
        return 1;
    }

    /* A rigid max_dof request must never inherit scale from an exploratory coarse seed.
       This especially guards the hard-zero multi-start's scale-bracket strategy. */
    coreg_fast_opts rigid = coreg_fast_opts_default();
    rigid.max_dof = 6;
    rc = coreg_fast_estimate(mov, fix, &rigid, &res);
    if (rc) {
        fprintf(stderr, "coreg_fast_estimate(max_dof=6) returned %d\n", rc);
        nifti_image_free(mov);
        nifti_image_free(fix);
        return 1;
    }
    float det = res.fixed_to_moving.m[0][0] *
                    (res.fixed_to_moving.m[1][1] * res.fixed_to_moving.m[2][2] -
                     res.fixed_to_moving.m[1][2] * res.fixed_to_moving.m[2][1]) -
                res.fixed_to_moving.m[0][1] *
                    (res.fixed_to_moving.m[1][0] * res.fixed_to_moving.m[2][2] -
                     res.fixed_to_moving.m[1][2] * res.fixed_to_moving.m[2][0]) +
                res.fixed_to_moving.m[0][2] *
                    (res.fixed_to_moving.m[1][0] * res.fixed_to_moving.m[2][1] -
                     res.fixed_to_moving.m[1][1] * res.fixed_to_moving.m[2][0]);
    fprintf(stderr, "coreg_fast_estimate(max_dof=6) returned %d, dof=%d, det=%.7g\n",
            rc, res.resolved_dof, det);
    if (res.resolved_dof != 6 || !(det > 0.9999f && det < 1.0001f)) {
        nifti_image_free(mov);
        nifti_image_free(fix);
        return 1;
    }

#ifdef COREG_FAST_TEST_ALLOC
    /* Make the fixed image hard-zeroed so the default runs multiple strategies.
       Strategy 0 consumes sample calls 1..3; fail a PARTIAL allocation at strategy
       1's coarse sample rebuild (call 4). The estimator must fail atomically and
       must not polish the prior winner using stale ns/partial replacement arrays. */
    if (fix->datatype != DT_FLOAT32 || !fix->data || fix->nvox < 32) {
        fprintf(stderr, "fault-injection fixture is not float32\n");
        nifti_image_free(mov);
        nifti_image_free(fix);
        return 1;
    }
    float *fd = (float *)fix->data;
    for (size_t i = 0; i < (size_t)fix->nvox / 3; i++) fd[i] = 0.0f;
    coreg_fast_result sentinel, before;
    memset(&sentinel, 0xA5, sizeof sentinel);
    memcpy(&before, &sentinel, sizeof before);
    coreg_fast_test_fail_samples_on_call(4);
    coreg_fast_opts injected = coreg_fast_opts_default();
    injected.max_dof = 6;
    rc = coreg_fast_estimate(mov, fix, &injected, &sentinel);
    int sample_calls = coreg_fast_test_sample_call_count();
    fprintf(stderr, "sample fault injection returned %d after %d sample builds\n",
            rc, sample_calls);
    if (rc == 0 || sample_calls != 4 ||
        memcmp(&sentinel, &before, sizeof sentinel) != 0) {
        fprintf(stderr, "sample-build failure was not clean and atomic\n");
        nifti_image_free(mov);
        nifti_image_free(fix);
        return 1;
    }
#endif

    nifti_image_free(mov);
    nifti_image_free(fix);

    /* A Gaussian radius wider than its row must be clipped before integer conversion
       and allocation; only three samples can contribute here despite the tiny spacing. */
    float impulse[3] = {0.0f, 1.0f, 0.0f};
    if (nifti_smooth_gauss_f32(impulse, 3, 1, 1, 1, 1e-9f, 1.0f, 1.0f,
                               1.0f, 0.0f, 0.0f, -6.0f)) {
        fprintf(stderr, "nifti_smooth_gauss_f32 rejected a bounded wide kernel\n");
        return 1;
    }
    return 0;
}
