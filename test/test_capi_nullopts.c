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
#include "nifti_io.h"
#include "coreg_fast.h"
#include "miniCoreFLT.h"

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
    coreg_fast_result res;
    /* The load-bearing call: NULL opts must resolve to defaults, not dereference NULL. */
    int rc = coreg_fast_estimate(mov, fix, NULL, &res);
    fprintf(stderr, "coreg_fast_estimate(NULL opts) returned %d\n", rc);
    nifti_image_free(mov);
    nifti_image_free(fix);
    if (rc) return 1;

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
