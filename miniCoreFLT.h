#ifndef MINICOREFLT_H
#define MINICOREFLT_H

/* miniCoreFLT — a minimal port of selected niimath coreFLT.c preprocessing
 * operators for the allineate test bed. These mimic niimath behavior so ideas
 * can be prototyped here without cloning the full niimath toolkit.
 *
 * NOTE for chained use: robustfov changes the image dimensions and shifts any
 * valid spatial transform (sform/qform code > 0). Any downstream step (e.g.
 * -allineate) must therefore read the updated geometry from the nifti_image, not
 * cache it.
 */

#include "nifti_io.h"

/* Ensure nim->data is DT_FLOAT32, converting in place from the common
 * integer/float datatypes and applying scl_slope/scl_inter. On success the data
 * pointer is replaced, datatype/nbyper updated, and scaling reset to identity.
 * Returns 0 on success, 1 on error / unsupported datatype. */
int nii_ensure_float32(nifti_image *nim);

/* Robust field-of-view head truncation (emulates FSL robustfov; clean-room port
 * of niimath's nifti_robustfov). Detects the superior-inferior voxel axis from
 * the spatial transform, finds the top of the head from per-slice intensity
 * variance in a central sagittal slab, and keeps `fovmm` mm of slices below it,
 * discarding superior air and lower head/neck. `fovmm` must be positive and finite
 * (NaN/Inf/<=0 is rejected with a nonzero return; the CLI defaults it to 170 mm).
 * Adjusts dims and shifts sform/qform so kept voxels retain world coordinates.
 * Requires DT_FLOAT32 data (call nii_ensure_float32 first). Only a sform/qform
 * whose code is > 0 is shifted; an image with no valid transform (both codes 0) is
 * cropped but keeps no valid spatial transform.
 * Returns 0 on success (image may be modified in place), 1 on error. */
int nifti_robustfov(nifti_image *nim, double fovmm);

#endif /* MINICOREFLT_H */
