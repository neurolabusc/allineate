#ifndef ALLINEATE_H
#define ALLINEATE_H

/* Affine (12 DOF) image registration
   Adapted from AFNI's 3dAllineate by RW Cox (public domain)
   Supports Hellinger (default), lpc, lpa, and Pearson (ls) cost functions
   with twopass coarse-to-fine optimization and TOHD blok local correlation.
   Compile with -DAL_LPC_MICHO to use lpc+ZZ combined cost (slower, adds
   Hellinger/MI/NMI/CrA helper costs with +ZZ pure-lpc refinal pass). */

#include "nifti_io.h"
#include <string.h>

/* Cost function codes (user-facing subset) */
#define AL_COST_LPC      0  /* lpc (cross-modal, e.g. fMRI→T1) */
#define AL_COST_LPA      1  /* lpa (cross-modal, default for deface) */
#define AL_COST_HELLINGER 2  /* Hellinger (default for allineate) */
#define AL_COST_PEARSON   3  /* Global Pearson correlation (within-modality, fast) */

/* Center-of-mass modes */
#define AL_CMASS_NONE     0  /* No center-of-mass alignment */
#define AL_CMASS_YES      1  /* Use center-of-mass for initial shift */

/* Registration engine selector (al_opts.fast). The fast engine is a second, explicitly
   selected estimator (coreg_fast.c); the value also carries which fast cost to use. */
#define AL_ENGINE_ALLINEATE 0  /* default allineate engine (nii_allineate) */
#define AL_ENGINE_FAST_CR   1  /* fast engine, correlation-ratio cost  (-cost fastcr) */
#define AL_ENGINE_FAST_HEL  2  /* fast engine, Hellinger cost          (-cost fast) */

/* al_opts.cli_set bits — which override options the user explicitly passed. Split
   matching-interp (-interp) from output-interp (-final/-nearest/-linear/-cubic): the
   fast engine honors -final for its one output reslice but ignores matching -interp. */
#define AL_CLI_COST   0x1u
#define AL_CLI_WARP   0x2u
#define AL_CLI_INTERP 0x4u   /* -interp (fine-pass MATCHING interpolation) */
#define AL_CLI_FINAL  0x8u   /* -final / -nearest / -linear / -cubic (OUTPUT interpolation) */
#define AL_CLI_CMASS  0x10u  /* -cmass / -nocmass */

/* Warp type codes (number of free parameters) */
#define AL_WARP_SHIFT_ONLY          3  /* shift_only / sho: 3 DOF */
#define AL_WARP_SHIFT_ROTATE        6  /* shift_rotate / shr: 6 DOF */
#define AL_WARP_SHIFT_ROTATE_SCALE  9  /* shift_rotate_scale / srs: 9 DOF */
#define AL_WARP_AFFINE_GENERAL     12  /* affine_general / aff: 12 DOF (default) */

/* Interpolation codes (shared between -interp and -final) */
#define AL_INTERP_NN       0  /* Nearest-neighbor */
#define AL_INTERP_LINEAR   1  /* Trilinear (default for -interp, matching AFNI) */
#define AL_INTERP_CUBIC    3  /* Tricubic */
#define AL_INTERP_DEFAULT -1  /* Use mode-appropriate default (cubic for allineate, linear for deface) */

/* Options for nii_allineate and nii_deface */
typedef struct {
    int cost;              /* AL_COST_* code (default: AL_COST_HELLINGER) */
    int cmass;             /* AL_CMASS_* code (default: AL_CMASS_NONE) */
    int source_automask;   /* if nonzero, fill outside of source automask with noise */
    int dark_automask;     /* -dark_automask: drop matched pairs where the base or warped
                              source value is at that image's darkest value (background/pad) */
    int interp;            /* AL_INTERP_* for fine-pass matching (default: LINEAR) */
    int final_interp;      /* AL_INTERP_* for output reslicing (default: AL_INTERP_DEFAULT) */
    int warp;              /* AL_WARP_* DOF count (default: AL_WARP_AFFINE_GENERAL = 12) */
    /* --- CLI-workflow fields below: interpreted by main.c's pre-/post-processing
       chain, NOT by nii_allineate()/nii_deface(). A direct library caller setting
       these gets a silent no-op (documented DEFERRED decision — see CLAUDE.md:
       "DEFERRED splitting al_opts"). Drive these via the CLI, not the API. --- */
    const char *skullstrip; /* CLI-only: -skullstrip brain mask (NULL = normal registration) */
    double robustfov;       /* CLI-only: -robustfov crop mm applied to the moving image (0 = off) */
    const char *savemat;    /* CLI-only: -savemat path; main.c saves the fitted affine as JSON
                               (via nii_last_affine()). NULL = don't save. */
    const char *applymat;   /* CLI-only: -applymat path; apply a saved affine JSON to reslice
                               the moving image onto the stationary grid (no registration). */
    const char *master;     /* CLI-only: -master output-grid image. Register at the stationary
                               resolution, but reslice the result onto THIS grid instead (must
                               share the stationary world frame, e.g. a higher-res template).
                               NULL = output on the stationary grid. */
    int com;                /* CLI-only: -com set origin to brightness center of mass (1 = on) */
    int sym;                /* CLI-only: -sym/-symd midsagittal alignment (1 = enabled, 0 = disabled) */
    int sym_deoblique;      /* CLI-only: 0 = -sym; 1 = -symd (snap the frame axis-aligned
                               before the mirror fit); 2 = -symb (auto-compete both) */
    int sagseed;            /* CLI-only: -sagseed in-MSP rigid seed after -sym (default 1;
                               -nosagseed disables). Only acts when -sym runs with a template. */
    unsigned cli_set;       /* CLI-only: bitmask of explicitly-passed override options
                               (AL_CLI_*), so a distinct engine can reject options it
                               cannot honor (e.g. the fast engine rejects -warp/-interp; it
                               honors -cmass/-nocmass via AL_CLI_CMASS). */
    int fast;               /* CLI-only: registration engine selector (AL_ENGINE_*). Interpreted
                               by main.c (dispatches to coreg_fast_estimate); nii_allineate()
                               ignores it. */
    int zoom;               /* CLI-only: -zoom — flags abnormal size (e.g. infant vs adult
                               template): adds a global isotropic scale DOF to the -sagseed
                               seed fit AND widens the main affine's scale range to [0.5,2.0]
                               (from 0.711..1.406). Only the scale limits relax; all other
                               regularization and default (adult) behavior are unchanged. */
} al_opts;

/* Initialize options to defaults */
static inline al_opts al_opts_default(void) {
    al_opts o;
    o.cost = AL_COST_HELLINGER;
    o.cmass = AL_CMASS_NONE;
    o.source_automask = 0;
    o.dark_automask = 0;
    o.interp = AL_INTERP_LINEAR;
    o.final_interp = AL_INTERP_DEFAULT;
    o.warp = AL_WARP_AFFINE_GENERAL;
    o.skullstrip = NULL;
    o.robustfov = 0.0;
    o.savemat = NULL;
    o.applymat = NULL;
    o.master = NULL;
    o.com = 0;
    o.sym = 0;
    o.sym_deoblique = 0;
    o.sagseed = 1;
    o.cli_set = 0;
    o.fast = 0;
    o.zoom = 0;
    return o;
}

/* Parse cost function name string into AL_COST_* code.
   Returns 0 on success, 1 on unrecognized name. */
static inline int al_parse_cost(const char *name, int *cost_out) {
    if (!strcmp(name, "lpc"))                        { *cost_out = AL_COST_LPC; return 0; }
    if (!strcmp(name, "lpa"))                        { *cost_out = AL_COST_LPA; return 0; }
    if (!strcmp(name, "hel"))                        { *cost_out = AL_COST_HELLINGER; return 0; }
    if (!strcmp(name, "ls") || !strcmp(name, "pearson")) { *cost_out = AL_COST_PEARSON; return 0; }
    return 1;
}

/* Parse interpolation name string into AL_INTERP_* code.
   Returns 0 on success, 1 on unrecognized name. */
static inline int al_parse_interp(const char *name, int *code_out) {
    if (!strcmp(name, "NN") || !strcmp(name, "nearest") ||
        !strcmp(name, "nearestneighbour") || !strcmp(name, "nearestneighbor"))
        { *code_out = AL_INTERP_NN; return 0; }
    if (!strcmp(name, "linear") || !strcmp(name, "trilinear"))
        { *code_out = AL_INTERP_LINEAR; return 0; }
    if (!strcmp(name, "cubic") || !strcmp(name, "tricubic"))
        { *code_out = AL_INTERP_CUBIC; return 0; }
    return 1;
}

/* Parse warp type name into AL_WARP_* code.
   Returns 0 on success, 1 on unrecognized name. */
static inline int al_parse_warp(const char *name, int *warp_out) {
    if (!strcmp(name, "shift_only") || !strcmp(name, "sho"))
        { *warp_out = AL_WARP_SHIFT_ONLY; return 0; }
    if (!strcmp(name, "shift_rotate") || !strcmp(name, "shr"))
        { *warp_out = AL_WARP_SHIFT_ROTATE; return 0; }
    if (!strcmp(name, "shift_rotate_scale") || !strcmp(name, "srs"))
        { *warp_out = AL_WARP_SHIFT_ROTATE_SCALE; return 0; }
    if (!strcmp(name, "affine_general") || !strcmp(name, "aff"))
        { *warp_out = AL_WARP_AFFINE_GENERAL; return 0; }
    return 1;
}

/* Return human-readable name for an AL_WARP_* code. */
static inline const char *al_warp_name(int warp) {
    switch (warp) {
        case AL_WARP_SHIFT_ONLY:         return "shift_only";
        case AL_WARP_SHIFT_ROTATE:       return "shift_rotate";
        case AL_WARP_SHIFT_ROTATE_SCALE: return "shift_rotate_scale";
        default:                         return "affine_general";
    }
}

/* Canonical (re-parseable, matches al_parse_cost) name for an AL_COST_* code. */
static inline const char *al_cost_name(int cost) {
    switch (cost) {
        case AL_COST_LPC:      return "lpc";
        case AL_COST_LPA:      return "lpa";
        case AL_COST_PEARSON:  return "ls";
        default:               return "hel";
    }
}

/* The registration/deface API uses process-global and thread-local workspaces.
   It is safe for serial CLI-style use, but not for concurrent calls. */

/* Choose an image's index->world (mm) transform with NIfTI sform/qform precedence
   and validity policy (prefer sform when sform_code >= qform_code, else qform; fall
   back to whichever form is usable; a degenerate/bogus preferred form is skipped).
   Writes *out and returns 0 on success; returns 1 when neither form is usable.
   Pure helper (no global state) shared with the fast coreg path (coreg_fast.c).
   Exposed non-static in Phase 2 of the fast-coreg work with NO logic change —
   exact parity holds (a no-logic-change exposure). */
int al_image_xform(const nifti_image *nim, mat44 *out);

/* Register source image to base image grid using affine (12 DOF) alignment.
   source: the moving image (will be modified in-place: data replaced, dims updated)
   base: the stationary/reference image
   opts: registration options (cost function, cmass, interpolation, etc.)
   final_interp default: cubic.
   Returns 0 on success, nonzero on error. */
int nii_allineate(nifti_image *source, nifti_image *base, al_opts opts);

/* Copy the most recent nii_allineate() fit's world-space FIXED(base)->MOVING(source)
   affine (mm) into *out. Returns 0 if a fit has run, nonzero otherwise. The matrix
   is the "pull"/resampling transform (fixed-grid world coords -> moving world coords)
   and reflects the moving image as passed to registration (after any -com/-sym fold).
   Serial-only (reads a process-global, like the rest of the engine). */
int nii_last_affine(mat44 *out);

/* Reslice `source` onto `base`'s grid using an explicit base-index -> source-index
 * affine `gam` (0-based NIfTI voxel indices). Replaces source->data with the
 * resliced float volume and adopts base dims + sform/qform. interp: AL_INTERP_*;
 * out-of-FOV voxels set to `fillv` (0 or NaN). Returns 0 on success. Shared BSD
 * interpolation (used by -allineate output and the GPL -spm_coreg reslice). */
int nii_reslice_affine(nifti_image *source, const nifti_image *base,
                       mat44 gam, int interp, float fillv);

/* Apply a saved world-space FIXED->MOVING affine (`-savemat` `fixed_to_moving`) to
   reslice `input` (an image in the prior registration's MOVING space) onto `target`'s
   grid — `target` may have any resolution/FOV/origin sharing the fixed world frame
   (the matrix is world-mm, not grid-specific). input->data is replaced with the
   resliced volume on target's grid. interp: AL_INTERP_*; out-of-FOV -> fillv.
   Returns 0 on success, nonzero on error. */
int nii_apply_affine(nifti_image *input, const nifti_image *target,
                     mat44 fixed_to_moving, int interp, float fillv);

/* Template-free midsagittal-plane (MSP) alignment: register `nim` to its world-X
   mirror, take the half of the recovered rigid transform, and either resample the
   data symmetric about world X=0 (reslice != 0, standalone use) or fold the
   correction into the header as a registration seed (reslice == 0, pre-step use).
   If `C_out` is non-NULL it receives the 4x4 world-space correction. The mirror fit
   uses the `ls`/Pearson cost. `deoblique` (nonzero, `-symd`) first snaps the frame
   to axis-aligned (treats the voxel grid as anatomical) so an obliquely-acquired but
   grid-symmetric head is not rotated onto the oblique world frame. `deoblique`:
   0 = `-sym` (image's world frame); 1 = `-symd` (snap the frame axis-aligned first);
   2 = `-symb` (fit both frames; keep the de-obliqued frame only if its correction
   rotation is smaller by more than AL_SYMB_ROT_TOL_DEG, else keep the original world
   frame — a tie/dead-band bias, and only rotation is compared, not fit cost).
   `dark_automask` (nonzero, `-dark_automask`) drops background/pad matched pairs (at
   the image minimum) from the mirror-fit cost. Uses the process-global registration
   workspaces, so it is serial-only (see nii_allineate).
   Returns 0 on success, nonzero on error. */
int nii_symmetry(nifti_image *nim, mat44 *C_out, int reslice, int deoblique, int dark_automask);

/* -sagseed: in-MSP rigid seed, the complement of -sym. Precondition: -sym has
   folded its MSP correction into `nim`'s header (midsagittal plane at world X=0)
   and `tmpl` is an MSP-aligned template. Runs a 3-DOF-constrained fit freeing only
   the MSP-preserving isometries {y-shift, z-shift, pitch} — the rigid DOF -sym is
   blind to — and folds the correction into nim's header as a full-rigid seed for a
   subsequent nii_allineate. Uses the user's cost/interp/source_automask/cmass from
   `opts`. Serial-only (process-global workspaces). Returns 0 on success. */
int nii_sagseed(nifti_image *nim, nifti_image *tmpl, al_opts opts);

/* -com: set the image origin to its brightness center of mass. Computes the
   intensity-weighted centroid (positive voxels), maps it to world coordinates via
   the selected index->world transform, and folds a pure translation into the
   header so the centroid sits at world (0,0,0). Header-only (no reslice); a cheap
   origin reset, intended to run early (right after -robustfov). Template-free:
   falls back to a pixdim-centered frame when the input has no usable sform/qform.
   Returns 0 on success, nonzero on error. */
int nii_center_of_mass(nifti_image *nim);

/* Deface: register INPUT to TEMPLATE (the well-posed direction, base =
   template, same as -allineate), INVERT the transform, warp the template-space mask
   onto the input's native grid, and set voxels where warped mask < 0.5 to the input's
   minimum value. (Registering template->input instead mislocates the mask.)
   input: the image to modify (modified in-place, stays in its own native space)
   tmpl: template image — the registration base/fixed (input is the moving image)
   mask: mask in template space (>=0.5 = keep, <0.5 = remove). CONSUMED: its data is
         freed and replaced with the resliced mask on the input grid (regridded to
         input geometry). Caller still owns the nifti_image and must free it.
   opts: registration options (cost function, cmass)
   final_interp default: linear (to avoid ringing in the mask).
   Returns 0 on success, nonzero on error. */
int nii_deface(nifti_image *input, nifti_image *tmpl, nifti_image *mask, al_opts opts);

/* Zero (to image minimum) input voxels where the input-grid `warped_mask` < 0.5.
   Ensures input is float32. BSD; shared by -deface and the GPL -spm_deface.
   Returns #voxels masked, or -1 on error. */
long nii_apply_deface_mask(nifti_image *input, const float *warped_mask);

#endif /* ALLINEATE_H */
