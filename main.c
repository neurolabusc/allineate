#ifdef _WIN32
	#include <fcntl.h>
	#include <io.h>
#endif
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "nifti_io.h"
#include "allineate.h"
#include "miniCoreFLT.h"
#include "coreg_fast.h"
#include "qwarp.h"
#include "reface.h"
#if defined(_OPENMP)
	#include <omp.h>
#endif

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#if defined(_OPENMP)
   #define kOMPsuf " OpenMP"
#else
   #define kOMPsuf ""
#endif
#if defined(__ICC) || defined(__INTEL_COMPILER)
	#define kCCsuf  " IntelCC" STR(__INTEL_COMPILER)
#elif defined(_MSC_VER)
	#define kCCsuf  " MSC" STR(_MSC_VER)
#elif defined(__clang__)
	#define kCCsuf  " Clang" STR(__clang_major__) "." STR(__clang_minor__) "." STR(__clang_patchlevel__)
#elif defined(__GNUC__) || defined(__GNUG__)
    #define kCCsuf  " GCC" STR(__GNUC__) "." STR(__GNUC_MINOR__) "." STR(__GNUC_PATCHLEVEL__)
#else
	#define kCCsuf " CompilerNA" //unknown compiler!
#endif
#if defined(__APPLE__)
	#define kOS "MacOS"
#elif (defined(__linux) || defined(__linux__))
	#define kOS "Linux"
#else
	#define kOS "Windows"
#endif

#define kMTHdate "v1.0.20260712"
#define kMTHvers kMTHdate kOMPsuf kCCsuf

int show_help( void ) {
	printf("allineate version %s (%llu-bit %s)\n",kMTHvers, (unsigned long long) sizeof(size_t)*8, kOS);
	printf(" <moving> <stationary> [opts] <output>: co-register the moving image match the stationary image\n");
	printf(" <moving> [opts] <output>            : preprocess only (e.g. -robustfov), no registration\n");
	printf("                 Use '-' for <moving> to read from stdin, '-' for <output> to write to stdout\n");
	printf("                 opts: -cost XX  cost function:\n");
	printf("                                 options: fast/fastx (default), fasthel, fastcr, hel, lpc, lpa, ls;\n");
	printf("                                 'fast'/'fastx' let HEL and CR coarse fits compete\n");
	printf("                                 'fasthel' forces HEL-only fast registration\n");
	printf("                                 'hel' (Hellinger) is a robust cross-modal method\n");
	printf("                                 Other cost functions for special cases\n");
	printf("                                 The fast engine has a fixed config: -warp/-interp/\n");
	printf("                                 -source_automask/-dark_automask/-zoom/-skullstrip are\n");
	printf("                                 not supported with it (-final sets output interp; default/\n");
	printf("                                 -cmass auto-select initialization, -com forces COM,\n");
	printf("                                 -nocmass forces the supplied affine; the -com/-sym header\n");
	printf("                                 seeds DO work with it).\n");
	printf("                       -cmass -nocmass -source_automask\n");
	printf("                       -dark_automask  ignore matched pairs where either image is at its\n");
	printf("                                 darkest value (background/zero-pad); uses each image's own\n");
	printf("                                 minimum so it is safe for signed data (e.g. CT Hounsfield).\n");
	printf("                       -interp XX (NN,linear,cubic) matching interpolation [default: linear]\n");
	printf("                         NOTE: linear is always used for the coarse pass;\n");
	printf("                               -interp only affects the fine pass.\n");
	printf("                               -interp does NOT affect the output image (use -final).\n");
	printf("                       -final XX  (NN,linear,cubic) output interpolation [default: cubic]\n");
	printf("                       -nearest -linear -cubic (shortcuts for -final)\n");
	printf("                       -fill XX  (auto,zero,nan) out-of-FOV output fill [default: auto =\n");
	printf("                                 0, or the darkest voxel if it is <0 (CT/HU air)]\n");
	printf("                       -warp XX (sho,shr,srs,aff) transform type [default: aff]\n");
	printf("                         sho=shift(3), shr=shift+rotate(6), srs=+scale(9), aff=+shear(12)\n");
	printf("                       -skullstrip XX  brain mask in moving space; output = stationary\n");
	printf("                                 with non-brain voxels set to darkest value [default final: linear]\n");
	printf("                       -weight XX  AFNI 3dAllineate-style graded weight image in stationary\n");
	printf("                                 (base) space (dims + world frame match stationary); normalized\n");
	printf("                                 to [0,1] and used per voxel. BOTH engines: the ordinary engine\n");
	printf("                                 replaces its autoweight; the fast engine applies it at the\n");
	printf("                                 FINEST (2mm) stage only (coarse capture + global scale stay\n");
	printf("                                 whole-head). Keep the out-of-ROI head attenuated (nonzero).\n");
	printf("                       -robustfov [mm] crop the moving image to a robust FOV (default 170mm),\n");
	printf("                                 removing lower head/neck (emulates FSL robustfov); adjusts\n");
	printf("                                 dim and valid sform/qform. A good pre-step before registration.\n");
	printf("                       -savemat out.json  save the fitted registration affine (world-space\n");
	printf("                                 4x4, FIXED->MOVING and its inverse) as JSON; needs a stationary image.\n");
	printf("                                 The output image filename may be omitted to save only the matrix.\n");
	printf("                       -applymat in.json  reslice the moving image onto the stationary (target)\n");
	printf("                                 grid using a saved matrix (no registration). The target may have any\n");
	printf("                                 resolution/FOV/origin sharing the fixed world frame. -final sets interp.\n");
	printf("                       -master grid.nii  register at the stationary resolution, but reslice the\n");
	printf("                                 output onto THIS grid (e.g. a higher-res template sharing the\n");
	printf("                                 stationary world frame). Works with both engines. -final sets interp.\n");
	printf("                       -reface XX  ANONYMIZE: register the moving (subject) to the stationary\n");
	printf("                                 (template), back-project the face-replacement shell XX (a\n");
	printf("                                 template-space image) onto the ORIGINAL subject grid, and\n");
	printf("                                 composite an anonymized image (shell>0 replace scaled,\n");
	printf("                                 ==0 keep, <0 zero; 3dBrickStat mean-ratio brightness match +\n");
	printf("                                 masked edge blend). Output stays on the subject grid. Works\n");
	printf("                                 with both engines; honors -weight; -robustfov is\n");
	printf("                                 fast-engine only. Emulates AFNI afni_refacer2 -mode_reface.\n");
	printf("                       -qwarp  NONLINEAR (deformable) registration; EXCLUSIVE mode. Only form:\n");
	printf("                                 allineate <moving> <stationary> -qwarp <output>. Inputs must already\n");
	printf("                                 be unifized + affine-aligned + skull-stripped + share the stationary grid.\n");
	printf("                                 Port of AFNI 3dQwarp -blur 0 3. No other option is accepted with it.\n");
	printf("                       -com  set the origin to the brightness center of mass (header-only\n");
	printf("                                 translation of the affine); a cheap centered start for\n");
	printf("                                 symmetric images. Runs after -robustfov, before -sym.\n");
	printf("                       -sym  template-free midsagittal (MSP) alignment. Standalone (no\n");
	printf("                                 stationary): reslice symmetric about world X=0. With a\n");
	printf("                                 stationary image: seed the moving header before registration.\n");
	printf("                       -symd  like -sym but first snaps the frame to axis-aligned (treats the\n");
	printf("                                 voxel grid as anatomical) so an obliquely-acquired but grid-\n");
	printf("                                 symmetric head is not rotated onto the oblique world frame.\n");
	printf("                       -symb  auto-compete: run both -sym and -symd; keep de-oblique only if its\n");
	printf("                                 correction rotation is smaller by >0.5 deg, else keep the world\n");
	printf("                                 frame (ties favor the original frame; fit cost is not compared).\n");
	printf("                       -nosagseed  disable the -sagseed in-MSP rigid seed. -sagseed is on by\n");
	printf("                                 default when -sym runs with a stationary template: after -sym\n");
	printf("                                 centers the MSP, it recovers the y-shift/z-shift/pitch DOF -sym\n");
	printf("                                 cannot see, completing a full-rigid seed for the main fit.\n");
	printf("                       -zoom  flag that size may be abnormal (e.g. an infant brain vs an adult\n");
	printf("                                 template): adds a global isotropic scale to the -sagseed seed\n");
	printf("                                 (needs -sym + template) AND relaxes the main affine scale range\n");
	printf("                                 to [0.5,2.0] (default 0.711..1.406). Other regularization and\n");
	printf("                                 the default (adult) behavior are unchanged.\n");
	printf("                       -p <threads>  set the maximum number of parallel threads\n");
	printf("                                 (0 = use all available); only affects OpenMP builds\n");
	printf("                       default cost: fast (no -cost == -cost fast); use -cost hel for AFNI\n");
	printf("                                 3dAllineate-style behavior, and -source_automask with lpc/lpa\n");
	return 0;
}

/* Deep-copy the header + voxel data of a nifti_image (the fields the affine reslice and
   writer read: dims, sform/qform, pixdim, and the data buffer). Filenames and extensions
   are dropped — the writer sets a fresh output name — so nifti_image_free() on the copy
   frees only its own data. Used by -master to preserve the moving image as it enters the
   allineate engine (which reslices it in place) so it can be re-resliced onto the master
   grid afterward. Returns a malloc'd image (free with nifti_image_free) or NULL. */
static nifti_image *nim_deep_copy(const nifti_image *s) {
	if (!s || !s->data) return NULL;
	nifti_image *d = (nifti_image *)malloc(sizeof *d);
	if (!d) return NULL;
	*d = *s;                     /* copies all scalar/array header fields (dims, sto/qto, pixdim) */
	d->fname = NULL; d->iname = NULL;    /* owned by the source; the writer sets fresh names */
	d->num_ext = 0;  d->ext_list = NULL; /* drop extensions (not needed for the resliced output) */
	size_t nbytes = (size_t)s->nvox * s->nbyper;
	d->data = malloc(nbytes ? nbytes : 1);
	if (!d->data) { free(d); return NULL; }
	memcpy(d->data, s->data, nbytes);
	return d;
}

/* Resolve the OUTPUT-warp interpolation: -final if set, else the mode default (cubic). This
   is the documented default→cubic rule (see allineate.h AL_INTERP_DEFAULT); shared by the
   -applymat, fast-engine, and -master reslice sites so the default lives in one place. */
static int resolve_output_interp(const al_opts *o) {
	return (o->final_interp == AL_INTERP_DEFAULT) ? AL_INTERP_CUBIC : o->final_interp;
}

/* Out-of-FOV output fill for the apply-affine CLI paths (the normal-registration path
   honors opts.fillmode inside nii_allineate). al_image_fillv extracts float internally so
   AUTO sees negative Hounsfield air even when the moving image is a raw int16 CT. */
static float resolve_output_fill(const al_opts *o, nifti_image *moving) {
	return al_image_fillv(o->fillmode, moving);
}

/* Finalize output: for stdout, force single-file NIfTI-1 and set names to "-";
   otherwise set the output filename. Then write. Returns 0 on success. */
static int write_result(nifti_image *out, const char *output_name, int isStdOut) {
	if (isStdOut) {
		free(out->fname); out->fname = nifti_strdup("-");
		free(out->iname); out->iname = nifti_strdup("-");
		out->nifti_type = NIFTI_FTYPE_NIFTI1_1;
	} else if (nifti_set_filenames(out, output_name, 0, 0)) {
		fprintf(stderr, "Failed to set output filename '%s'\n", output_name);
		return 1;
	}
	if (nifti_image_write_status(out)) {
		fprintf(stderr, "Failed to write output '%s'\n", output_name);
		return 1;
	}
	return 0;
}

/* Emit `s` as a JSON string literal (RFC 8259: escape \, ", and control bytes).
   Filenames may legally contain newline/tab/etc. on POSIX, which must be escaped
   or the emitted JSON is invalid. */
static void json_str(FILE *f, const char *s) {
	fputc('"', f);
	for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
		switch (*p) {
			case '\\': fputs("\\\\", f); break;
			case '"':  fputs("\\\"", f); break;
			case '\b': fputs("\\b", f);  break;
			case '\f': fputs("\\f", f);  break;
			case '\n': fputs("\\n", f);  break;
			case '\r': fputs("\\r", f);  break;
			case '\t': fputs("\\t", f);  break;
			default:
				if (*p < 0x20) fprintf(f, "\\u%04x", (unsigned)*p);
				else fputc(*p, f);
		}
	}
	fputc('"', f);
}

static void json_mat44(FILE *f, mat44 m, const char *key) {
	fprintf(f, "  \"%s\": [\n", key);
	for (int i = 0; i < 4; i++)
		fprintf(f, "    [%.10g, %.10g, %.10g, %.10g]%s\n",
		        m.m[i][0], m.m[i][1], m.m[i][2], m.m[i][3], i < 3 ? "," : "");
	fprintf(f, "  ]");
}

/* Save the fitted registration affine (from nii_last_affine()) as self-describing
   JSON. `fwd` is the world-space FIXED(base)->MOVING(source) "pull" transform.
   Returns 0 on success. */
static int write_affine_json(const char *path, mat44 fwd, const char *engine,
                             int dof, const char *cost_name,
                             const char *fixed_name, const char *moving_name,
                             const char *weight_name) {
	FILE *f = fopen(path, "w");
	if (!f) { fprintf(stderr, "Failed to open '%s' for -savemat\n", path); return 1; }
	mat44 inv = nifti_mat44_inverse(fwd);
	fprintf(f, "{\n");
	fprintf(f, "  \"type\": \"allineate_affine\",\n");
	fprintf(f, "  \"version\": 1,\n");
	fprintf(f, "  \"space\": \"world\",\n");
	fprintf(f, "  \"units\": \"mm\",\n");
	fprintf(f, "  \"engine\": "); json_str(f, engine); fprintf(f, ",\n");
	fprintf(f, "  \"dof\": %d,\n", dof);
	fprintf(f, "  \"cost\": "); json_str(f, cost_name); fprintf(f, ",\n");
	fprintf(f, "  \"fixed\": "); json_str(f, fixed_name ? fixed_name : ""); fprintf(f, ",\n");
	fprintf(f, "  \"moving\": "); json_str(f, moving_name ? moving_name : ""); fprintf(f, ",\n");
	/* Record the fine-stage region weight so two materially different fits (whole-head vs
	   ROI-weighted) have distinguishable provenance. Omitted entirely when unweighted. */
	if (weight_name) { fprintf(f, "  \"weight\": "); json_str(f, weight_name); fprintf(f, ",\n"); }
	fprintf(f, "  \"comment\": \"fixed_to_moving maps FIXED (stationary) world-mm to MOVING "
	           "(source) world-mm: the pull/resampling transform (for each fixed-grid voxel, "
	           "transform to moving world to sample). moving_to_fixed is its inverse. If "
	           "-com/-sym/-sagseed pre-steps ran, the matrix is relative to the seeded moving "
	           "header.\",\n");
	json_mat44(f, fwd, "fixed_to_moving"); fprintf(f, ",\n");
	json_mat44(f, inv, "moving_to_fixed"); fprintf(f, "\n");
	fprintf(f, "}\n");
	/* Read the stream error state BEFORE closing: `ferror(f) | fclose(f)` is unsequenced, so
	   fclose() could run first and ferror() would then inspect a closed stream (UB). */
	int werr = ferror(f);
	int cerr = fclose(f);
	if (werr || cerr) {
		fprintf(stderr, "Failed to write '%s' for -savemat\n", path);
		return 1;
	}
	return 0;
}

/* Read the 4x4 "fixed_to_moving" matrix from a -savemat JSON into `out`. Minimal
   parser: slurp the file, find the key, read the next 16 numbers (bracket/comma
   layout ignored; strtof handles scientific notation). Returns 0 on success. */
static int read_affine_json(const char *path, mat44 *out) {
	FILE *f = fopen(path, "rb");
	if (!f) { fprintf(stderr, "Failed to open matrix '%s'\n", path); return 1; }
	if (fseek(f, 0, SEEK_END) != 0) { fclose(f); fprintf(stderr, "Cannot read '%s'\n", path); return 1; }
	long n = ftell(f);
	if (n <= 0 || n > (1L << 20)) { fclose(f); fprintf(stderr, "Bad matrix file size '%s'\n", path); return 1; }
	rewind(f);
	char *buf = (char *)malloc((size_t)n + 1);
	if (!buf) { fclose(f); fprintf(stderr, "Memory allocation failed\n"); return 1; }
	size_t got = fread(buf, 1, (size_t)n, f);
	fclose(f);
	buf[got] = '\0';
	char *p = strstr(buf, "\"fixed_to_moving\"");
	if (!p) { free(buf); fprintf(stderr, "'%s' has no \"fixed_to_moving\" matrix\n", path); return 1; }
	p += strlen("\"fixed_to_moving\"");
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			while (*p && !((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' || *p == '.'))
				p++;
			char *end;
			out->m[i][j] = strtof(p, &end);
			if (end == p) { free(buf); fprintf(stderr, "'%s': could not read 16 numbers\n", path); return 1; }
			p = end;
		}
	}
	free(buf);
	return 0;
}

/* Parse CLI sub-arguments.
   *ac points to the last positional arg consumed; on return it points to the
   last sub-argument consumed. `reface_out` (may be NULL) receives the `-reface`
   shell filename — kept local to main.c so the shared al_opts/allineate.h stay
   unchanged (per the reface goal's shared-source policy). Returns 0 on success,
   1 on error. */
static int cli_parse_subopts(int *ac, int argc, char **argv, al_opts *opts,
                             const char *cmd_name, const char **reface_out) {
	while (*ac + 1 < argc && argv[*ac + 1][0] == '-') {
		(*ac)++;
		if (!strcmp(argv[*ac], "-p")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -p requires a thread count (0 = all available)\n", cmd_name);
				return 1;
			}
			char *end;
			long n = strtol(argv[*ac], &end, 10);
			if (*end != '\0' || n < 0 || n > 100000) {
				fprintf(stderr, "%s -p thread count must be a non-negative integer, got '%s'\n",
				        cmd_name, argv[*ac]);
				return 1;
			}
#if defined(_OPENMP)
			int nthreads = (n == 0) ? omp_get_num_procs() : (int)n;
			omp_set_num_threads(nthreads);
			fprintf(stderr, " + Using up to %d OpenMP thread%s\n", nthreads, nthreads == 1 ? "" : "s");
#else
			fprintf(stderr, " + -p ignored: built without OpenMP (single-threaded)\n");
#endif
		} else if (!strcmp(argv[*ac], "-skullstrip")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -skullstrip requires a brain mask filename\n", cmd_name);
				return 1;
			}
			opts->skullstrip = argv[*ac];
		} else if (!strcmp(argv[*ac], "-fast") || !strcmp(argv[*ac], "-fasthel")) {
			fprintf(stderr, "%s: -fast/-fasthel were replaced by cost selectors — use "
			                "'-cost fast'/'-cost fastx' (mixed HEL/CR fast engine), "
			                "'-cost fasthel' (fast engine, Hellinger only), or "
			                "'-cost fastcr' (fast engine, correlation-ratio), or "
			                "'-cost hel' (ordinary AFNI-style engine)\n", cmd_name);
			return 1;
		} else if (!strcmp(argv[*ac], "-coreg")) {
			fprintf(stderr, "%s: -coreg was replaced by '-cost fast' "
			                "(allineate is the default; '-cost fast' for the fast engine)\n", cmd_name);
			return 1;
		} else if (!strcmp(argv[*ac], "-robustfov")) {
			opts->robustfov = 170.0; /* default FOV in mm */
			if (*ac + 1 < argc) {
				char *end;
				double v = strtod(argv[*ac + 1], &end);
				if (end != argv[*ac + 1] && *end == '\0') {
					if (!(v > 0.0) || !isfinite(v)) {
						fprintf(stderr, "%s -robustfov value must be a positive number in mm, got '%s'\n",
						        cmd_name, argv[*ac + 1]);
						return 1;
					}
					opts->robustfov = v;
					(*ac)++;
				}
			}
		} else if (!strcmp(argv[*ac], "-reface")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -reface requires a shell image filename\n", cmd_name);
				return 1;
			}
			if (!reface_out) {
				fprintf(stderr, "%s -reface is not accepted here\n", cmd_name);
				return 1;
			}
			*reface_out = argv[*ac];
		} else {
			/* Delegate every shared registration option to the parser copied verbatim
			   from niimath's allineate.h. Back up to the last consumed argument first. */
			(*ac)--;
			int before = *ac;
			if (al_parse_subopts(ac, argc, argv, opts, cmd_name, AL_CAP_ALL))
				return 1;
			if (*ac == before)
				break;  /* unrecognized option: leave it for the positional validator */
		}
	}
	return 0;
}

/* Physical voxel volume (mm^3) from the |det| of a world transform's 3x3 linear block, normalized
   for xyz_units. Uses the coded transform that actually places/reslices the shell rather than the
   pixdim product, so a scaled/sheared sform (or pixdims disagreeing with the coded scale) cannot
   distort the reface coverage ratio and invert the <10% privacy gate. Mirrors niimath's
   reface_voxvol_mm3 (coreFLT.c); kept self-contained here since miniCoreFLT has no xyz_units_to_mm. */
static double reface_voxvol_mm3(mat44 M, int xyz_units) {
	double det = (double)M.m[0][0] * (M.m[1][1]*M.m[2][2] - M.m[1][2]*M.m[2][1])
	           - (double)M.m[0][1] * (M.m[1][0]*M.m[2][2] - M.m[1][2]*M.m[2][0])
	           + (double)M.m[0][2] * (M.m[1][0]*M.m[2][1] - M.m[1][1]*M.m[2][0]);
	double mm;
	switch (xyz_units) {           /* normalize to mm (see niimath core.c xyz_units_to_mm) */
		case NIFTI_UNITS_METER:  mm = 1000.0; break;
		case NIFTI_UNITS_MICRON: mm = 0.001;  break;
		default:                 mm = 1.0;    break;  /* NIFTI_UNITS_MM or unspecified */
	}
	return fabs(det) * (mm * mm * mm);
}

/* -reface: back-project the template-space shell onto the original subject grid and
   composite an anonymized image. `subject_orig` is the UNMODIFIED original moving
   image (captured before any -robustfov crop); `fixed_to_moving` is the world-mm
   FIXED(stationary)->MOVING transform from either engine. The shell shares the
   stationary template world frame. Reslice the shell onto the subject grid with NN
   (inverse transform), remove isolated speckle, then run the reface composite. The
   output stays on the original subject grid. Writes `output_name` only after every
   step succeeds. Returns 0 on success, nonzero on error. */
static int do_reface(nifti_image *subject_orig, const char *shell_name,
                     mat44 fixed_to_moving, const char *output_name, int isStdOut) {
	nifti_image *shell = nifti_image_read(shell_name, 1);
	if (!shell) { fprintf(stderr, "Failed to read -reface shell '%s'\n", shell_name); return 1; }
	/* Single-volume 3D shell with a usable coded transform (it defines the template
	   world frame; a no-form shell cannot be placed relative to the registration). A 4D
	   image has nvox = nx*ny*nz*nt with nt>1, so the nvox test alone rejects it. */
	if (shell->nvox != (size_t)shell->nx * shell->ny * shell->nz) {
		fprintf(stderr, "-reface shell '%s' must be a single-volume 3D image\n", shell_name);
		nifti_image_free(shell); return 1;
	}
	/* Require a USABLE coded transform, not merely a positive code: a shell with
	   sform_code>0 but a singular/non-finite sform would otherwise pass here yet make
	   nii_apply_affine silently synthesize a pixdim-centered frame (al_image_xform_or_pixdim),
	   placing the shell wrong while reporting success. al_image_xform fails on exactly that. */
	mat44 shell_xform;
	if (al_image_xform(shell, &shell_xform)) {
		fprintf(stderr, "-reface shell '%s' has no usable coded sform/qform (needed to place it in template space)\n",
		        shell_name);
		nifti_image_free(shell); return 1;
	}
	/* Convert the shell to physical float32 up front so the pre/post face-coverage counts
	   read the same values (nii_apply_affine then takes the float alias path). */
	if (nii_ensure_float32(shell)) {
		fprintf(stderr, "-reface: failed to convert the shell to float32\n");
		nifti_image_free(shell); return 1;
	}
	if (nii_ensure_float32(subject_orig)) {
		fprintf(stderr, "-reface: failed to convert the subject to float32\n");
		nifti_image_free(shell); return 1;
	}
	/* Face-coverage diagnostic (anonymization safety): a registration the engine reports
	   as "successful" but that is grossly wrong can land almost none of the shell's face
	   region on the subject, leaving the face largely UN-replaced in a written output. We
	   cannot verify the fit, but we can measure how much of the face volume mapped into the
	   subject FOV. Use physical VOLUME from the |det| of each image's coded transform (template
	   shell vs subject, separately) so the fraction is genuinely independent of the shell-vs-subject
	   resolution difference — the pixdim product would be wrong for a scaled/sheared sform. */
	size_t shell_n = (size_t)shell->nx * shell->ny * shell->nz;
	const float *sd0 = (const float *)shell->data;
	size_t face_tmpl = 0;
	for (size_t i = 0; i < shell_n; i++) if (sd0[i] > 0.0f) face_tmpl++;
	double vox_tmpl = reface_voxvol_mm3(shell_xform, shell->xyz_units);
	/* Back-project: shell (template world frame) -> subject grid, NN, fill 0. The
	   engine matrix is fixed->moving world; the inverse maps subject world -> template
	   world, exactly what nii_apply_affine samples for each subject voxel. */
	mat44 moving_to_fixed = nifti_mat44_inverse(fixed_to_moving);
	if (nii_apply_affine(shell, subject_orig, moving_to_fixed, AL_INTERP_NN, 0.0f)) {
		fprintf(stderr, "-reface: shell back-projection onto the subject grid failed\n");
		nifti_image_free(shell); return 1;
	}
	reface_isola((float *)shell->data, shell->nx, shell->ny, shell->nz);
	{
		size_t sub_n = (size_t)shell->nx * shell->ny * shell->nz;
		const float *sd1 = (const float *)shell->data;
		size_t face_subj = 0;
		for (size_t i = 0; i < sub_n; i++) if (sd1[i] > 0.0f) face_subj++;
		/* Subject-grid voxel volume from the |det| of the subject's own world transform — the frame
		   nii_apply_affine sampled the shell into — so template-vs-subject is on one physical scale. */
		mat44 subj_xform; al_image_xform_or_pixdim(subject_orig, &subj_xform, "reface subject");
		double vox_subj = reface_voxvol_mm3(subj_xform, subject_orig->xyz_units);
		double face_vol_tot = (double)face_tmpl * vox_tmpl;
		double cov = (face_vol_tot > 0.0) ? ((double)face_subj * vox_subj) / face_vol_tot : 0.0;
		fprintf(stderr, " + reface: %zu face voxels replaced (~%.0f%% of the shell face volume mapped into the subject FOV)\n",
		        face_subj, 100.0 * cov);
		/* FAIL CLOSED for privacy (matches niimath): -reface is an anonymization command, so if
		   almost none of the face shell reached the subject (mislocated fit or FOV mismatch) refuse
		   to write a possibly-unanonymized image rather than warn-and-succeed. */
		if (cov < 0.10) {
			fprintf(stderr, "-reface: only ~%.0f%% of the face shell mapped into the subject FOV -- the "
			                "registration likely failed; refusing to write a possibly-unanonymized image\n", 100.0 * cov);
			nifti_image_free(shell); return 1;
		}
	}
	nifti_image *out = NULL;
	int rc = reface_apply(subject_orig, shell, &out);
	nifti_image_free(shell);
	if (rc || !out) { fprintf(stderr, "-reface compositing failed\n"); return 1; }
	rc = write_result(out, output_name, isStdOut);
	nifti_image_free(out);
	return rc ? 1 : 0;
}

/* Exclusive -qwarp mode: `allineate <moving> <stationary> -qwarp <output>` — a faithful
   minimal port of `3dQwarp -blur 0 3 -source <moving> -base <stationary> -prefix <output>`
   (nonlinear stage only). No other option, extra operand, or stdin/stdout ('-') form is
   accepted, and the grammar is validated BEFORE any large image is read. Kept entirely in
   main.c (no al_opts field) per the qwarp goal; the engine lives in qwarp.c. */
static int run_qwarp(int argc, char *argv[]) {
	/* Enforce the EXACT positional grammar — argv = { prog, <moving>, <stationary>, -qwarp,
	   <output> } — not just "-qwarp appears somewhere". This rejects reordered forms such as
	   `moving stationary output -qwarp` or `moving -qwarp stationary output` that a
	   position-agnostic parse would silently accept, matching the documented single form. */
	if (argc != 5 || strcmp(argv[3], "-qwarp") != 0) {
		fprintf(stderr, "-qwarp is an exclusive mode; the only accepted form is:\n"
		                "  allineate <moving> <stationary> -qwarp <output>\n");
		return 1;
	}
	const char *moving_name = argv[1], *stationary_name = argv[2], *output_name = argv[4];
	/* the three operand slots must be real names, not a stray flag or a stdin/stdout '-' */
	if (moving_name[0] == '-' || stationary_name[0] == '-' || output_name[0] == '-') {
		fprintf(stderr, "-qwarp operands must be filenames (no options or stdin/stdout '-')\n");
		return 1;
	}

	nifti_image *moving = nifti_image_read(moving_name, 1);
	if (!moving) { fprintf(stderr, "Failed to read moving image '%s'\n", moving_name); return 1; }
	nifti_image *stationary = nifti_image_read(stationary_name, 1);
	if (!stationary) {
		fprintf(stderr, "Failed to read stationary image '%s'\n", stationary_name);
		nifti_image_free(moving); return 1;
	}
	nifti_image *out = NULL;
	int rc = qwarp_run(moving, stationary, &out);   /* atomic: out set only on full success */
	nifti_image_free(moving); nifti_image_free(stationary);
	if (rc || !out) return 1;                        /* reason already printed; no output written */
	rc = write_result(out, output_name, 0);
	nifti_image_free(out);
	return rc ? 1 : 0;
}

int main(int argc, char * argv[]) {
	if (argc < 4) {
		show_help();
		return (argc > 1) ? 1 : 0;
	}
	/* Exclusive -qwarp mode is handled entirely before the normal registration parsing. */
	for (int i = 1; i < argc; i++)
		if (!strcmp(argv[i], "-qwarp")) return run_qwarp(argc, argv);
#ifdef _WIN32
	/* Set binary mode for stdin/stdout when piping NIfTI data */
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
#endif
	int ac = 1;
	const char *moving_name = argv[ac];
	ac++;
	al_opts opts = al_opts_default();
	/* -reface shell filename: kept local (not in al_opts) per the reface goal. */
	const char *reface_name = NULL;
	/* The stationary image is optional: it is present only if the next token is
	   not a flag. Without it, we run preprocessing only (e.g. -robustfov). */
	const char *stationary_name = NULL;
	if (ac < argc && argv[ac][0] != '-') {
		stationary_name = argv[ac];
		ac++;
	}
	/* al_parse_subopts wants the index of the last positional consumed. */
	int pac = ac - 1;
	if (cli_parse_subopts(&pac, argc, argv, &opts, "allineate", &reface_name))
		return 1;
	ac = pac + 1;
	/* The output filename is the remaining positional (may be omitted in matrix-only
	   mode; validated after the option checks below). Options may appear on EITHER
	   side of the output, so parse any trailing flags after it too — this makes e.g.
	   `moving fixed out -p 1` work, not just `moving fixed -p 1 out`. */
	const char *output_name = (ac < argc) ? argv[ac] : NULL;
	if (output_name && ac + 1 < argc) {
		int pac2 = ac;   /* points at the output; consumes flags after it */
		if (cli_parse_subopts(&pac2, argc, argv, &opts, "allineate", &reface_name))
			return 1;
		if (pac2 + 1 < argc) {
			fprintf(stderr, "Unexpected extra argument '%s' after output '%s'\n",
			        argv[pac2 + 1], output_name);
			return 1;
		}
	}
	/* Only the moving image may be read from stdin ('-'). The stationary image
	   and the skullstrip mask must be files, otherwise the cached stdin buffer
	   would be silently reused, reading the same image twice. */
	if (stationary_name && stationary_name[0] == '-' && stationary_name[1] == '\0') {
		fprintf(stderr, "Only the moving image can be read from stdin ('-'); the stationary image must be a file\n");
		return 1;
	}
	if (opts.skullstrip && opts.skullstrip[0] == '-' && opts.skullstrip[1] == '\0') {
		fprintf(stderr, "The -skullstrip brain mask must be a file, not stdin ('-')\n");
		return 1;
	}
	if (opts.master && opts.master[0] == '-' && opts.master[1] == '\0') {
		fprintf(stderr, "The -master output grid must be a file, not stdin ('-')\n");
		return 1;
	}
	/* -sym has no defined meaning with -skullstrip: skullstrip's registration base
	   is the moving image used as a template, and seeding its header with an MSP
	   correction would silently move the mask target. Reject rather than ignore. */
	if (opts.sym && opts.skullstrip) {
		fprintf(stderr, "-sym cannot be combined with -skullstrip\n");
		return 1;
	}
	/* -savemat records the affine from a normal registration fit (nii_allineate);
	   it needs a stationary image and does not apply to the -skullstrip (deface) path. */
	if (opts.savemat && opts.skullstrip) {
		fprintf(stderr, "-savemat cannot be combined with -skullstrip\n");
		return 1;
	}
	if (opts.savemat && !stationary_name) {
		fprintf(stderr, "-savemat requires a <stationary> image to register against\n");
		return 1;
	}
	/* Default cost is the fast SPM/FLIRT-inspired engine for a plain registration (a moving+
	   stationary pair with no explicit -cost and no special mode). A fast failure on a tiny or
	   degenerate volume falls back to the Hellinger engine below, so a bare registration never
	   regresses. Modes with their own engine (-applymat, -skullstrip) and preprocessing-only runs
	   (no stationary) keep their paths; an explicit -cost
	   (fast/fastx/fasthel/fastcr or hel/lpc/lpa/ls) wins.
	   The fast engine cannot honor -zoom/-source_automask/-dark_automask/-warp/-interp; if the
	   user gave any of those WITHOUT an explicit -cost fast, stay on the ordinary engine rather
	   than defaulting to fast and erroring (an explicit -cost fast + such an option still errors
	   below). -com and -sym/-symd/-symb are header-only seeds applied below to BOTH engines, so
	   they do NOT disable default-fast. */
	int fast_incompatible = opts.zoom || opts.source_automask || opts.dark_automask ||
	                        (opts.cli_set & (AL_CLI_WARP | AL_CLI_INTERP));
	int fast_default = !(opts.cli_set & AL_CLI_COST) && !opts.fast &&
	                   stationary_name && !opts.applymat && !opts.skullstrip && !fast_incompatible;
	if (fast_default)
		opts.fast = AL_ENGINE_FAST_X;
	/* The fast engine (-cost fast/-cost fastx/-cost fasthel/-cost fastcr) is a distinct
	   estimator: it needs a
	   stationary image. It consumes the -com/-sym header seeds (applied below, before the
	   engine branch) but cannot compose with -skullstrip or -zoom (the latter relaxes the
	   affine scale range the fast pyramid's fixed scale capture cannot widen). */
	if (opts.fast) {
		const char *fflag = (opts.fast == AL_ENGINE_FAST_HEL) ? "-cost fasthel" :
		                    (opts.fast == AL_ENGINE_FAST_CR)  ? "-cost fastcr" : "-cost fast/fastx";
		if (!stationary_name && !opts.applymat) {
			fprintf(stderr, "%s requires a <stationary> image to register against\n", fflag);
			return 1;
		}
		if (opts.skullstrip || opts.zoom) {
			fprintf(stderr, "%s cannot be combined with -skullstrip/-zoom "
			                "(-zoom relaxes the affine scale range the fast pyramid cannot widen)\n", fflag);
			return 1;
		}
		/* The fast engine has a FIXED configuration (the cost is chosen by the selector —
		   correlation-ratio for '-cost fastcr', Hellinger for '-cost fasthel', HEL/CR coarse
		   competition for '-cost fast'/'-cost fastx' — 12-DOF affine, its
		   own matching interpolation, its own internal COM seeding). Reject allineate tuning options
		   the user explicitly passed rather than silently ignoring them (an ignored option would
		   also make -savemat metadata misleading). Only -final (output interpolation) and
		   operational flags (-p, -savemat, -robustfov, -com) apply to the fast path. */
		if (opts.cli_set & (AL_CLI_COST | AL_CLI_WARP)) {
			fprintf(stderr, "%s uses a fixed cost function and 12-DOF affine; "
			                "-warp and a further allineate -cost are not supported with it\n", fflag);
			return 1;
		}
		if (opts.cli_set & AL_CLI_INTERP) {
			fprintf(stderr, "%s ignores -interp (matching interpolation is fixed); "
			                "use -final to set the OUTPUT interpolation\n", fflag);
			return 1;
		}
		/* Fast initialization overrides are honored. Plain mode selects between the
		   supplied affine and a COM-recentered frame from their initial HEL/overlap
		   score. -com has already recentered the header and forces that frame;
		   -nocmass forces the supplied affine. Wired into cfo.use_cmass below. */
		if (opts.source_automask || opts.dark_automask) {
			fprintf(stderr, "%s does not support -source_automask/-dark_automask\n", fflag);
			return 1;
		}
	}
	/* -weight is a base(stationary)-space GRADED registration weight (AFNI 3dAllineate style),
	   supported by BOTH engines. It requires a real registration, so reject preprocessing-only,
	   -applymat (no registration), and -skullstrip (which uses its own template-registration path
	   that does not consume the weight) rather than silently ignoring it. */
	if (opts.weight) {
		if (!stationary_name) {
			fprintf(stderr, "-weight requires a <stationary> image (it weights the registration)\n");
			return 1;
		}
		if (opts.applymat || opts.skullstrip) {
			fprintf(stderr, "-weight is not supported with -applymat or -skullstrip\n");
			return 1;
		}
	}
	/* Reject stdin for -weight: only the moving image may be piped, and its cached buffer
	   would otherwise be re-read and mistaken for a plausible weight when dims match. */
	if (opts.weight && !strcmp(opts.weight, "-")) {
		fprintf(stderr, "-weight does not support stdin ('-'); give a weight-image file\n");
		return 1;
	}
	/* -applymat is a standalone apply mode: reslice the moving image onto the
	   stationary (target) grid using a saved world-space matrix; it does NO
	   registration, so it is exclusive with the registration/preprocessing modes. */
	if (opts.applymat) {
		if (!stationary_name) {
			fprintf(stderr, "-applymat requires a <target> image (it defines the output grid)\n");
			return 1;
		}
		if (opts.fast || opts.savemat || opts.skullstrip || opts.sym || opts.com || opts.robustfov > 0.0) {
			fprintf(stderr, "-applymat does no registration; it cannot be combined with "
			                "-cost fast/fastx/fasthel/fastcr/-savemat/-skullstrip/-sym/-com/-robustfov\n");
			return 1;
		}
		if ((opts.cli_set & (AL_CLI_COST | AL_CLI_WARP | AL_CLI_INTERP | AL_CLI_CMASS)) ||
		    opts.source_automask || opts.dark_automask) {
			fprintf(stderr, "-applymat ignores registration options "
			                "(-cost/-warp/-interp/-cmass/-source_automask/-dark_automask); "
			                "only -final sets the output interpolation\n");
			return 1;
		}
	}
	/* -master: register at the stationary resolution but reslice the result onto a
	   different (typically higher-res) grid that shares the stationary world frame.
	   It needs a real registration (moving+stationary) and an output image. -applymat
	   already takes its own <target> grid, and -skullstrip has fixed output semantics. */
	if (opts.master) {
		if (!stationary_name) {
			fprintf(stderr, "-master needs a <stationary> image (it reslices a registration result)\n");
			return 1;
		}
		if (opts.applymat || opts.skullstrip) {
			fprintf(stderr, "-master is not supported with -applymat/-skullstrip\n");
			return 1;
		}
		if (!output_name) {
			fprintf(stderr, "-master produces an output image; provide an output filename\n");
			return 1;
		}
	}
	/* -reface: register the subject (moving) to the template (stationary), back-project the
	   face-replacement shell onto the ORIGINAL subject grid, and composite an anonymized image.
	   It has fixed output semantics (subject grid, NN shell), so it rejects the other output
	   modes and the header-seed pre-steps whose fitted matrix would be relative to a modified
	   moving header. -robustfov is honored ONLY with the fast engine. */
	int reface_fast_only = 0;   /* a fast-only reface option was supplied (disables default-fast fallback) */
	if (reface_name) {
		if (reface_name[0] == '-' && reface_name[1] == '\0') {
			fprintf(stderr, "The -reface shell must be a file, not stdin ('-')\n");
			return 1;
		}
		if (!stationary_name) {
			fprintf(stderr, "-reface requires a <stationary> template image to register against\n");
			return 1;
		}
		if (!output_name) {
			fprintf(stderr, "-reface produces an output image; provide an output filename\n");
			return 1;
		}
		if (opts.skullstrip || opts.applymat || opts.master || opts.savemat) {
			fprintf(stderr, "-reface cannot be combined with -skullstrip/-applymat/-master/-savemat\n");
			return 1;
		}
		if (opts.sym || opts.com) {
			fprintf(stderr, "-reface cannot be combined with the header-seed pre-steps -com/-sym/-symd/-symb "
			                "(the fitted matrix would be relative to a modified subject header)\n");
			return 1;
		}
		if ((opts.cli_set & (AL_CLI_FINAL | AL_CLI_INTERP)) || opts.fillmode != AL_FILL_AUTO) {
			fprintf(stderr, "-reface uses fixed nearest-neighbor shell resampling; "
			                "-final/-interp/-fill are not supported with it\n");
			return 1;
		}
		/* Tight option whitelist: -reface accepts only -cost, -weight, -p and (fast-engine)
		   -robustfov. Reject the ordinary-engine tuning knobs rather than let them silently alter
		   the fit or (for the fast-incompatible ones) silently switch reface onto the ordinary
		   engine. -nosagseed is caught via opts.sagseed==0 (default is 1). */
		if ((opts.cli_set & (AL_CLI_WARP | AL_CLI_CMASS)) || opts.source_automask ||
		    opts.dark_automask || opts.zoom || opts.sagseed == 0) {
			fprintf(stderr, "-reface does not accept registration-tuning options "
			                "(-warp/-cmass/-nocmass/-source_automask/-dark_automask/-zoom/-nosagseed); "
			                "it uses the engine defaults\n");
			return 1;
		}
		/* Fast-only option: the robust-FOV crop is only wired on the fast path. With an explicitly
		   selected ordinary cost it must error rather than be ignored. */
		int explicit_ordinary = (opts.cli_set & AL_CLI_COST) && !opts.fast;
		reface_fast_only = (opts.robustfov > 0.0);
		if (explicit_ordinary && reface_fast_only) {
			fprintf(stderr, "-reface with an explicit ordinary -cost does not support "
			                "-robustfov (fast-engine only)\n");
			return 1;
		}
	}
	/* The output image is optional only in matrix-only mode: a registration with
	   -savemat but no output filename saves just the transform. Every other mode
	   (preprocessing, -applymat, -skullstrip, plain registration) writes an image. */
	if (!output_name && !opts.savemat) {
		fprintf(stderr, "Missing output filename\n");
		return 1;
	}
	int isStdOut = output_name && output_name[0] == '-' && output_name[1] == '\0';
	/* If output has no recognized NIfTI extension, default to .nii.gz (unless stdout) */
	char *output_ext = NULL;
	if (output_name && !isStdOut && !nifti_find_file_extension(output_name)) {
		size_t len = strlen(output_name);
		output_ext = (char *)malloc(len + 8); /* .nii.gz + NUL */
		if (!output_ext) {
			fprintf(stderr, "Memory allocation failed\n");
			return 1;
		}
		snprintf(output_ext, len + 8, "%s.nii.gz", output_name);
		output_name = output_ext;
	}
	int rc = 1;
	/* Refuse to point -savemat at the FINAL (extension-resolved) output image path:
	   the two would clobber each other (the loser silently lost). Compared here,
	   after the .nii.gz default is applied, so `-savemat out.nii.gz out` is caught and
	   `-savemat out out` (image -> out.nii.gz) is not falsely rejected. */
	if (opts.savemat && output_name && !strcmp(opts.savemat, output_name)) {
		fprintf(stderr, "-savemat path must differ from the output image path ('%s')\n", output_name);
		goto cleanup;
	}
	nifti_image *moving = nifti_image_read(moving_name, 1);
	if (!moving) {
		fprintf(stderr, "Failed to read moving image '%s'\n", moving_name);
		goto cleanup;
	}
	/* -reface: retain the UNMODIFIED original subject (before any -robustfov crop). The
	   engine fits/reslices `moving` (possibly cropped); the composite and output use this
	   pristine copy on the original grid. robustfov preserves retained-voxel world
	   coordinates, so the world-mm fit matrix is valid for back-projection onto it. */
	nifti_image *subject_orig = NULL;
	if (reface_name) {
		subject_orig = nim_deep_copy(moving);
		if (!subject_orig) {
			fprintf(stderr, "-reface: failed to copy the subject image\n");
			nifti_image_free(moving);
			goto cleanup;
		}
	}
	/* Preprocessing on the moving image (changes dims + sform/qform). Applied
	   before registration so allineate sees the cropped geometry. */
	if (opts.robustfov > 0.0) {
		if (nii_ensure_float32(moving)) {
			fprintf(stderr, "Failed to convert '%s' to float32 for -robustfov\n", moving_name);
			nifti_image_free(moving); nifti_image_free(subject_orig);
			goto cleanup;
		}
		if (nifti_robustfov(moving, opts.robustfov)) {
			fprintf(stderr, "robustfov failed on '%s'\n", moving_name);
			nifti_image_free(moving); nifti_image_free(subject_orig);
			goto cleanup;
		}
	}
	/* -com: reset the origin to the brightness center of mass (header-only). Runs
	   after -robustfov (so it sees the cropped geometry) and before -sym /
	   registration, giving a centered starting point for symmetric images. */
	if (opts.com) {
		if (nii_center_of_mass(moving)) {
			fprintf(stderr, "-com failed on '%s'\n", moving_name);
			nifti_image_free(moving);
			goto cleanup;
		}
	}
	if (!stationary_name) {
		/* Preprocessing-only mode: no stationary image, so no registration.
		   Write the (preprocessed) moving image. */
		if (opts.skullstrip) {
			fprintf(stderr, "-skullstrip requires a <stationary> image (the scan to strip)\n");
			nifti_image_free(moving);
			goto cleanup;
		}
		if (opts.robustfov <= 0.0 && !opts.sym && !opts.com) {
			fprintf(stderr, "No operation requested: give a <stationary> image to register, or a preprocessing option such as -robustfov, -com, or -sym\n");
			nifti_image_free(moving);
			goto cleanup;
		}
		if (opts.sym) {
			/* Standalone MSP alignment: reslice the data symmetric about world X=0. */
			if (nii_ensure_float32(moving)) {
				fprintf(stderr, "Failed to convert '%s' to float32 for -sym\n", moving_name);
				nifti_image_free(moving);
				goto cleanup;
			}
			if (nii_symmetry(moving, NULL, 1 /*reslice*/, opts.sym_deoblique, opts.dark_automask)) {
				fprintf(stderr, "-sym failed on '%s'\n", moving_name);
				nifti_image_free(moving);
				goto cleanup;
			}
		}
		if (write_result(moving, output_name, isStdOut)) {
			nifti_image_free(moving);
			goto cleanup;
		}
		nifti_image_free(moving);
		rc = 0;
		goto cleanup;
	}
	nifti_image *stationary = nifti_image_read(stationary_name, 1);
	if (!stationary) {
		fprintf(stderr, "Failed to read stationary image '%s'\n", stationary_name);
		nifti_image_free(moving);
		nifti_image_free(subject_orig);   /* NULL unless in -reface mode */
		goto cleanup;
	}
	if (opts.applymat) {
		/* Apply a saved world-space FIXED->MOVING matrix: reslice the moving image
		   onto the stationary (target) grid. No registration. -final controls interp
		   (default cubic; use -nearest for label/atlas volumes). */
		mat44 ftm;
		if (read_affine_json(opts.applymat, &ftm)) {
			nifti_image_free(moving); nifti_image_free(stationary); goto cleanup;
		}
		int interp = resolve_output_interp(&opts);
		if (nii_apply_affine(moving, stationary, ftm, interp, resolve_output_fill(&opts, moving))) {
			fprintf(stderr, "-applymat failed on '%s'\n", moving_name);
			nifti_image_free(moving); nifti_image_free(stationary); goto cleanup;
		}
		nifti_image_free(stationary);
		if (write_result(moving, output_name, isStdOut)) { nifti_image_free(moving); goto cleanup; }
		fprintf(stderr, " + Applied '%s' -> output on target grid\n", opts.applymat);
		nifti_image_free(moving);
		rc = 0;
		goto cleanup;
	}
	if (opts.skullstrip) {
		/* Skullstrip mode: nii_deface registers stationary (input) to moving
		   (template), warps the template-space brain mask onto the stationary
		   grid, and zeros non-brain voxels. Output is the stationary image. */
		nifti_image *mask = nifti_image_read(opts.skullstrip, 1);
		if (!mask) {
			fprintf(stderr, "Failed to read brain mask '%s'\n", opts.skullstrip);
			nifti_image_free(moving);
			nifti_image_free(stationary);
			goto cleanup;
		}
		int ret = nii_deface(stationary, moving, mask, opts);
		nifti_image_free(mask);
		nifti_image_free(moving);
		if (ret) {
			fprintf(stderr, "Skullstrip failed\n");
			nifti_image_free(stationary);
			goto cleanup;
		}
		if (write_result(stationary, output_name, isStdOut)) {
			nifti_image_free(stationary);
			goto cleanup;
		}
		nifti_image_free(stationary);
	} else {
	  /* -sym pre-step (header-only seed, consumed by EITHER engine): fold the midsagittal
	     correction into the moving header as an initial estimate before registering to the
	     template, then -sagseed (default on) recovers the 3 in-MSP DOF -sym is blind to via a
	     constrained fit to the template. Applied here, before the engine branch, so the fast
	     engine starts from the seeded pose (as it already does for the -com seed above). */
	  if (opts.sym) {
		if (nii_ensure_float32(moving)) {
			fprintf(stderr, "Failed to convert '%s' to float32 for -sym\n", moving_name);
			nifti_image_free(moving); nifti_image_free(stationary);
			goto cleanup;
		}
		if (nii_symmetry(moving, NULL, 0 /*seed header, no reslice*/, opts.sym_deoblique, opts.dark_automask)) {
			fprintf(stderr, "-sym pre-step failed on '%s'\n", moving_name);
			nifti_image_free(moving); nifti_image_free(stationary);
			goto cleanup;
		}
		if (opts.sagseed && nii_sagseed(moving, stationary, opts)) {
			fprintf(stderr, "-sagseed pre-step failed on '%s'\n", moving_name);
			nifti_image_free(moving); nifti_image_free(stationary);
			goto cleanup;
		}
	  }
	  if (opts.fast) {
		/* Fast SPM/FLIRT-inspired affine path (-cost fast/fastx/fasthel/fastcr).
		   Estimates a world-mm
		   FIXED->MOVING affine without mutating the inputs, then reslices the moving
		   image onto the stationary grid once. Result-based -savemat. */
		coreg_fast_opts cfo = coreg_fast_opts_default();
		cfo.cost = (opts.fast == AL_ENGINE_FAST_HEL) ? CF_COST_HEL :
		           (opts.fast == AL_ENGINE_FAST_CR)  ? CF_COST_CR  : CF_COST_HEL_CR;
		/* -com and -nocmass are strict overrides; otherwise auto-select initialization. */
		cfo.use_cmass = !opts.com &&
		                 !((opts.cli_set & AL_CLI_CMASS) && opts.cmass == AL_CMASS_NONE);
		/* -weight: a stationary-space region weight for the fine stage. Loaded here and freed
		   immediately after the estimate (coreg_fast_estimate only reads it during the call).
		   The CLI only owns what it must (read + stdin already rejected at parse); the estimator
		   boundary owns weight validation (dims + world frame + LS/3D) and its diagnostics. */
		nifti_image *weight_img = NULL;
		if (opts.weight) {
			weight_img = nifti_image_read(opts.weight, 1);
			if (!weight_img) {
				fprintf(stderr, "Failed to read -weight image '%s'\n", opts.weight);
				nifti_image_free(moving); nifti_image_free(stationary);
				nifti_image_free(subject_orig);   /* NULL unless in -reface mode */
				goto cleanup;
			}
			cfo.weight = weight_img;
			fprintf(stderr, " + Fine-stage region weighting from '%s'\n", opts.weight);
		}
		coreg_fast_result res;
		int est_rc = coreg_fast_estimate(moving, stationary, &cfo, &res);
		nifti_image_free(weight_img); weight_img = NULL;
		if (est_rc) {
			/* Default-fast fallback to the Hellinger engine keeps a bare registration robust.
			   For -reface it is suppressed when a fast-only reface option (-robustfov) was
			   supplied: it is not honored by the ordinary path, so silently falling back would
			   change semantics — fail instead. */
			if (fast_default && !reface_fast_only) {
				/* The default fast engine could not register this image (too small/degenerate for
				   its pyramid). Fall back to the robust Hellinger engine below so a bare
				   registration never regresses; an explicit fast-engine selector still errors.
				   moving/stationary are intact (the failed estimate does not mutate them). Any
				   -weight is honored on the fallback too — the ordinary engine now consumes it. */
				fprintf(stderr, " + Fast registration failed; falling back to -cost hel\n");
				opts.fast = 0;
				opts.cost = AL_COST_HELLINGER;
			} else {
				fprintf(stderr, "Fast registration failed\n");
				nifti_image_free(moving); nifti_image_free(stationary);
				nifti_image_free(subject_orig);
				goto cleanup;
			}
		} else {
		fprintf(stderr, "Fast registration: %d levels, %d evals, cost=%.5f, %.0f ms\n",
			res.levels_completed, res.evaluations, res.final_cost, res.registration_ms);
		/* -reface: back-project the shell onto the ORIGINAL subject grid and composite an
		   anonymized image, instead of the normal reslice-onto-stationary output. */
		if (reface_name) {
			int rr = do_reface(subject_orig, reface_name, res.fixed_to_moving, output_name, isStdOut);
			nifti_image_free(stationary);
			nifti_image_free(moving);
			nifti_image_free(subject_orig);
			rc = rr ? 1 : 0;
			goto cleanup;
		}
		/* Only reslice the moving image if an output image is requested. Matrix-only
		   mode (-savemat with no output) skips the full-resolution warp entirely — it
		   costs time/memory and an unused reslice failure must not fail a valid save. */
		if (output_name) {
			int interp = resolve_output_interp(&opts);
			/* -master: reslice onto the given grid (not the stationary grid). The fast
			   engine leaves `moving` unmutated, so no copy is needed — just target the
			   master grid with the same world-mm transform. */
			nifti_image *grid = stationary;
			nifti_image *master = NULL;
			if (opts.master) {
				master = nifti_image_read(opts.master, 1);
				if (!master) {
					fprintf(stderr, "Failed to read -master grid '%s'\n", opts.master);
					nifti_image_free(moving); nifti_image_free(stationary);
					goto cleanup;
				}
				grid = master;
			}
			int arc = nii_apply_affine(moving, grid, res.fixed_to_moving, interp, resolve_output_fill(&opts, moving));
			if (master) nifti_image_free(master);
			if (arc) {
				fprintf(stderr, "Fast registration final reslice failed\n");
				nifti_image_free(moving); nifti_image_free(stationary);
				goto cleanup;
			}
		}
		nifti_image_free(stationary);
		if (output_name && write_result(moving, output_name, isStdOut)) {
			nifti_image_free(moving); goto cleanup;
		}
		/* Record the validated fast-engine configuration actually used. */
		const char *fast_cost = (res.resolved_cost == CF_COST_LS) ? "ls" :
		                        (res.resolved_cost == CF_COST_HEL) ? "hel" :
		                        (res.resolved_cost == CF_COST_CR) ? "cr" : "hel+cr";
		if (opts.savemat &&
		    write_affine_json(opts.savemat, res.fixed_to_moving, "coreg_fast", res.resolved_dof,
		                      fast_cost, stationary_name, moving_name, opts.weight)) {
			fprintf(stderr, "Failed to save affine to '%s'\n", opts.savemat);
			nifti_image_free(moving); goto cleanup;
		}
		if (opts.savemat) fprintf(stderr, " + Saved affine to '%s'\n", opts.savemat);
		nifti_image_free(moving);
		}
	  }
	  if (!opts.fast) {
		/* Normal allineate engine (an explicit non-fast -cost, or the default-fast fallback above).
		   The -com/-sym/-sagseed header seeds were applied above (shared with the fast engine);
		   moving is already seeded as it enters the fit. */
		/* -reface: estimate the transform WITHOUT the reslice-then-discard that nii_allineate would
		   do (nii_allineate_estimate does not mutate `moving`), then composite on the ORIGINAL
		   subject grid. -master/-savemat are rejected with -reface, so this bypasses them cleanly. */
		if (reface_name) {
			mat44 aff;
			int rr = nii_allineate_estimate(moving, stationary, opts, &aff)
			         ? 1 : do_reface(subject_orig, reface_name, aff, output_name, isStdOut);
			nifti_image_free(stationary);
			nifti_image_free(moving);
			nifti_image_free(subject_orig);
			rc = rr ? 1 : 0;
			goto cleanup;
		}
		/* -master: nii_allineate reslices `moving` in place onto the stationary grid, so
		   preserve the moving image AS IT ENTERS the fit (after any -sym/-com/-robustfov
		   pre-steps — the fitted matrix is relative to that seeded header). After the fit
		   we re-reslice this copy onto the master grid with the same transform. */
		nifti_image *master_out = NULL;
		if (opts.master) {
			master_out = nim_deep_copy(moving);
			if (!master_out) {
				fprintf(stderr, "-master: failed to copy the moving image\n");
				nifti_image_free(moving); nifti_image_free(stationary);
				goto cleanup;
			}
		}
		/* Normal registration mode */
		int ret = nii_allineate(moving, stationary, opts);
		nifti_image_free(stationary);
		if (ret) {
			fprintf(stderr, "Registration failed\n");
			nifti_image_free(moving);
			if (master_out) nifti_image_free(master_out);
			nifti_image_free(subject_orig);
			goto cleanup;
		}
		if (opts.master) {
			/* Reslice the preserved pre-fit moving onto the master grid using the fitted
			   affine, then make it the image we write/return. The saved matrix is still the
			   stationary-frame transform (unchanged by the output grid). */
			mat44 aff;
			int interp = resolve_output_interp(&opts);
			nifti_image *grid = nifti_image_read(opts.master, 1);
			if (!grid || nii_last_affine(&aff) ||
			    nii_apply_affine(master_out, grid, aff, interp, resolve_output_fill(&opts, master_out))) {
				fprintf(stderr, "-master reslice onto '%s' failed\n", opts.master);
				if (grid) nifti_image_free(grid);
				nifti_image_free(master_out); nifti_image_free(moving);
				goto cleanup;
			}
			nifti_image_free(grid);
			nifti_image_free(moving);   /* discard the stationary-grid result */
			moving = master_out;        /* write_result/-savemat below operate on `moving` */
		}
		/* Write the registered image first (the primary deliverable), then the
		   -savemat JSON — so a failed image write never leaves a stale matrix
		   artifact for a command that reports failure. */
		if (output_name && write_result(moving, output_name, isStdOut)) {
			nifti_image_free(moving);
			goto cleanup;
		}
		if (opts.savemat) {
			mat44 aff;
			if (nii_last_affine(&aff) ||
			    write_affine_json(opts.savemat, aff, "allineate", opts.warp,
			                      al_cost_name(opts.cost), stationary_name, moving_name, NULL)) {
				fprintf(stderr, "Failed to save affine to '%s'\n", opts.savemat);
				nifti_image_free(moving);
				goto cleanup;
			}
			fprintf(stderr, " + Saved affine to '%s'\n", opts.savemat);
		}
		nifti_image_free(moving);
	  }
	}
	rc = 0;
cleanup:
	free(output_ext);
	return rc;
} //main()
