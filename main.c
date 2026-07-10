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

#define kMTHdate "v1.0.20260315"
#define kMTHvers kMTHdate kOMPsuf kCCsuf

int show_help( void ) {
	printf("allineate version %s (%llu-bit %s)\n",kMTHvers, (unsigned long long) sizeof(size_t)*8, kOS);
	printf(" <moving> <stationary> [opts] <output>: co-register the moving image match the stationary image\n");
	printf(" <moving> [opts] <output>            : preprocess only (e.g. -robustfov), no registration\n");
	printf("                 Use '-' for <moving> to read from stdin, '-' for <output> to write to stdout\n");
	printf("                 opts: -cost XX (hel,lpc,lpa,ls) -cmass -nocmass -source_automask\n");
	printf("                       -dark_automask  ignore matched pairs where either image is at its\n");
	printf("                                 darkest value (background/zero-pad); uses each image's own\n");
	printf("                                 minimum so it is safe for signed data (e.g. CT Hounsfield).\n");
	printf("                       -interp XX (NN,linear,cubic) matching interpolation [default: linear]\n");
	printf("                         NOTE: linear is always used for the coarse pass;\n");
	printf("                               -interp only affects the fine pass.\n");
	printf("                               -interp does NOT affect the output image (use -final).\n");
	printf("                       -final XX  (NN,linear,cubic) output interpolation [default: cubic]\n");
	printf("                       -nearest -linear -cubic (shortcuts for -final)\n");
	printf("                       -warp XX (sho,shr,srs,aff) transform type [default: aff]\n");
	printf("                         sho=shift(3), shr=shift+rotate(6), srs=+scale(9), aff=+shear(12)\n");
	printf("                       -skullstrip XX  brain mask in moving space; output = stationary\n");
	printf("                                 with non-brain voxels set to darkest value [default final: linear]\n");
	printf("                       -robustfov [mm] crop the moving image to a robust FOV (default 170mm),\n");
	printf("                                 removing lower head/neck (emulates FSL robustfov); adjusts\n");
	printf("                                 dim and valid sform/qform. A good pre-step before registration.\n");
	printf("                       -savemat out.json  save the fitted registration affine (world-space\n");
	printf("                                 4x4, FIXED->MOVING and its inverse) as JSON; needs a stationary image.\n");
	printf("                                 The output image filename may be omitted to save only the matrix.\n");
	printf("                       -applymat in.json  reslice the moving image onto the stationary (target)\n");
	printf("                                 grid using a saved matrix (no registration). The target may have any\n");
	printf("                                 resolution/FOV/origin sharing the fixed world frame. -final sets interp.\n");
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
	printf("                       default cost: Hellinger; use -source_automask with lpc/lpa\n");
	return 0;
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
static int write_affine_json(const char *path, mat44 fwd, al_opts opts,
                             const char *fixed_name, const char *moving_name) {
	FILE *f = fopen(path, "w");
	if (!f) { fprintf(stderr, "Failed to open '%s' for -savemat\n", path); return 1; }
	mat44 inv = nifti_mat44_inverse(fwd);
	fprintf(f, "{\n");
	fprintf(f, "  \"type\": \"allineate_affine\",\n");
	fprintf(f, "  \"version\": 1,\n");
	fprintf(f, "  \"space\": \"world\",\n");
	fprintf(f, "  \"units\": \"mm\",\n");
	fprintf(f, "  \"dof\": %d,\n", opts.warp);
	fprintf(f, "  \"cost\": "); json_str(f, al_cost_name(opts.cost)); fprintf(f, ",\n");
	fprintf(f, "  \"fixed\": "); json_str(f, fixed_name ? fixed_name : ""); fprintf(f, ",\n");
	fprintf(f, "  \"moving\": "); json_str(f, moving_name ? moving_name : ""); fprintf(f, ",\n");
	fprintf(f, "  \"comment\": \"fixed_to_moving maps FIXED (stationary) world-mm to MOVING "
	           "(source) world-mm: the pull/resampling transform (for each fixed-grid voxel, "
	           "transform to moving world to sample). moving_to_fixed is its inverse. If "
	           "-com/-sym/-sagseed pre-steps ran, the matrix is relative to the seeded moving "
	           "header.\",\n");
	json_mat44(f, fwd, "fixed_to_moving"); fprintf(f, ",\n");
	json_mat44(f, inv, "moving_to_fixed"); fprintf(f, "\n");
	fprintf(f, "}\n");
	if (ferror(f) | fclose(f)) {
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
   last sub-argument consumed. Returns 0 on success, 1 on error. */
static int al_parse_subopts(int *ac, int argc, char **argv, al_opts *opts,
                            const char *cmd_name) {
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
		} else if (!strcmp(argv[*ac], "-cmass")) {
			opts->cmass = AL_CMASS_YES;
		} else if (!strcmp(argv[*ac], "-nocmass")) {
			opts->cmass = AL_CMASS_NONE;
		} else if (!strcmp(argv[*ac], "-source_automask")) {
			opts->source_automask = 1;
		} else if (!strcmp(argv[*ac], "-dark_automask")) {
			opts->dark_automask = 1;
		} else if (!strcmp(argv[*ac], "-nearest") || !strcmp(argv[*ac], "-NN")) {
			opts->final_interp = AL_INTERP_NN;
		} else if (!strcmp(argv[*ac], "-linear") || !strcmp(argv[*ac], "-trilinear")) {
			opts->final_interp = AL_INTERP_LINEAR;
		} else if (!strcmp(argv[*ac], "-cubic") || !strcmp(argv[*ac], "-tricubic")) {
			opts->final_interp = AL_INTERP_CUBIC;
		} else if (!strcmp(argv[*ac], "-warp")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -warp requires a type (sho, shr, srs, aff)\n", cmd_name);
				return 1;
			}
			if (al_parse_warp(argv[*ac], &opts->warp)) {
				fprintf(stderr, "Unknown warp '%s' (use: sho, shr, srs, aff)\n", argv[*ac]);
				return 1;
			}
		} else if (!strcmp(argv[*ac], "-interp")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -interp requires an interpolation name\n", cmd_name);
				return 1;
			}
			if (al_parse_interp(argv[*ac], &opts->interp)) {
				fprintf(stderr, "Unknown interp '%s' (use: NN, linear, cubic)\n", argv[*ac]);
				return 1;
			}
		} else if (!strcmp(argv[*ac], "-final")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -final requires an interpolation name\n", cmd_name);
				return 1;
			}
			if (al_parse_interp(argv[*ac], &opts->final_interp)) {
				fprintf(stderr, "Unknown final interp '%s' (use: NN, linear, cubic)\n", argv[*ac]);
				return 1;
			}
		} else if (!strcmp(argv[*ac], "-skullstrip")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -skullstrip requires a brain mask filename\n", cmd_name);
				return 1;
			}
			opts->skullstrip = argv[*ac];
		} else if (!strcmp(argv[*ac], "-savemat")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -savemat requires an output filename (.json)\n", cmd_name);
				return 1;
			}
			opts->savemat = argv[*ac];
		} else if (!strcmp(argv[*ac], "-applymat")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -applymat requires a matrix filename (.json from -savemat)\n", cmd_name);
				return 1;
			}
			opts->applymat = argv[*ac];
		} else if (!strcmp(argv[*ac], "-com")) {
			opts->com = 1;
		} else if (!strcmp(argv[*ac], "-sym")) {
			opts->sym = 1;
			opts->sym_deoblique = 0;   /* last-option-wins: -symd -sym runs plain -sym */
		} else if (!strcmp(argv[*ac], "-symd")) {
			opts->sym = 1;
			opts->sym_deoblique = 1;
		} else if (!strcmp(argv[*ac], "-symb")) {
			opts->sym = 1;
			opts->sym_deoblique = 2;   /* auto-compete: best of -sym / -symd */
		} else if (!strcmp(argv[*ac], "-nosagseed")) {
			opts->sagseed = 0;
		} else if (!strcmp(argv[*ac], "-zoom")) {
			opts->zoom = 1;
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
		} else if (!strcmp(argv[*ac], "-cost")) {
			(*ac)++;
			if (*ac >= argc) {
				fprintf(stderr, "%s -cost requires a cost function name\n", cmd_name);
				return 1;
			}
			if (al_parse_cost(argv[*ac], &opts->cost)) {
				fprintf(stderr, "Unknown cost function '%s' (use: lpc, lpa, hel, ls)\n", argv[*ac]);
				return 1;
			}
		} else {
			(*ac)--;
			break;
		}
	}
	return 0;
}

int main(int argc, char * argv[]) {
	if (argc < 4) {
		show_help();
		return (argc > 1) ? 1 : 0;
	}
#ifdef _WIN32
	/* Set binary mode for stdin/stdout when piping NIfTI data */
	_setmode(_fileno(stdin), _O_BINARY);
	_setmode(_fileno(stdout), _O_BINARY);
#endif
	int ac = 1;
	const char *moving_name = argv[ac];
	ac++;
	al_opts opts = al_opts_default();
	/* The stationary image is optional: it is present only if the next token is
	   not a flag. Without it, we run preprocessing only (e.g. -robustfov). */
	const char *stationary_name = NULL;
	if (ac < argc && argv[ac][0] != '-') {
		stationary_name = argv[ac];
		ac++;
	}
	/* al_parse_subopts wants the index of the last positional consumed. */
	int pac = ac - 1;
	if (al_parse_subopts(&pac, argc, argv, &opts, "allineate"))
		return 1;
	ac = pac + 1;
	/* The output filename is the remaining positional (may be omitted in matrix-only
	   mode; validated after the option checks below). Options may appear on EITHER
	   side of the output, so parse any trailing flags after it too — this makes e.g.
	   `moving fixed out -p 1` work, not just `moving fixed -p 1 out`. */
	const char *output_name = (ac < argc) ? argv[ac] : NULL;
	if (output_name && ac + 1 < argc) {
		int pac2 = ac;   /* points at the output; consumes flags after it */
		if (al_parse_subopts(&pac2, argc, argv, &opts, "allineate"))
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
	/* -applymat is a standalone apply mode: reslice the moving image onto the
	   stationary (target) grid using a saved world-space matrix; it does NO
	   registration, so it is exclusive with the registration/preprocessing modes. */
	if (opts.applymat) {
		if (!stationary_name) {
			fprintf(stderr, "-applymat requires a <target> image (it defines the output grid)\n");
			return 1;
		}
		if (opts.savemat || opts.skullstrip || opts.sym || opts.com || opts.robustfov > 0.0) {
			fprintf(stderr, "-applymat cannot be combined with registration/preprocessing options\n");
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
	/* Preprocessing on the moving image (changes dims + sform/qform). Applied
	   before registration so allineate sees the cropped geometry. */
	if (opts.robustfov > 0.0) {
		if (nii_ensure_float32(moving)) {
			fprintf(stderr, "Failed to convert '%s' to float32 for -robustfov\n", moving_name);
			nifti_image_free(moving);
			goto cleanup;
		}
		if (nifti_robustfov(moving, opts.robustfov)) {
			fprintf(stderr, "robustfov failed on '%s'\n", moving_name);
			nifti_image_free(moving);
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
		int interp = (opts.final_interp == AL_INTERP_DEFAULT) ? AL_INTERP_CUBIC : opts.final_interp;
		if (nii_apply_affine(moving, stationary, ftm, interp, 0.0f)) {
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
		/* Optional -sym pre-step: fold the midsagittal correction into the moving
		   image header as an initial estimate before registering to the template. */
		if (opts.sym) {
			if (nii_ensure_float32(moving)) {
				fprintf(stderr, "Failed to convert '%s' to float32 for -sym\n", moving_name);
				nifti_image_free(moving);
				nifti_image_free(stationary);
				goto cleanup;
			}
			if (nii_symmetry(moving, NULL, 0 /*seed header, no reslice*/, opts.sym_deoblique, opts.dark_automask)) {
				fprintf(stderr, "-sym pre-step failed on '%s'\n", moving_name);
				nifti_image_free(moving);
				nifti_image_free(stationary);
				goto cleanup;
			}
			/* -sagseed (default on): recover the 3 in-MSP DOF -sym is blind to
			   (y-shift, z-shift, pitch) via a constrained fit to the template,
			   completing the full-rigid seed. -nosagseed disables it. */
			if (opts.sagseed && nii_sagseed(moving, stationary, opts)) {
				fprintf(stderr, "-sagseed pre-step failed on '%s'\n", moving_name);
				nifti_image_free(moving);
				nifti_image_free(stationary);
				goto cleanup;
			}
		}
		/* Normal registration mode */
		int ret = nii_allineate(moving, stationary, opts);
		nifti_image_free(stationary);
		if (ret) {
			fprintf(stderr, "Registration failed\n");
			nifti_image_free(moving);
			goto cleanup;
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
			    write_affine_json(opts.savemat, aff, opts, stationary_name, moving_name)) {
				fprintf(stderr, "Failed to save affine to '%s'\n", opts.savemat);
				nifti_image_free(moving);
				goto cleanup;
			}
			fprintf(stderr, " + Saved affine to '%s'\n", opts.savemat);
		}
		nifti_image_free(moving);
	}
	rc = 0;
cleanup:
	free(output_ext);
	return rc;
} //main()
