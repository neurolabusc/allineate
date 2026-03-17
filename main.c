#ifdef _WIN32
	#include <fcntl.h>
	#include <io.h>
#endif
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "nifti_io.h"
#include "allineate.h"

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
	printf("                 opts: -cost XX (hel,lpc,lpa,ls) -cmass -nocmass -source_automask\n");
	printf("                       -interp XX (NN,linear,cubic) matching interpolation [default: linear]\n");
	printf("                         NOTE: linear is always used for the coarse pass;\n");
	printf("                               -interp only affects the fine pass.\n");
	printf("                               -interp does NOT affect the output image (use -final).\n");
	printf("                       -final XX  (NN,linear,cubic) output interpolation [default: cubic]\n");
	printf("                       -nearest -linear -cubic (shortcuts for -final)\n");
	printf("                       -skullstrip XX  brain mask in moving space; output = stationary\n");
	printf("                                 with non-brain voxels set to darkest value [default final: linear]\n");
	printf("                       default cost: Hellinger; use -source_automask with lpc/lpa\n");
	return 0;
}

int main(int argc, char * argv[]) {
	if (argc < 4) {
		show_help();
		return (argc > 1) ? 1 : 0;
	}
	int ac = 1;
	const char *moving_name = argv[ac];
	ac++;
	const char *stationary_name = argv[ac];
	al_opts opts = al_opts_default();
	if (al_parse_subopts(&ac, argc, argv, &opts, "allineate"))
		return 1;
	ac++;
	if (ac >= argc) {
		fprintf(stderr, "Missing output filename\n");
		return 1;
	}
	const char *output_name = argv[ac];
	if (ac + 1 < argc)
		fprintf(stderr, "Warning: ignoring extra arguments after '%s'\n", output_name);
	/* If output has no recognized NIfTI extension, default to .nii.gz */
	char *output_ext = NULL;
	if (!nifti_find_file_extension(output_name)) {
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
	nifti_image *moving = nifti_image_read(moving_name, 1);
	if (!moving) {
		fprintf(stderr, "Failed to read moving image '%s'\n", moving_name);
		goto cleanup;
	}
	nifti_image *stationary = nifti_image_read(stationary_name, 1);
	if (!stationary) {
		fprintf(stderr, "Failed to read stationary image '%s'\n", stationary_name);
		nifti_image_free(moving);
		goto cleanup;
	}
	if (opts.skullstrip) {
		/* Skullstrip mode: register moving (template) to stationary, warp
		   brain mask, zero non-brain voxels. Output is the stationary image. */
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
		if (nifti_set_filenames(stationary, output_name, 0, 0)) {
			fprintf(stderr, "Failed to set output filename '%s'\n", output_name);
			nifti_image_free(stationary);
			goto cleanup;
		}
		nifti_image_write(stationary);
		nifti_image_free(stationary);
	} else {
		/* Normal registration mode */
		int ret = nii_allineate(moving, stationary, opts);
		nifti_image_free(stationary);
		if (ret) {
			fprintf(stderr, "Registration failed\n");
			nifti_image_free(moving);
			goto cleanup;
		}
		if (nifti_set_filenames(moving, output_name, 0, 0)) {
			fprintf(stderr, "Failed to set output filename '%s'\n", output_name);
			nifti_image_free(moving);
			goto cleanup;
		}
		nifti_image_write(moving);
		nifti_image_free(moving);
	}
	/* Verify the output was actually written (nifti_image_write returns void) */
	FILE *fcheck = fopen(output_name, "rb");
	if (fcheck) {
		fclose(fcheck);
		rc = 0;
	} else {
		fprintf(stderr, "Failed to write output '%s' (does the directory exist?)\n", output_name);
	}
cleanup:
	free(output_ext);
	return rc;
} //main()
