# Makefile for allineate — standalone affine (12 DOF) NIfTI registration.
# Modeled on niimath's Makefile.
#
#   make            optimized release build (default) -> ./allineate
#   make test       build, then run the deterministic regression gate (test/test_regression.py)
#   make sanitize   AddressSanitizer build for heap-overflow / use-after-free testing
#                   (Apple Silicon: NOT leaks — use `leaks --atExit`, see below)
#   make debug      alias for `sanitize`
#   make profile    release build with per-stage timing (-DAL_PROFILE)
#   make clean      remove build products
#
# Options (append to any target):
#   CNAME=gcc-15    use a different compiler (default: clang / Apple Clang)
#   OMP=0           build without OpenMP (single-threaded)
#   ZSTD=1          enable experimental .zst support (needs libzstd)
#
# Registration runtime dominates, so the release build uses -O3 -ffast-math;
# the optimization penalty at build time is negligible next to the run time.

CNAME ?= clang
SRC    = main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c
OUT    = allineate

# zlib is required; zstd is optional and experimental (ZSTD=1).
ZFLAGS = -DHAVE_ZLIB
ZLIB   = -lz
ifeq "$(ZSTD)" "1"
	ZFLAGS += -DHAVE_ZSTD
	ZLIB   += -lzstd
endif

# OpenMP flags are platform-specific. Apple Clang needs libomp
# (brew install libomp); GCC/Clang elsewhere use -fopenmp directly.
# Homebrew GCC on macOS: override, e.g. `make CNAME=gcc-15 OMPFLAGS=-fopenmp OMPLINK=-fopenmp`.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OMPFLAGS ?= -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include
	OMPLINK  ?= -L/opt/homebrew/opt/libomp/lib -lomp
else
	OMPFLAGS ?= -fopenmp
	OMPLINK  ?= -fopenmp
endif
ifeq "$(OMP)" "0"
	OMPFLAGS =
	OMPLINK  =
endif

# Release optimization. -O2 works too with a small penalty; -O3 is the default.
# -fno-finite-math-only matches niimath's AFASTMATH: -ffast-math alone implies
# -ffinite-math-only, which makes the compiler assume no NaN/Inf (breaking the
# non-finite voxel filtering in allineate.c) and warns on INFINITY use. Keeping inf/nan
# semantics while still getting fast-math reassociation is load-bearing here.
CFLAGS ?= -O3 -ffast-math -fno-finite-math-only

.PHONY: all test sanitize debug profile clean

# `all` is phony so `make` ALWAYS rebuilds and overwrites the binary — no stale
# binary, no confusing "Nothing to be done". This is a whole-program compile with
# no .o files, so incremental tracking buys nothing and the full build is only a
# few seconds. The `-o $(OUT)` overwrites any existing executable in place.
all:
	rm -f $(OUT)
	$(CNAME) $(CFLAGS) $(OMPFLAGS) $(ZFLAGS) $(SRC) $(ZLIB) $(OMPLINK) -lm -o $(OUT)

# Deterministic correctness gate: builds synthetic (non-facial) fixtures and asserts
# the geometry/preprocessing/affine paths. Exits nonzero on any failure, so CI or a
# release script can gate on it. Needs python3 + numpy + nibabel.
test: all
	python3 test/test_regression.py --allineate ./$(OUT)

# AddressSanitizer build for heap-overflow / use-after-free testing (NOT leaks on
# Apple Silicon — see the LeakSanitizer note below).
# Built WITHOUT OpenMP: libomp's thread pool retains allocations that show up as
# noise, and ASan is far slower on the parallel interpolation paths.
# NOTE (Apple Silicon macOS): LeakSanitizer is unavailable, so ASan here catches
# heap-overflow / use-after-free but NOT leaks. For leak detection use Xcode's
# `leaks` tool on a PLAIN (non-ASan) binary — running `leaks` against an ASan build
# is not the right workflow. The optimized `make` binary works well, e.g.:
#     make
#     MallocNanoZone=0 leaks --atExit -- ./allineate <moving> <stationary> out.nii.gz
# (a plain `-O0 -g` build also works and is easier to symbolicate). ASan is slow on
# full-size volumes (the optimizer runs many interpolation trials); test small.
sanitize debug: $(SRC)
	$(CNAME) -O1 -g -Wno-deprecated -fsanitize=address -fno-omit-frame-pointer \
		$(ZFLAGS) $(SRC) $(ZLIB) -lm -o $(OUT)

# Release build with per-stage timing breakdown printed to stderr.
profile: $(SRC)
	$(CNAME) $(CFLAGS) $(OMPFLAGS) $(ZFLAGS) -DAL_PROFILE $(SRC) $(ZLIB) $(OMPLINK) -lm -o $(OUT)

clean:
	rm -f $(OUT) allineate.baseline *.o
