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
SRC    = main.c allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c coreg_fast.c
OUT    = allineate

# zlib is required; zstd is optional and experimental (ZSTD=1).
ZFLAGS = -DHAVE_ZLIB
ZLIB   = -lz
ifeq "$(ZSTD)" "1"
	ZSTD_CFLAGS := $(shell pkg-config --cflags libzstd 2>/dev/null)
	ZSTD_LIBS   := $(shell pkg-config --libs libzstd 2>/dev/null)
	ZFLAGS += -DHAVE_ZSTD $(ZSTD_CFLAGS)
	ZLIB   += $(if $(ZSTD_LIBS),$(ZSTD_LIBS),-lzstd)
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

.PHONY: all test parity sanitize debug profile clean

# `all` is phony so `make` ALWAYS rebuilds and overwrites the binary — no stale
# binary, no confusing "Nothing to be done". This is a whole-program compile with
# no .o files, so incremental tracking buys nothing and the full build is only a
# few seconds. The `-o $(OUT)` overwrites any existing executable in place.
all:
	rm -f $(OUT)
	$(CNAME) $(CFLAGS) $(OMPFLAGS) $(ZFLAGS) $(SRC) $(ZLIB) $(OMPLINK) -lm -o $(OUT)

# Deterministic correctness gate: builds generated non-facial fixtures plus two
# known-transform fixtures derived from the bundled non-subject MNI atlas, then asserts the
# geometry/preprocessing/affine paths. Exits nonzero on any failure, so CI or a release script
# can gate on it. Needs python3 + numpy + nibabel.
# The C-API harness links the estimator WITHOUT main.c to exercise coreg_fast_estimate()
# with NULL opts (the default-options contract) — a path the CLI never takes. Built here so
# `make test` also gates it; test_regression.py (§13) generates fixtures and runs it.
CAPI_OUT = test_capi_nullopts
test: all
	$(CNAME) $(CFLAGS) $(OMPFLAGS) $(ZFLAGS) -I. test/test_capi_nullopts.c \
		allineate.c nifti_io.c powell_newuoa.c miniCoreFLT.c coreg_fast.c \
		$(ZLIB) $(OMPLINK) -lm -o $(CAPI_OUT)
	python3 test/test_regression.py --allineate ./$(OUT)
	python3 test/test_coreg_fast.py

# Optional shared-source parity check — NOT part of `make test`, so a clean clone with no
# niimath checkout still builds and gates. Confirms the eight drop-in files are byte-for-byte
# identical to their niimath source of truth (silent drift here has twice shipped a stale
# nifti_io.c past a green `make test`). Supply the checkout path; exits nonzero listing any
# drifted file so a release/CI step can gate on it separately from the correctness gate:
#     make parity NIIMATH=/path/to/niimath
SHARED = allineate.c allineate.h coreg_fast.c coreg_fast.h powell_newuoa.c nifti_io.c nifti_io.h core32.h
parity:
	@test -n "$(NIIMATH)" || { echo "set NIIMATH=/path/to/niimath (its src/ holds the source of truth)"; exit 2; }
	@rc=0; for f in $(SHARED); do \
		if cmp -s "$$f" "$(NIIMATH)/src/$$f"; then echo "  ok    $$f"; \
		else echo "  DRIFT $$f"; rc=1; fi; \
	done; \
	if [ $$rc -ne 0 ]; then echo "** parity FAILED — resync drifted files from $(NIIMATH)/src"; fi; \
	exit $$rc

# AddressSanitizer build for heap-overflow / use-after-free testing.
#
# BUILT WITHOUT OpenMP ON PURPOSE — DO NOT ADD $(OMPFLAGS)/$(OMPLINK) HERE.
# On Apple Silicon macOS with Apple Clang, `-fsanitize=address` + Homebrew libomp
# HARD-DEADLOCKS at libomp's runtime init (verified: even a trivial `#pragma omp
# parallel for` hangs). `OMP_NUM_THREADS=1` does NOT help — the hang is at library
# load, before any parallel region. Omitting the OpenMP flags makes `#pragma omp`
# a no-op so ASan runs serial (slower, but it actually runs). The alternative is to
# build ASan with Homebrew LLVM clang instead of Apple Clang
# (CNAME=/opt/homebrew/opt/llvm/bin/clang), whose ASan+OpenMP works.
#
# LEAKS: LeakSanitizer is unavailable on Apple Silicon, so ASan here catches
# heap-overflow / use-after-free but NOT leaks. For leak detection use Xcode's
# `leaks` tool on a PLAIN (non-ASan) binary — not an ASan build:
#     make
#     MallocNanoZone=0 leaks --atExit -- ./allineate <moving> <stationary> out.nii.gz
# ASan is slow on full-size volumes (many interpolation trials); test small.
sanitize debug: $(SRC)
	$(CNAME) -O1 -g -Wno-deprecated -fsanitize=address -fno-omit-frame-pointer \
		$(ZFLAGS) $(SRC) $(ZLIB) -lm -o $(OUT)

# Release build with per-stage timing breakdown printed to stderr.
profile: $(SRC)
	$(CNAME) $(CFLAGS) $(OMPFLAGS) $(ZFLAGS) -DAL_PROFILE $(SRC) $(ZLIB) $(OMPLINK) -lm -o $(OUT)

clean:
	rm -f $(OUT) allineate.baseline test_capi_nullopts *.o
