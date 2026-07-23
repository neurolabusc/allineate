# Audit response — fastx mixed-default, NMS, portability, and hard-zero benchmarks

This response dispositions the external `audit_temp.md` review and the independent memory/resource, edge-case/side-effect, performance, and refactoring audits against the live allineate and niimath worktrees. The external review was based on an earlier snapshot, so several standing observations were already resolved when round two began; every claim below was rechecked against the current sources.

## Correctness and memory/resource safety

### Exploratory NMS candidate list — fixed and regression-guarded

The old in-place basin replacement could leave the list out of cost order and could retain two candidates closer than the intended angular floor. Exploratory candidates are now collected into a bounded pool, repeatedly selected in global cost order, and greedily accepted only when separated from every retained candidate in all three rotation axes. Enumeration order breaks equal-cost ties deterministically. Non-positive or non-finite separation/conversion parameters fail closed, which also resolves the reviewer's latent `min_deg == 0` duplicate-selection case.

`coreg_fast_nms.h` contains the shared selection and historical top-N insertion helpers. `test/test_coreg_fast_nms.c` directly covers cost ordering, equal-cost stability, the late-better-candidate failure mode, three-axis separation, invalid separation, non-finite/penalty costs, and the historical insertion helper.

### Later-strategy sample-allocation failure — fixed

An independent audit found a more serious path not identified by the external reviewer. If `cf_make_samples()` partially failed while rebuilding samples for a later hard-zero strategy, prior counts and partially replaced pointers could coexist. The previously successful strategy could then be restored and polished against inconsistent sample state.

`cf_make_samples()` now clears the previous sample set before rebuilding, and every degeneracy or allocation failure leaves all sample pointers, counts, weights, and statistics in a consistent empty state. A sample failure aborts all remaining levels and strategies and suppresses winner restoration/polish. A test-only partial-allocation hook fails the fourth sample build—the first level of a later hard-zero strategy—and asserts nonzero return plus byte-unchanged caller result storage.

### 32-bit float-buffer allocation overflow — fixed

The engines previously treated `nvox <= INT_MAX` as a sufficient allocation bound. That is incorrect on wasm32/native 32-bit: a count above `SIZE_MAX / sizeof(float)` can fit `int` while `nvox * sizeof(float)` wraps, producing an undersized allocation followed by out-of-bounds writes.

The shared `al_size_guard.h` now defines the portable float-buffer byte bound. It is applied at:

- ordinary allineate input/reslice boundaries;
- fast-engine input and derived-pyramid grids;
- qwarp input and padded grids;
- reface's public subject/shell boundary.

The C-API harnesses compile with a simulated `AL_SIZE_MAX=UINT32_MAX` on the 64-bit test host and present a 1,074,790,400-voxel header backed by a one-float dummy. All four APIs must reject before reading data, allocating, mutating source storage, or publishing a result. Qwarp separately asserts `*result == NULL`; fast registration asserts its caller-owned result remains byte-unchanged.

This guard complements niimath's `nvox_t`; it does not replace it. `nvox_t` is the signed count/index type for niimath core operations and permits 64-bit core processing when `FORCE_INT32_MAX` is absent. The shared registration engines still contain int-based algorithms and deliberately cap counts at `INT_MAX`. More importantly, even `int32_t nvox <= INT32_MAX` does not prove that a float allocation fits a 32-bit `size_t`; the byte bound is still required. Depending on `core.h` would also break allineate's self-contained drop-in contract. Therefore the shared, narrowly scoped byte guard is the correct parity mechanism.

### NEWUOA direct-call arithmetic — fixed

Both NEWUOA entry points formed quadratic point/workspace expressions in signed integer arithmetic. In-tree dimensions are small, but an oversized direct caller—or extreme `mfac`/`afac`—could overflow before allocation. A shared checked-shape helper now:

- rejects dimensions before `10 + 5 * ndim` can overflow;
- validates sampling-factor conversion;
- computes the triangular point limit and workspace terms with checked `size_t` multiplication/addition;
- requires the final workspace to fit the f2c integer length and `SIZE_MAX / sizeof(double)`;
- returns `-8` before allocation or callback when the shape is unsupported.

`test/test_powell_bounds.c` covers unconstrained/constrained `INT_MAX` dimensions and overflowing sampling factors, asserting that the objective callback is never entered.

### Resource ownership and leaks — verified

No additional leak, double-free, or stale-state defect was found. The independent review rechecked:

- per-job `cr_acc` ownership and fixed-order aggregation;
- sample-set teardown after every success/failure;
- qwarp pipeline/context cleanup;
- reface's atomic result ownership;
- ordinary and fast worker-local NEWUOA workspace teardown.

The release registration path reports zero leaks under Darwin `leaks`.

## Side effects, concurrency, and portability

### Compiler/TLS handling — centralized

`al_thread_local.h` is the single shared mapping:

- MSVC: `__declspec(thread)`;
- C11: `_Thread_local`;
- older GCC/Clang: `__thread`;
- serial builds: ordinary static storage.

All shared registration sources use it. niimath's optional GPL SPM module previously retained a private GNU `__thread` definition in `GPL/xalloc.h`; `SC_TLOCAL` now aliases the same `AL_THREAD_LOCAL`, and the real `GPL=1` build passes. An MSVC/OpenMP syntax probe also accepts the header.

### Persistent OpenMP worker state — fixed

Every parallel NEWUOA job now captures the worker's prior `mfac`/`afac`, installs the caller's requested factors, and restores the prior values afterward. This prevents one registration from leaking sampling configuration into persistent OpenMP workers while retaining thread-count-independent behavior.

### Failure atomicity and wasted work — tightened

Fast registration now stops immediately after optimizer allocation failure instead of running finer levels that cannot produce a valid result. Sample or optimizer failure prevents prior-winner restoration and high-DOF polish. Caller-visible result storage remains untouched until all required levels, cost validation, and affine validation succeed.

### Other edge cases — verified

- `-cost nmi` is a valid ordinary-engine selector backed by the shared AFNI-derived histogram cost. Although the reviewer considered an additional test unnecessary, an end-to-end regression now verifies parser dispatch and saved metadata (`engine=allineate`, `cost=nmi`).
- NMS rejects zero/non-finite thresholds and invalid/penalty costs.
- Candidate selection remains deterministic on equal costs and across thread counts.
- Fasthel/fastcr strategy isolation, max-DOF enforcement, qwarp transform selection, and output atomicity remain intact.

## Benchmark correctness

### Hard-zero path is now self-verifying

`benchmark/hardzero.py` computes the exact-minimum voxel fraction using the same strict `> 0.25` criterion as the engine. It validates both generated stripped templates and the tracked SSW base before running registrations. A future negative-valued template whose zero-filled background is not its minimum now fails explicitly rather than silently benchmarking the whole-head path.

Current fixture fractions are:

- SSW: 0.6218;
- generated MNI152 strip: 0.7470;
- generated avg152 strip: 0.7388.

### NCC handling — fixed

NCC is computed only for T1-labeled moving images. The guarded implementation rejects empty, non-finite, and zero-variance samples without NumPy warnings. An unavailable same-modal NCC prints `FAIL`, emits a diagnostic, increments the failure count, and makes the script exit nonzero; cross-modal rows show an intentional em dash and do no NCC work.

### Engine help — corrected

The `--engine both` help now states the actual selectors used: ordinary `allineate` plus `fast`, where `fast` is the mixed-default strategy. `fastx` remains documented as its explicit alias.

Focused benchmark-helper tests cover the exact-minimum gate, a negative-minimum rejection, valid/constant/non-finite NCC, and same-modal failure reporting.

## Performance and refactoring disposition

### Changes adopted

- Centralized the overlap-aware dependence score used by seed/candidate selection.
- Centralized the coarse angle set and historical sorted candidate insertion.
- Centralized TLS compiler handling and float-buffer byte limits.
- Added fail-fast behavior after unrecoverable sample/optimizer errors.

### Changes deliberately not made

- **Repeated-scan NMS:** at most six selections scan at most 750 candidates, negligible beside 750 volumetric cost evaluations. A comparator `qsort` is slower in WASM and adds tie-order concerns.
- **Approximately 84 KB candidate pool on the stack:** bounded and safe against niimath's 8 MB Windows stack, Emscripten's 4 MB stack, and WASI's 16 MB stack. Heap conversion adds an allocation/cleanup failure path without measured benefit.
- **Three hard-zero full-depth strategies:** measured cases show that pruning at 4 mm selects the wrong final winner. The expensive 9/12-DOF polish is already performed only once.
- **Full coarse-search consolidation:** the optimizer-parallel whole-head job and hard-zero multi-start path differ in grid shape, scale/z bracketing, NMS, inner-cost threading, and candidate propagation. Shared constants/helpers capture the true invariants without a mode-heavy hot-path abstraction.
- **Benchmark image caching:** scoring is outside registration timing, while retaining multiple 1 mm arrays would inflate memory and complicate a descriptive tool.

## Synchronization and validation

The shared drop-ins are byte-identical between allineate and `niimath/src/`, including `al_thread_local.h`, `al_size_guard.h`, `coreg_fast_nms.h`, allineate, coreg_fast, NEWUOA, qwarp, and reface.

Completed checks:

- allineate `make test`;
- benchmark helper, NMS, NEWUOA, simulated-wasm32 C-API boundaries;
- fast registration 17/17, including hard-zero ARC and `-p1 == -p4`;
- qwarp CLI 26/26 and qwarp C API 10/10;
- niimath OpenMP, serial, and `GPL=1` release smoke builds;
- Emscripten compilation of all shared engines and NEWUOA;
- MSVC/OpenMP TLS syntax probe;
- warning-focused C11 compilation;
- release-like serial `-O3` UBSan regression, fast-engine, and qwarp suites (no diagnostic);
- Darwin release-binary leak check (zero leaks);
- `make parity NIIMATH=/Users/chris/src/niimath`.
