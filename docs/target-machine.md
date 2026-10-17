# Target machine

The 3090 Ti is the intended device. Nothing in this repository has been timed or executed on it. The browser picture is a collocated CPU preview. The checked solver is the TypeScript MAC reference and the host path of `kernels/karman3d.cu`.

Run these in order after the clone finishes. Do not add `--use_fast_math`.

## 1. Confirm the tree

```text
git log -1 --oneline
```

The baseline these notes were written against is `d2955a1`. If you are later than that, re-read `docs/numerical-status.md` before treating the numbers below as current.

## 2. Host check (no GPU)

This is the path that already passed where the repository was built.

```text
g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu
./karman3d
```

Expect the self-checks to print `PASS` for the cases they cover. A failure here means the host path is broken; do not move on to `nvcc`.

## 3. TypeScript MAC reference (no GPU)

```text
node --experimental-strip-types --test verify/*.test.mjs
```

Or the headless runner, which steps one Couette case and prints the wall shear:

```text
node --experimental-strip-types scripts/mac-run.mjs
```

Both use `src/cfd/mac.ts`. They are the numbers to compare against before you trust a CUDA result.

## 4. Confirm the 3090 Ti is visible

```text
nvidia-smi -L
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv
nvcc --version
```

A 3090 Ti reports compute capability 8.6. If `nvidia-smi` does not list a device, stop. The driver or the card is not available to this process.

## 5. First device build

```text
nvcc -O3 -arch=sm_86 -o karman3d kernels/karman3d.cu
./karman3d
```

`scripts/build.sh` does the host build first and only calls `nvcc` if it is on `PATH`. It never adds `--use_fast_math`.

## 6. What must still pass

After the first GPU binary runs:

- `./karman3d` prints the same self-check `PASS` lines as the host build.
- `node --experimental-strip-types --test verify/*.test.mjs` still passes. The GPU binary does not replace these tests.
- `node --experimental-strip-types scripts/mac-run.mjs` still prints a Couette error consistent with the host run.

Until those three agree, device execution remains unverified. A successful `nvcc` compile is not verification.

## What "device execution unverified" means

No process in this repository has called a CUDA kernel on a 3090 Ti and compared the result to the MAC reference. The PTX and AVX2 files are retained. They are not the authoritative path. `kernels/karman3d.cu` is. Do not report a timing, a speedup, or a force coefficient from the GPU until the checks above have been run on that machine and the numbers are recorded.

## What this is not

This is constant-density Newtonian laminar flow on a structured Cartesian grid, with optional passive scalar and Boussinesq buoyancy. It is not a general-purpose, engineering-validated CFD package. Compressible flow, shocks, multiphase, RANS/LES, AMR, and unstructured meshing are future work.
