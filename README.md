# CFD simulator

A 3D incompressible bench, plus a staggered MAC reference that has been checked against a few laminar solutions. This is not an engineering-validated CFD package.

The picture in the browser is still a collocated CPU preview. It does not run on the GPU. The checked solver is `src/cfd/mac.ts` (double precision) and the host path of `kernels/karman3d.cu`. Stored pressure is kinematic, `φ = p/ρ`.

## Checks

```text
node --experimental-strip-types --test verify/*.test.mjs
node --experimental-strip-types scripts/mac-run.mjs
g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu
./karman3d
```

`scripts/mac-run.mjs` is the headless Couette check against `src/cfd/mac.ts`. Use it on the 5900X before comparing a CUDA binary. The sequence for the 3090 Ti is in [docs/target-machine.md](docs/target-machine.md). `scripts/build.sh` builds the host path first and calls `nvcc` only if it is present.

`nvcc` and a GPU were not available where those checks were run. Device execution is unverified. For a 3090 Ti, compile with `-arch=sm_86` and do not add `--use_fast_math` until that flag has been measured against the same checks.

```text
nvcc -O3 -arch=sm_86 -o karman3d kernels/karman3d.cu
```

What passed, and what did not, is in [docs/numerical-status.md](docs/numerical-status.md).

## Web bench

```text
npm install
npm run dev
```

The Suite tab can still place voxel solids and estimate a plane-stress section. That section is an experimental one-way load, not fluid-structure interaction.
