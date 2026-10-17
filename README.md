# CFD simulator

Live 2D incompressible Navier–Stokes bench (Karman), plus the same method written for an RTX 3090 Ti and a Ryzen 9 5900X.

The interactive field is a collocated Chorin projection: semi-Lagrangian advection, implicit diffusion, and a pressure Poisson solve. On a cylinder in a channel the shedding sits near a Strouhal number of 0.20.

## Kernels

| File | Target |
| --- | --- |
| `kernels/karman_ga102.cu` | CUDA C, `sm_86`, 16×16 blocks, shared-memory Jacobi |
| `kernels/divergence_sm86.ptx` | Hand-written PTX divergence stencil |
| `kernels/znver3_ke.s` | AVX2 kinetic-energy reduction for Zen 3 |

```text
nvcc -O3 -arch=sm_86 --use_fast_math -c kernels/karman_ga102.cu
ptxas -arch=sm_86 -O3 kernels/divergence_sm86.ptx -o divergence_sm86.cubin
clang -c -o znver3_ke.o kernels/znver3_ke.s
```

The browser build runs the reference solver in `src/cfd/solver.ts`. It does not launch the GPU. The Rig and Kernels panels in the UI show the GA102 launch geometry and these sources.

## Web bench

```text
npm install
npm run dev
```

Presets: Kármán street, NACA 0012, lid-driven cavity, plane jet, thermal plume, Kelvin–Helmholtz, stirred ink.
