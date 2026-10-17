# CFD simulator

3D incompressible Navier–Stokes bench, plus the same method as one bare CUDA C file for an RTX 3090 Ti (GA102, sm_86). The host notes are for a Ryzen 9 5900X.

The live volume is a collocated projection: semi-Lagrangian advection, Jacobi/SOR pressure, dye, a temperature field, and an imposed streamwise pressure gradient. Solids are voxel shapes you can swap in the Suite tab: sphere, cylinder, box, cone, plate, NACA 0012. The mid-span cut of a solid is a plane-stress finite-element section loaded by the fluid pressure.

The browser runs the reference solver on the CPU. It does not launch the GPU.

## Kernels

| File | Target |
| --- | --- |
| `kernels/karman3d.cu` | The 3D suite. CUDA C, sm_86, no CFD library |
| `kernels/karman_ga102.cu` | 2D CUDA C, 16×16 blocks, shared-memory Jacobi |
| `kernels/divergence_sm86.ptx` | Hand-written PTX divergence stencil |
| `kernels/znver3_ke.s` | AVX2 kinetic-energy reduction for Zen 3 |

```text
nvcc -O3 -arch=sm_86 --use_fast_math -o karman3d kernels/karman3d.cu
./karman3d karman  64 32 32 400
./karman3d airfoil 64 32 32 400 0.2
```

Presets: `karman`, `airfoil`, `cavity`, `plume`, `jet`, `kelvin`, `ink`. Arguments are preset, `nx ny nz`, steps, and an optional streamwise pressure gradient.

## Web bench

```text
npm install
npm run dev
```

Cases: Kármán street, NACA 0012, lid cavity, thermal plume, round jet, Kelvin–Helmholtz, stirred ink. Orbit the volume, scroll or pinch to zoom, and use Suite to change the solid, the temperatures, and the pressure gradient.
