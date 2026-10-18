# Upstream inspiration

These projects informed the design of features added here. No source was copied.
Ideas were reimplemented against this repository's MAC layout and tests.

| Project | What was taken as inspiration | What was not taken |
|---|---|---|
| CFD-GO/TCLB | Structure-of-arrays layout and a collide-then-stream split for lattice methods | Any template, kernel, or build file. GPL-3.0; design notes only |
| sailfish-team/sailfish | The idea of a separate collision and streaming stage, and of reporting a lattice viscosity | Python package, OpenCL kernels, examples |
| specter-cfd/SPECTER | Treating boundary conditions as data on a structured grid rather than as a special mesh | Solver code, discretization choices |
| sreelekhanampally/gpu-cfd-solvers | A Poisson smoother as its own kernel, with a residual check separate from the time step | Their Jacobi implementation |

## Rules

- Do not copy source from these repositories into this tree.
- GPL-licensed projects are design references only.
- A new numerical method lands only with a regression test against a defined solution.
- The checked path remains the staggered MAC projection in `src/cfd/mac.ts` and the host path of `kernels/karman3d.cu`.

## What was built from that inspiration

- `src/cfd/multigrid.ts` — a geometric V-cycle for a Cartesian Poisson problem. It is not yet the pressure solver inside `MacSolver.project`. The existing conjugate gradient remains the reference.
- `src/cfd/lattice.ts` — a tiny D2Q9 collide-and-stream step used by the experimental LBM reading. It is not a 3D lattice Boltzmann solver.
