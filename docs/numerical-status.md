# Numerical status

## Phase 1 repair

The defects below were present at `0ba2500` and are fixed in the TypeScript MAC reference. The audit and the measured orders are in [docs/audit/current-state.md](audit/current-state.md) and [docs/audit/phase1-report.md](audit/phase1-report.md).

- Scalar reconstruction follows the face normal. A Courant-1 upwind shift matches the translated field to `1e-8` on x, y, and z.
- A scalar value of 0 is a real boundary state.
- A failed substep rolls back the fields, the clock, and the step counter. The report's substep count is the number actually committed.
- A closed inlet, and an outlet clamp that breaks mass balance, refuse the step instead of reporting a converged pressure solve.
- Case and restart documents are checked before the solver allocates.

The native file `kernels/karman3d.cu` was not brought up to this scheme. The browser picture is still the collocated preview. Taylor–Green remains near first order; that tolerance was not loosened.


The handoff sequence for the 3090 Ti is in [docs/target-machine.md](target-machine.md). `scripts/build.sh` builds the host path first. `scripts/mac-run.mjs` is the headless Couette check.

The Analysis panel lists experimental models in `src/cfd/experimental.ts`. They are validated for finite output by `verify/experimental.test.mjs`. They are not verified solvers.
## Earlier baseline

Baseline commit before the MAC reference: `aac72f47fddcd381d941dddcd600796b8bf00633`.
The MAC reference itself landed in `0ba2500`.

This environment has gcc 12.2.0 and Node 22. It does not have `nvcc` or a GPU. Device execution and any timing on an RTX 3090 Ti are unverified.

The browser picture is still the collocated CPU preview in `src/cfd/solver3d.ts`. It is not the verified scheme.

### Confirmed on the collocated preview

| # | Finding | Status |
| --- | --- | --- |
| 1 | CUDA computed `nu` and never applied it | Confirmed. The old step loop had no diffusion kernel. |
| 2 | Browser 3D diffusion used `min(dt·ν, 0.12)` | Confirmed in `solver3d.ts`. |
| 3 | Advection clipped the backtrace to 0.85 cells | Confirmed in the preview and the old CUDA source. |
| 4 | Collocated divergence, gradient, and Laplacian were not the same operator | Confirmed. Both used a centered half-difference that is not `div(grad)`. |
| 5 | Pressure used a fixed iteration count | Confirmed. CUDA stopped at 20 Jacobi iterations with no residual. |
| 6 | CUDA chose `dt` from the inlet speed | Confirmed. `maxs` was reset from `u_in` every step. |
| 7 | 3D curl dropped `ωx` and dissipation was not `2ν S:S` | Confirmed in `measure`. |
| 8 | Force was a pressure difference on a cell face, with no viscous traction | Confirmed. |
| 9 | The outlet branch assumed an x-facing outlet | Confirmed. |
| 10 | Browser and CUDA did not share temperature or the pressure drive | Confirmed. CUDA had no temperature field. |
| 11 | Nonfinite velocity was replaced with zero | Confirmed in `measure`. |
| 12 | The CUDA source was a second copy inside a TypeScript string | Confirmed. `kernels/karman3d.cu` is the file the UI imports. |

### Scheme checked at `0ba2500`

Incompressible constant-density Newtonian flow. Stored pressure is kinematic, `φ = p/ρ`.

Staggered MAC grid. The pressure matrix is the positive operator `−div(grad)` on the same free faces the projection corrects. Closed domains remove the mean of `φ` instead of pinning one cell. The reference solve is unpreconditioned CG with an absolute and a relative residual. A failed solve or a nonfinite field restores the step and does not advance time.

Momentum diffusion is explicit and subcycled at the viscous stability limit. `ν` is never clipped. Transport for the verified path is flux-form: first-order upwind, or limited linear reconstruction with Heun (MUSCL-RK2). On the Taylor–Green pair below the observed rate is about 1.1, not 2. Vorticity confinement is not part of this solver.

### Commands

```text
node --experimental-strip-types --test verify/*.test.mjs
g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu
./karman3d
```

### Results recorded at `0ba2500`

JavaScript reference, 13 tests, all passed:

- `div(grad)` on a non-cubic grid with an outlet agrees with an independent stencil to `1e-12`.
- A manufactured divergence on a closed `12×10×8` box drops below `1e-6` after one projection.
- A nonconverged CG solve does not advance time. A NaN is left in place.
- Mass from a south inlet leaves through a north outlet.
- One viscous step of a periodic shear mode matches the discrete eigenmode to `1e-12`.
- Couette at `t = 12`, `ny = 24`, `ν = 0.08`: velocity error below `1e-3`, wall shear within 2% of `μ U/H · Area`.
- Plane Poiseuille at `t = 8`, `ny = 32`: relative L2 below `0.03` against `u = (G/(2ν)) y (H−y)`.
- Periodic dye mass is conserved to `1e-10` by the upwind flux. A uniform stream stays uniform under MUSCL-RK2.
- Taylor–Green, `ν = 0.01`, `t = 0.5`, CFL `0.2`: relative velocity error `0.028` on `16²` and `0.013` on `32²`.

Native host path at that commit:

```text
decay  err 1.110e-16  PASS
project  div 2.487e-14  iters 4  residual 1.031e-10  PASS
gpu  nvcc=0  device-check=not-run  UNVERIFIED
OK
```

## Not claimed

This is not an engineering-validated CFD package. Lid-driven cavity data, cylinder shedding, and force coefficients on voxel bodies are not verified here. The interactive view still runs the old preview. Structural stress on the Suite tab is an experimental one-way plane-stress section, not fluid-structure interaction. Compressible flow, turbulence models, AMR, and two-way coupling are out of scope.
