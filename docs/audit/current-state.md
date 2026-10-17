# Current state at `0ba2500`

Recorded before the phase-1 repairs. Environment: Debian 12, g++ 12.2.0, Node 22.23.3. `nvcc` is absent. `nvidia-smi` is absent. No GPU was available, so nothing in this audit is a device measurement.

Baseline command, run on that commit's sources before editing:

```text
node --experimental-strip-types --test verify/numerics.test.mjs
```

Result: 13 tests, 13 passed, about 1.3 s. The native host program was not recompiled for this note; its last recorded result is in `docs/numerical-status.md` (`decay err 1.110e-16`, `project div 2.487e-14`). That program is not the full MAC solver.

Evidence labels: VERIFIED (a test or a direct run), CODE-CONFIRMED (read in the source), SUSPECTED, UNKNOWN.

## Who computes what

| Path | Where | What it does | Confidence |
| --- | --- | --- | --- |
| Browser picture | `src/cfd/solver3d.ts`, drawn by `src/components/bench.tsx` | Collocated preview on the CPU. Not the checked scheme. | VERIFIED. The preview still contains the clipped viscosity and the 0.85-cell backtrace. |
| MAC reference | `src/cfd/mac.ts` | Double-precision staggered projection, explicit viscosity, optional flux-form transport. | VERIFIED for the cases in `verify/numerics.test.mjs`. |
| Case file | `src/cfd/case.ts` | Version-1 document. Pressure is kinematic, φ = p/ρ. | CODE-CONFIRMED. |
| Native host | `kernels/karman3d.cu` under `-DKARMAN_CPU` | Periodic shear decay and a small projection. Not the TypeScript solver. | VERIFIED for those two checks only. |
| CUDA device | same file, without `KARMAN_CPU` | Kernels exist. They were not launched here. | UNKNOWN as an execution result. Compilation was not attempted: `nvcc` is missing. |
| Section stress | `src/cfd/fea.ts` | One-way plane-stress estimate on a slice. | CODE-CONFIRMED. Not fluid-structure interaction. |

## Scheme that was already checked

Constant-density Newtonian flow. MAC faces hold velocity. Cells hold φ, temperature, and dye. The pressure matrix is `−div(grad)` on the free faces. Closed domains remove the mean of φ. Viscosity is explicit and is not clipped. See `docs/numerical-status.md` for the Couette, Poiseuille, and Taylor–Green numbers from the previous pass.

## Defects confirmed before the repair

1. Scalar reconstruction was not axis-aware. `reconstruct` always sampled along x. `scalarFlux` used that same x-offset for y and z faces. VERIFIED by a one-step probe on commit `0ba2500`: a periodic upwind shift with a transversely varying scalar, Courant 1, n = 16, gave relative L2 errors 0.281 (x, positive), 0.326 (x, negative), 0.063 (y, positive), 0.326 (y, negative), 0.063 (z, positive), 0.326 (z, negative). A pure y-profile with positive v hid the bug, because the wrong-axis samples were equal. High confidence.

2. Outflow of a zero scalar used `inside || interior`. Zero was treated as missing. VERIFIED: west outflow, interior dye 0, exterior dye 5, one step, the boundary cell became −1. High confidence.

3. `step` saved the fields once, advanced `time` and `steps` after each successful substep, and on a later failure restored the fields but not the clock. `StepReport.substeps` was hard-coded to 1. CODE-CONFIRMED by reading `step` and `measure`. High confidence.

4. A closed box with an inlet reported `status: "ok"` with divergence L∞ = 0.5. The pressure solve removes the mean of the right-hand side, so algebraic convergence is not incompressibility. VERIFIED. High confidence.

5. Clamping a reversed outlet after the projection reported success with divergence L∞ = 2. Case: west inlet u = −0.25, east outlet with the default clamp. The outlet velocity was set to 0 and the mass imbalance was 0.25. VERIFIED. High confidence.

6. `caseFromJson` checked version, pressure name, density, and viscosity, and then trusted the rest of the object. Restart arrays were copied without a length or finiteness check. CODE-CONFIRMED. High confidence.

## Not defects, or not yet shown

- The momentum flux in `advectMomentum` already takes samples along the face normal. It was not the scalar bug. The uniform-stream test still passes. CODE-CONFIRMED.
- Taylor–Green velocity error from 16² to 32² is still about first order (the existing test requires a ratio above 1.9, which is an order near 1, not 2). This pass did not change the momentum scheme, so that rate is unchanged. VERIFIED by the existing test still passing after the scalar repair.
- VTK `STRUCTURED_POINTS` uses `ORIGIN 0 0 0` while the values are cell averages. The points are therefore half a cell away from the cell centers. CODE-CONFIRMED. Not changed in this pass.
- Force coefficients in the preview are not a validated aerodynamic result. The labels now say so. The integration on a voxel surface remains an approximation. CODE-CONFIRMED.
- Whether the CUDA diffusion kernel races on device is CODE-CONFIRMED by inspection of the source (it reads and writes one velocity array) and UNKNOWN as a sanitizer result. No device run.

## Unsupported

Compressible flow, shocks, turbulence models, adaptive mesh refinement, unstructured meshes, and two-way fluid-structure interaction are not in this solver. The browser does not run CUDA.
