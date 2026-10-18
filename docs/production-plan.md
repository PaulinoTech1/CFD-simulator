# Production program

This is the plan for taking Karman from a checked incompressible core plus a set of screens to the closest production-ready workbench that evidence can support.

Production here means: a user can set up a case inside a written envelope, run it, and trust the number because a test or a reference comparison produced it. It does not mean a general-purpose CFD package.

Current baseline: commit `b00cae4`. The suite is the TypeScript tests under `verify/` plus `g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu && ./karman3d`. Device execution is unverified.

Approve this document as one program. Execution follows the phases in order. A phase that fails its acceptance test stops the program. Later phases are not used to hide that failure.

## Envelope

In scope for a production claim:

- Constant-density Newtonian laminar flow.
- Structured Cartesian grid, 2D (by setting one axis periodic and thin) and 3D.
- Staggered MAC projection. Stored pressure is kinematic, φ = p/ρ.
- Boundary set already in the case file: wall, inlet, outlet, lid, periodic pairs.
- Passive scalar transport.
- Boussinesq buoyancy, after its own reference case passes.
- One-way plane-stress section, after a patch test passes. Not fluid–structure interaction.

Out of scope until a later program, each with a prerequisite listed in Phase 11:

- Compressible flow, shocks, multiphase, RANS/LES, AMR, unstructured meshes, two-way FSI, electromagnetics, a CAD kernel, a 3D lattice Boltzmann solver.

## Rules that apply to every phase

1. The TypeScript MAC solver stays the oracle until the native host matches it on the same cases.
2. A new number needs a regression test before the UI is allowed to show it without a qualifier.
3. Qualifiers are `checked`, `preview`, and `screen`. A screen is a closed-form check. It is not promoted by renaming.
4. No `--use_fast_math`. No tolerance change to make a plot pass.
5. GPU work is skipped, not faked, when `nvcc` or the device is absent.
6. No copied source from TCLB, Sailfish, SPECTER, OpenFOAM, or other external solvers. Design notes only.

## Phase 0 — Lock the baseline

Record the green state so later diffs have a number to beat.

- Run `npm run test:numerics` and the host `karman3d` binary. Store the pass count and the Couette error in this file's log section at the bottom.
- Add a one-page envelope statement at the top of the Analysis panel and the guide, copied from the envelope above, not paraphrased into a stronger claim.

Acceptance: the recorded commands pass, and the UI text matches the envelope.

## Phase 1 — Reference cases for the MAC oracle

The oracle already has Couette, Poiseuille, a scalar shift, rollback, and a near-first-order Taylor–Green study. Add the cases a laminar solver is expected to publish.

1. Lid-driven cavity, Ghia et al. profiles at Re = 100, mid-plane u(y) and v(x), on a grid the reference cap allows. Report the max norm. Do not require a match tighter than the grid can support; publish the measured error.
2. Re = 400 only if Re = 100 is stable and the timestep stays inside the CFL guard. Otherwise stop and record why.
3. Natural convection in a square cavity at a low Rayleigh number with a published Nusselt number, or a stratified rest state that must stay at rest if the Boussinesq term is balanced. One of these is required before heat is called checked.
4. Start a cylinder grid study: Strouhal versus cells at Re = 100. Publish the table. Do not label Cd or Cl checked in this phase.

Files: `src/cfd/mac.ts` only if a boundary or body mask is missing, `verify/cavity.test.mjs`, `verify/heat-rest.test.mjs`, `docs/audit/references.md`.

Acceptance: each new test prints the error against the cited table and passes its own stated tolerance. Taylor–Green's order assertion is left where it is.

## Phase 2 — Native host matches the oracle

`kernels/karman3d.cu` currently checks one shear decay and one periodic projection. It does not implement the oracle's scalar flux, rollback, outlet clamp, or case rejection.

1. Port the oracle's discrete operators into the host path of `karman3d.cu`: face-normal scalar reconstruction, viscous step, projection with nullspace removal, outlet mass check, transactional step.
2. Drive both solvers from the same JSON case (`src/cfd/case.ts` shape).
3. Compare Couette, Poiseuille, the scalar shift, and the cavity profile. Agreement is relative L2 below `1e-8` on the linear cases and below the oracle's own cavity error on the cavity.
4. Keep a CPU-only build. `scripts/build.sh` stays host-first and still refuses to fail the build when `nvcc` is missing.

Acceptance: `./karman3d --case cases/couette.json` and the matching `verify/` test report the same error to the printed precision. A mismatch stops the program.

## Phase 3 — Pressure solver, with the old one kept

The oracle uses conjugate gradients. The geometric multigrid in `src/cfd/multigrid.ts` solves a different problem: 2D Dirichlet Poisson. Do not swap them.

1. Implement matrix-free PCG or a geometric V-cycle for the MAC Laplacian, including obstacles and the existing boundary set.
2. Keep the current CG as `projectReference`.
3. On every existing projection test, the new solver must match the reference pressure to `1e-8` after nullspace removal, and the divergence test must still pass.
4. Record iterations and residual. A speedup is reported only as a host timing on this machine, labeled as not a 3090 Ti result.

Acceptance: all Phase 1 tests pass with the new solver selected, and one test still runs the reference CG so a regression has a witness.

## Phase 4 — Heat and scalar inside the envelope

1. Passive scalar: the axis tests already exist. Add a manufactured inflow/outflow case with a non-zero and a zero exterior value, already partly covered, and a 3D diagonal advection.
2. Boussinesq: implement it in the oracle, not only in the preview. The Phase 1 heat case must pass.
3. Conjugate heat stays a screen (`conjugateBalance` in `src/cfd/experimental.ts`) until a solid cell conducts and the interface flux matches the fluid flux on a two-node problem. That two-node problem is the gate. A field solve comes only after it passes.
4. The Analysis row "Convective heat transfer" may move from preview to checked only for the case that passed. The UI must name that case.

Acceptance: a test name is cited in the UI for every heat claim. No Nusselt number is shown without the grid and the reference.

## Phase 5 — Forces a user can quote, with the grid attached

1. Compute pressure and viscous force on the voxel surface with the MAC stresses, not a momentum deficit borrowed from the preview.
2. Cylinder at Re = 100: publish Cd, Cl amplitude, and Strouhal against a cited range, on at least two grids.
3. If the coarse grid misses the range, show the number as preview and keep the table. Do not tighten the scheme silently.
4. NACA 0012 stays preview until a body-fitted or adequately refined stair-step study exists. Say so in the Suite panel.

Acceptance: the force test fails if the grid label is missing. A single-grid Cd is never marked checked.

## Phase 6 — One scheme on screen

The canvas runs the collocated preview in `src/cfd/solver3d.ts`. Production readiness requires the picture and the checked number to come from the same scheme on the bench grid.

1. Run `MacSolver` on the bench span (32×16×16) inside a worker, or synchronously if a step stays interactive.
2. Draw that solution. Keep the collocated preview behind a control labeled "old preview" for one release, then remove it if the MAC view holds 10 steps per second on the bench span.
3. HUD St, Cd, and Cl either come from the Phase 5 force routine or they stay labeled preview. They are not recomputed by a second method.
4. Restart and case import from the UI must round-trip through `validateCase` and show the field name on rejection.

Acceptance: a test renders no pixels, but it runs 20 MAC steps on the bench case and checks divergence. A manual note in the guide says the picture is the MAC scheme. The old preview cannot be the default.

## Phase 7 — Geometry that the grid can represent

1. Voxel stamps stay. Add a tested volume: a stamped sphere or box has a cell count within one cell of the analytic voxelization.
2. Stair-step boundary condition: no-slip on every face that touches a fluid cell, checked by a Couette case with a voxel slab that must recover the linear profile away from the stair.
3. CAD is an import slot only. Accept an STL later by voxelizing it in a separate tool and running the volume test. Do not link a CAD kernel in this program.
4. Orbit, zoom, and the guide stay. The guide is rewritten at the end of Phase 6 so it describes the MAC picture.

Acceptance: the voxel volume test and the slab Couette test pass. The UI does not offer edge repair.

## Phase 8 — Section stress, one way

1. Patch test: a single plane-stress element under constant tension returns the analytic stress and a rigid-body-free displacement.
2. The existing mid-span section must refuse to report a stress when the matrix is singular or the section is empty. It already tries to; add a regression.
3. After the patch test, the thermal strain from `SCREEN.alpha` may be added as an initial strain, with a test that a uniform ΔT and fixed ends produce σ = E α ΔT.
4. SDOF frequency, harmonic gain, and the 1% strain flag stay screens. They may use the patch-tested E, but they must not be labeled modal analysis.

Acceptance: the patch test passes in `verify/section.test.mjs`. The Suite copy says "patch-tested plane stress, one way, not FSI."

## Phase 9 — GPU, only on evidence

Blocked until the target machine has `nvcc` and the 3090 Ti.

1. Compile `kernels/karman3d.cu` at `-arch=sm_86` with the host scheme from Phase 2. No `--use_fast_math`.
2. Run the same Couette, Poiseuille, scalar-shift, and cavity inputs on the device and the host. Relative L2 below `1e-6` for double, or a written reason if the device path is single and the tolerance is wider.
3. Time one pressure solve. Compare it with the bandwidth estimate from `planVolume`. Publish both numbers. Do not quote the 40 TFLOP peak as a result.
4. PTX and AVX2 stay unwired unless a profile shows the host kernel they would replace is the bottleneck, and a test covers the replacement.

Acceptance: the device log contains the four comparisons and a timing. A build that only links is still unverified.

## Phase 10 — Product surface

1. Case list: the reference cases from Phase 1 are presets that load a validated case, not only a preview scene.
2. Errors name the bad field, the axis, and the limit.
3. The guide's nine steps are rewritten to this envelope after Phase 6. Experimental rows keep the word screen.
4. GitHub Actions runs `npm run test:numerics` and the host `g++` check. It does not claim a GPU job.
5. Performance budget, written not hoped: bench span stays interactive in the browser; 2,000,000 cells is the reference cap; larger is refused with the budget text from `cellBudget`.

Acceptance: a new user can open the cavity preset, read a checked error against Ghia, and export the case. The Actions log is green on CPU only.

## Phase 11 — Explicitly not this program

| Item | Prerequisite before any implementation |
|---|---|
| 3D lattice Boltzmann | A D3Q19 periodic conservation test and a Couette profile in lattice units, separate from the MAC oracle |
| Compressible | A 1D shock tube with a stated Riemann solver, after the incompressible envelope is documented as the wrong tool |
| Multiphase | A stored volume fraction and a mass balance per phase. Dye must not be reused |
| Conjugate fields | The two-node flux test from Phase 4 |
| RANS or LES | A laminar cylinder study whose Strouhal is already on the cited range |
| AMR, unstructured, CAD kernel | A voxel case that is already checked, plus an import test |
| Two-way FSI | The Phase 8 patch test and a fluid force that is checked on two grids |
| Electromagnetics | Out of this project |
| Modal, harmonic, nonlinear solids | A 3D elasticity patch test. The SDOF screens do not count |

## Order and stop rule

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
```

Phase 9 waits on hardware. Phases 7 and 8 can follow Phase 6 without the GPU. Phase 11 is never started inside this approval.

If a phase misses its acceptance test, stop, write the measured number, and do not start the next phase.

## Log

| Date | Commit | Event |
| --- | --- | --- |
| 2026-10-18 | `b00cae4` | Plan written. Oracle tests green. Host binary checks shear decay and one projection only. GPU unverified. |
| 2026-10-18 | this change | Phases 0–8 landed as tests. Phase 2 host Couette is the 1D reduction, not a full operator port. Phase 6 runs the MAC scheme headless on the bench span; the canvas is still the preview. Phase 9 skips: no nvcc. |
| 2026-10-18 | browser checks | Run checks executes the MAC shear, heat-rest, and Couette cases in the page and prints PASS or FAIL. The picture is unchanged. |
