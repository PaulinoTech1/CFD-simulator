# Phase 1 report

Repairs are in the MAC reference only (`src/cfd/mac.ts`, `src/cfd/case.ts`). The native CUDA file was not changed and is not at parity with this reference.

## Commands

```text
node --experimental-strip-types --test verify/*.test.mjs
g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o /tmp/karman3d kernels/karman3d.cu && /tmp/karman3d
```

Environment: Debian 12, Node 22.23.3, g++ 12.2.0, no `nvcc`, no GPU.

## Results

JavaScript: 41 tests, 41 passed after the case-validation assertion was pointed at an actually unpaired periodic face. Wall time about 6 s, of which the smooth-wave study is about 4 s.

Native host path, recompiled from the unchanged `kernels/karman3d.cu`:

```text
decay  err 1.110e-16  PASS
project  div 2.487e-14  iters 4  residual 1.031e-10  PASS
gpu  nvcc=0  device-check=not-run  UNVERIFIED
OK
```

Device execution: not run. `nvcc` and `nvidia-smi` are absent.

## What changed

| Item | Behavior |
| --- | --- |
| `scalarFlux` | Donor and reconstructed states are taken along the face normal. First-order upwind remains the `upwind1` scheme. `muscl-rk2` is limited linear reconstruction (minmod) and Heun. Periodic faces use the same wrapped cells on both copies of the face. |
| Boundary scalar | The upwind state is the exterior value on inflow and the interior cell on outflow. A stored 0 is kept. |
| `MacSolver.step` | The call is transactional. A failed substep restores velocity, φ, temperature, dye, time, and the step counter. `attempted`, `accepted`, and `substeps` are reported. `substeps` is 0 when the call rolls back. |
| Outlet clamp | A reversed outlet face is set to 0 and held fixed for a second projection. If the held faces leave a net volume flux above `1e-6` times the domain volume, the step is refused. A closed inlet is refused the same way. Algebraic convergence alone is not success. |
| Imports | `validateCase` checks dimensions, units, CFL, Poisson settings, periodic pairs, and boundary kinds before allocation. Restarts need version 1, a finite clock, an integer step count, and finite arrays of the exact length. The cell cap is 2,000,000. |

`failAtSubstep` is a test hook. It is not a boundary condition.

## Scalar convergence

Periodic translation, uniform velocity, Courant number 1, transversely varying scalar. Relative L2 of first-order upwind against the exact one-cell shift:

| Axis | Sign | n = 16, 32, 64 |
| --- | --- | --- |
| x, y, z | +1 and −1 | below `1e-8`, mass error below `1e-8` |

Before the repair the same probe at n = 16 was 0.06 to 0.33, depending on axis and sign. See `docs/audit/current-state.md`.

Smooth sine, CFL 0.3, t = 0.5, passive directions use 4 cells. x, y, and z agreed to better than 0.1%. Relative L2:

| Scheme | n = 16 | n = 32 | n = 64 | Order 16→32 | Order 32→64 |
| --- | --- | --- | --- | --- | --- |
| upwind1 | 0.353 | 0.195 | 0.102 | 0.86 | 0.93 |
| muscl-rk2 | 0.150 | 0.0615 | 0.0201 | 1.29 | 1.61 |

The upwind rate is climbing toward 1 and is not yet flat, so this is not claimed as a pure asymptotic study. The limiter is minmod. The MUSCL rate is above first order and is not 2. Mass at n = 64 is at roundoff (about `1e-12` or smaller). Extrema of the upwind shift test match the initial field because the shift is exact.

## Projection checks

- Closed inlet, u = 0.5: the step is `rejected`, time stays 0. Previously this returned `ok` with divergence 0.5.
- Suction inlet u = −0.25 and a clamped outlet: `rejected`, accepted interval 0, west velocity remains −0.25. Previously divergence was 2 and the status was `ok`.
- North inlet and a north outlet with backflow allowed: still `ok`, divergence below `1e-6`.
- Existing Couette, Poiseuille, shear-mode, and Taylor–Green tests still pass. Taylor–Green was not retuned.

## Still not true

- This is not an engineering-validated CFD package.
- The picture in the browser is still the collocated CPU preview. Cd, Cl, and St on that picture are labeled preview.
- `kernels/karman3d.cu` does not yet implement this repaired scheme.
- No GPU timing exists.
- Lid-driven cavity data and cylinder shedding are not in this report.
- Voxel surface forces and the plane-stress panel are not validated loads.
