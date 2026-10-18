// Geometric multigrid for a Cartesian Poisson problem with homogeneous Dirichlet boundaries.
// Written from the standard V-cycle description. No external source was copied.
// Not wired into MacSolver.project. The conjugate gradient there remains the reference.

export type PoissonGrid = {
  nx: number;
  ny: number;
  h: number;
};

export type MgReport = {
  iterations: number;
  residual: number;
  converged: boolean;
};

function idx(i: number, j: number, nx: number): number {
  return i + nx * j;
}

function applyA(g: PoissonGrid, x: Float64Array, y: Float64Array): void {
  const { nx, ny, h } = g;
  const inv = 1 / (h * h);
  y.fill(0);
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const c = idx(i, j, nx);
      y[c] =
        (x[idx(i - 1, j, nx)]! + x[idx(i + 1, j, nx)]! + x[idx(i, j - 1, nx)]! + x[idx(i, j + 1, nx)]! - 4 * x[c]!) * inv;
    }
  }
}

function residualOf(g: PoissonGrid, x: Float64Array, b: Float64Array, r: Float64Array): number {
  applyA(g, x, r);
  let s = 0;
  let n = 0;
  for (let j = 1; j < g.ny - 1; j++) {
    for (let i = 1; i < g.nx - 1; i++) {
      const c = idx(i, j, g.nx);
      r[c] = b[c]! - r[c]!;
      s += r[c]! * r[c]!;
      n += 1;
    }
  }
  return n > 0 ? Math.sqrt(s / n) : 0;
}

function smooth(g: PoissonGrid, x: Float64Array, b: Float64Array, sweeps: number): void {
  const { nx, ny, h } = g;
  const inv = 1 / (h * h);
  const diag = -4 * inv;
  for (let s = 0; s < sweeps; s++) {
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const c = idx(i, j, nx);
        const off =
          (x[idx(i - 1, j, nx)]! + x[idx(i + 1, j, nx)]! + x[idx(i, j - 1, nx)]! + x[idx(i, j + 1, nx)]!) * inv;
        x[c] = (b[c]! - off) / diag;
      }
    }
  }
}

function restrict(fine: PoissonGrid, r: Float64Array): { grid: PoissonGrid; rhs: Float64Array } {
  const nx = Math.max(3, Math.floor(fine.nx / 2) + 1);
  const ny = Math.max(3, Math.floor(fine.ny / 2) + 1);
  const rhs = new Float64Array(nx * ny);
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const i0 = Math.min(fine.nx - 2, Math.max(1, i * 2));
      const j0 = Math.min(fine.ny - 2, Math.max(1, j * 2));
      rhs[idx(i, j, nx)] = r[idx(i0, j0, fine.nx)]!;
    }
  }
  return { grid: { nx, ny, h: fine.h * 2 }, rhs };
}

function prolong(coarse: PoissonGrid, e: Float64Array, fine: PoissonGrid, x: Float64Array): void {
  for (let j = 1; j < fine.ny - 1; j++) {
    for (let i = 1; i < fine.nx - 1; i++) {
      const ic = Math.min(coarse.nx - 2, Math.max(1, Math.floor(i / 2)));
      const jc = Math.min(coarse.ny - 2, Math.max(1, Math.floor(j / 2)));
      x[idx(i, j, fine.nx)] = x[idx(i, j, fine.nx)]! + e[idx(ic, jc, coarse.nx)]!;
    }
  }
}

function vcycle(g: PoissonGrid, x: Float64Array, b: Float64Array, depth: number): void {
  smooth(g, x, b, 3);
  if (depth === 0 || g.nx <= 5 || g.ny <= 5) {
    smooth(g, x, b, 30);
    return;
  }
  const r = new Float64Array(x.length);
  residualOf(g, x, b, r);
  const coarse = restrict(g, r);
  const e = new Float64Array(coarse.rhs.length);
  vcycle(coarse.grid, e, coarse.rhs, depth - 1);
  prolong(coarse.grid, e, g, x);
  smooth(g, x, b, 3);
}

/** Solve ∇²φ = b with homogeneous Dirichlet boundaries, by geometric V-cycles. */
export function poissonVCycle(g: PoissonGrid, b: Float64Array, x: Float64Array, tol = 1e-6, maxCycles = 40): MgReport {
  let residualNorm = residualOf(g, x, b, new Float64Array(x.length));
  let iterations = 0;
  while (residualNorm > tol && iterations < maxCycles) {
    vcycle(g, x, b, 3);
    residualNorm = residualOf(g, x, b, new Float64Array(x.length));
    iterations += 1;
  }
  return { iterations, residual: residualNorm, converged: residualNorm <= tol };
}
