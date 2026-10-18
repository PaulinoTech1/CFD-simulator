// Minimal D2Q9 collide-and-stream step.
// Written from the standard lattice description. No external source was copied.
// Used only by the experimental LBM reading. Not a 3D solver.

const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];

export type LatticeReport = {
  steps: number;
  maxSpeed: number;
  meanUx: number;
  meanDensity: number;
};

function eq(rho: number, ux: number, uy: number, i: number): number {
  const cu = CX[i]! * ux + CY[i]! * uy;
  const uu = ux * ux + uy * uy;
  return W[i]! * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uu);
}

/** One collide-stream step on a periodic nx by ny lattice. Returns the updated distributions. */
export function latticeStep(nx: number, ny: number, f: Float64Array, tau: number): Float64Array<ArrayBuffer> {
  const next = new Float64Array(f.length) as Float64Array<ArrayBuffer>;
  const omega = 1 / tau;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const id = i + nx * j;
      let rho = 0;
      let ux = 0;
      let uy = 0;
      for (let q = 0; q < 9; q++) {
        const fq = f[id * 9 + q]!;
        rho += fq;
        ux += CX[q]! * fq;
        uy += CY[q]! * fq;
      }
      ux = rho > 0 ? ux / rho : 0;
      uy = rho > 0 ? uy / rho : 0;
      for (let q = 0; q < 9; q++) {
        const feq = eq(rho, ux, uy, q);
        const collided = f[id * 9 + q]! - omega * (f[id * 9 + q]! - feq);
        const ni = (i + CX[q]! + nx) % nx;
        const nj = (j + CY[q]! + ny) % ny;
        next[(ni + nx * nj) * 9 + q] = collided;
      }
    }
  }
  return next;
}

/** Periodic D2Q9 from equilibrium at density 1 and velocity (ux0, 0). Mass and momentum should stay put. */
export function latticeRun(nx: number, ny: number, steps: number, tau: number, ux0 = 0): LatticeReport {
  const f = new Float64Array(nx * ny * 9);
  for (let c = 0; c < nx * ny; c++) {
    for (let q = 0; q < 9; q++) f[c * 9 + q] = eq(1, ux0, 0, q);
  }
  let cur = f;
  for (let s = 0; s < steps; s++) cur = latticeStep(nx, ny, cur, tau);
  let rhoSum = 0;
  let uxSum = 0;
  let maxSpeed = 0;
  for (let c = 0; c < nx * ny; c++) {
    let rho = 0;
    let ux = 0;
    let uy = 0;
    for (let q = 0; q < 9; q++) {
      const fq = cur[c * 9 + q]!;
      rho += fq;
      ux += CX[q]! * fq;
      uy += CY[q]! * fq;
    }
    rhoSum += rho;
    const vx = rho > 0 ? ux / rho : 0;
    const vy = rho > 0 ? uy / rho : 0;
    uxSum += vx;
    const sp = Math.hypot(vx, vy);
    if (sp > maxSpeed) maxSpeed = sp;
  }
  const cells = nx * ny;
  return { steps, maxSpeed, meanUx: uxSum / cells, meanDensity: rhoSum / cells };
}
