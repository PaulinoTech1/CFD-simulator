import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { applyPoisson, poissonVCycle } from "../src/cfd/multigrid.ts";
import { latticeRun } from "../src/cfd/lattice.ts";

describe("geometric multigrid", () => {
  it("drives a smooth Poisson residual down", () => {
    const nx = 16;
    const ny = 16;
    const h = 1 / nx;
    const b = new Float64Array(nx * ny);
    const x = new Float64Array(nx * ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        b[i + nx * j] = Math.sin((2 * Math.PI * i) / nx) * Math.sin((2 * Math.PI * j) / ny);
      }
    }
    const report = poissonVCycle({ nx, ny, h }, b, x, 1e-6, 30);
    assert.equal(report.converged, true);
    assert.ok(report.residual < 1e-6, `residual ${report.residual}`);
    assert.ok(report.iterations > 0);
  });

  it("recovers a field whose Laplacian is the right-hand side", () => {
    const nx = 17;
    const ny = 17;
    const h = 1 / 16;
    const g = { nx, ny, h };
    const exact = new Float64Array(nx * ny);
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        exact[i + nx * j] = Math.sin((Math.PI * i) / (nx - 1)) * Math.sin((Math.PI * j) / (ny - 1));
      }
    }
    const b = new Float64Array(nx * ny);
    applyPoisson(g, exact, b);
    const x = new Float64Array(nx * ny);
    const report = poissonVCycle(g, b, x, 1e-8, 40);
    assert.equal(report.converged, true);
    let err = 0;
    let n = 0;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const d = x[i + nx * j] - exact[i + nx * j];
        err += d * d;
        n += 1;
      }
    }
    const rms = Math.sqrt(err / n);
    assert.ok(rms < 1e-4, `rms ${rms}`);
  });
});

describe("experimental lattice step", () => {
  it("keeps a uniform velocity on a periodic lattice", () => {
    const r = latticeRun(8, 8, 10, 0.8, 0.05);
    assert.ok(Math.abs(r.meanDensity - 1) < 1e-9, `density ${r.meanDensity}`);
    assert.ok(Math.abs(r.meanUx - 0.05) < 1e-9, `ux ${r.meanUx}`);
  });
});
