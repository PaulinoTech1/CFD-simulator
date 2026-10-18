import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { poissonVCycle } from "../src/cfd/multigrid.ts";
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
});

describe("experimental lattice step", () => {
  it("keeps a uniform rest state at unit density", () => {
    const r = latticeRun(8, 8, 5, 0.8);
    assert.ok(Math.abs(r.meanDensity - 1) < 1e-9, `density ${r.meanDensity}`);
    assert.ok(r.maxSpeed < 1e-9, `speed ${r.maxSpeed}`);
  });
});
