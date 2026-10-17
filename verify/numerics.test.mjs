import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { emptyBoundaries, inlet, lid, makeCase, outlet, periodic, wall } from "../src/cfd/case.ts";
import { MacSolver } from "../src/cfd/mac.ts";

const preview = readFileSync(new URL("../src/cfd/solver3d.ts", import.meta.url), "utf8");

function periodicBox(n, nu, advection = "none") {
  const b = emptyBoundaries();
  b.west = periodic();
  b.east = periodic();
  b.south = periodic();
  b.north = periodic();
  b.back = periodic();
  b.front = periodic();
  return makeCase({
    name: "periodic",
    nx: n,
    ny: n,
    nz: 1,
    lx: 1,
    ly: 1,
    lz: 1,
    nu,
    boundaries: b,
    advection,
    uRef: 1,
    lRef: 1,
  });
}

describe("baseline defects still present in the collocated preview", () => {
  it("keeps the known preview shortcuts, which the MAC reference does not use", () => {
    assert.match(preview, /Math\.min\(dt \* nu, 0\.12\)/);
    assert.match(preview, /if \(mag > 0\.85\)/);
    assert.match(preview, /u\[id\] = 0;\s*v\[id\] = 0;\s*w\[id\] = 0;/);
    assert.match(preview, /const curl = Math\.hypot\(dudy - 0,/);
  });
});

describe("compatible MAC projection", () => {
  it("div(grad) matches an independent non-cubic stencil, including an outlet", () => {
    const b = emptyBoundaries();
    b.east = outlet(0.25);
    const spec = makeCase({
      name: "operator",
      nx: 5,
      ny: 4,
      nz: 3,
      lx: 0.5,
      ly: 0.8,
      lz: 1.2,
      nu: 0,
      boundaries: b,
    });
    const sim = new MacSolver(spec);
    for (let c = 0; c < sim.phi.length; c++) sim.phi[c] = Math.sin(0.7 * c) + 0.15 * c;
    sim.writeGradient(1, sim.u, sim.v, sim.w);
    const dx = spec.lx / spec.nx;
    const dy = spec.ly / spec.ny;
    const dz = spec.lz / spec.nz;
    let err = 0;
    for (let k = 0; k < spec.nz; k++) {
      for (let j = 0; j < spec.ny; j++) {
        for (let i = 0; i < spec.nx; i++) {
          const id = sim.ic(i, j, k);
          const c = sim.phi[id];
          const xm = i === 0 ? c : sim.phi[sim.ic(i - 1, j, k)];
          const xp = i === spec.nx - 1 ? spec.boundaries.east.phi : sim.phi[sim.ic(i + 1, j, k)];
          const ym = j === 0 ? c : sim.phi[sim.ic(i, j - 1, k)];
          const yp = j === spec.ny - 1 ? c : sim.phi[sim.ic(i, j + 1, k)];
          const zm = k === 0 ? c : sim.phi[sim.ic(i, j, k - 1)];
          const zp = k === spec.nz - 1 ? c : sim.phi[sim.ic(i, j, k + 1)];
          const lap =
            (xm - 2 * c + xp) / (dx * dx) + (ym - 2 * c + yp) / (dy * dy) + (zm - 2 * c + zp) / (dz * dz);
          const div =
            (sim.u[sim.iu(i + 1, j, k)] - sim.u[sim.iu(i, j, k)]) / dx +
            (sim.v[sim.iv(i, j + 1, k)] - sim.v[sim.iv(i, j, k)]) / dy +
            (sim.w[sim.iw(i, j, k + 1)] - sim.w[sim.iw(i, j, k)]) / dz;
          err = Math.max(err, Math.abs(div - lap));
        }
      }
    }
    assert.ok(err < 1e-12, `operator mismatch ${err}`);
  });

  it("removes a manufactured divergence and reports the algebraic residual", () => {
    const spec = makeCase({
      name: "project",
      nx: 12,
      ny: 10,
      nz: 8,
      lx: 1.2,
      ly: 1,
      lz: 0.8,
      nu: 0,
      boundaries: emptyBoundaries(),
      poisson: { atol: 1e-10, rtol: 1e-8, maxIterations: 2000 },
    });
    const sim = new MacSolver(spec);
    for (let k = 0; k < spec.nz; k++) {
      for (let j = 0; j < spec.ny; j++) {
        for (let i = 0; i <= spec.nx; i++) sim.u[sim.iu(i, j, k)] = Math.sin((2 * i) / spec.nx) + 0.2 * j;
      }
    }
    for (let k = 0; k < spec.nz; k++) {
      for (let j = 0; j <= spec.ny; j++) {
        for (let i = 0; i < spec.nx; i++) sim.v[sim.iv(i, j, k)] = Math.cos((3 * j) / spec.ny);
      }
    }
    const before = sim.divergenceNorms().linf;
    const report = sim.step(1e-3);
    assert.equal(report.status, "ok");
    assert.ok(report.pressureConverged);
    assert.ok(before > 0.1, `expected a divergent field, got ${before}`);
    // CG stops at atol + rtol·‖b‖. Divergence after the step is that residual times dt.
    assert.ok(report.divLinf < 1e-6, `div Linf ${report.divLinf}`);
    assert.ok(report.divL2 < 1e-6, `div L2 ${report.divL2}`);
  });

  it("refuses a nonconverged pressure solve without advancing time", () => {
    const spec = makeCase({
      name: "fail",
      nx: 8,
      ny: 8,
      nz: 8,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: emptyBoundaries(),
      poisson: { atol: 1e-14, rtol: 1e-14, maxIterations: 1 },
    });
    const sim = new MacSolver(spec);
    sim.u[sim.iu(3, 3, 3)] = 4;
    const report = sim.step(0.01);
    assert.equal(report.status, "pressure-not-converged");
    assert.equal(sim.time, 0);
    assert.equal(sim.u[sim.iu(3, 3, 3)], 4);
  });

  it("does not replace a nonfinite velocity with zero", () => {
    const sim = new MacSolver(
      makeCase({
        name: "nan",
        nx: 4,
        ny: 4,
        nz: 4,
        lx: 1,
        ly: 1,
        lz: 1,
        nu: 0.01,
        boundaries: emptyBoundaries(),
      }),
    );
    sim.u[sim.iu(1, 1, 1)] = Number.NaN;
    const report = sim.step(0.01);
    assert.equal(report.status, "nonfinite");
    assert.equal(sim.time, 0);
    assert.ok(Number.isNaN(sim.u[sim.iu(1, 1, 1)]));
  });

  it("sends mass out of a north outlet, not an east face", () => {
    const b = emptyBoundaries();
    b.south = inlet(0, 0.4, 0);
    b.north = outlet(0);
    b.north.backflow = "allow";
    const spec = makeCase({
      name: "north-outlet",
      nx: 6,
      ny: 8,
      nz: 4,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: b,
    });
    const sim = new MacSolver(spec);
    for (let k = 0; k < spec.nz; k++) {
      for (let j = 0; j <= spec.ny; j++) {
        for (let i = 0; i < spec.nx; i++) sim.v[sim.iv(i, j, k)] = 0.4;
      }
    }
    const report = sim.step(0.01);
    assert.equal(report.status, "ok");
    assert.ok(Math.abs(report.fluxes.east) < 1e-8, `east flux ${report.fluxes.east}`);
    assert.ok(report.fluxes.north > 0.2, `north flux ${report.fluxes.north}`);
    assert.ok(report.fluxes.south < -0.2, `south flux ${report.fluxes.south}`);
    assert.ok(Math.abs(report.massImbalance) < 1e-6, `imbalance ${report.massImbalance}`);
  });
});

describe("viscosity", () => {
  it("matches one discrete viscous step of a shear mode", () => {
    const n = 24;
    const nu = 0.02;
    const sim = new MacSolver(periodicBox(n, nu));
    for (let j = 0; j < n; j++) {
      const value = Math.sin((2 * Math.PI * (j + 0.5)) / n);
      for (let i = 0; i <= n; i++) sim.u[sim.iu(i, j, 0)] = value;
    }
    const kappa = (2 * (Math.cos((2 * Math.PI) / n) - 1)) / sim.dy ** 2;
    const dt = 0.2 * sim.stableDt();
    const report = sim.step(dt);
    assert.equal(report.status, "ok");
    assert.equal(report.dt, dt);
    let err = 0;
    for (let j = 0; j < n; j++) {
      const expected = Math.sin((2 * Math.PI * (j + 0.5)) / n) * (1 + dt * nu * kappa);
      err = Math.max(err, Math.abs(sim.u[sim.iu(0, j, 0)] - expected));
    }
    assert.ok(err < 1e-12, `discrete decay error ${err}`);
    assert.ok(report.divLinf < 1e-8);
  });

  it("approaches Couette's linear profile without changing nu", () => {
    const ny = 24;
    const nu = 0.08;
    const speed = 1;
    const b = emptyBoundaries();
    b.west = periodic();
    b.east = periodic();
    b.back = periodic();
    b.front = periodic();
    b.south = wall(0, 0, 0);
    b.north = lid(speed, 0, 0);
    const spec = makeCase({
      name: "couette",
      nx: 4,
      ny,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu,
      boundaries: b,
      uRef: speed,
      lRef: 1,
    });
    const sim = new MacSolver(spec);
    let guard = 0;
    while (sim.time < 12 && guard < 60000) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
      guard += 1;
    }
    let err = 0;
    for (let j = 0; j < ny; j++) err = Math.max(err, Math.abs(sim.u[sim.iu(0, j, 0)] - speed * (j + 0.5) * sim.dy));
    assert.ok(err < 1e-3, `Couette Linf ${err}`);
    const shear = sim.wallShear();
    const expectedForce = spec.rho * nu * (speed / spec.ly) * spec.lx * spec.lz;
    assert.ok(Math.abs(shear.south - expectedForce) / expectedForce < 0.02, `south shear ${shear.south}`);
  });

  it("approaches the plane Poiseuille parabola under a constant force", () => {
    const ny = 32;
    const nu = 0.05;
    const g = 0.2;
    const b = emptyBoundaries();
    b.west = periodic();
    b.east = periodic();
    b.back = periodic();
    b.front = periodic();
    const spec = makeCase({
      name: "poiseuille",
      nx: 4,
      ny,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu,
      force: { x: g, y: 0, z: 0 },
      boundaries: b,
    });
    const sim = new MacSolver(spec);
    let guard = 0;
    while (sim.time < 8 && guard < 50000) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
      guard += 1;
    }
    let err = 0;
    let norm = 0;
    for (let j = 0; j < ny; j++) {
      const y = (j + 0.5) * sim.dy;
      const expected = (g / (2 * nu)) * y * (1 - y);
      const got = sim.u[sim.iu(1, j, 0)];
      err += (got - expected) ** 2;
      norm += expected ** 2;
    }
    const rel = Math.sqrt(err / norm);
    assert.ok(rel < 0.03, `Poiseuille relative L2 ${rel}`);
  });
});

describe("transport and scalars", () => {
  it("conserves dye mass for a periodic upwind step", () => {
    const sim = new MacSolver(periodicBox(16, 0, "upwind1"));
    for (let i = 0; i <= 16; i++) for (let j = 0; j < 16; j++) sim.u[sim.iu(i, j, 0)] = 0.3;
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 16; j++) sim.dye[sim.ic(i, j, 0)] = i > 3 && i < 8 && j > 4 && j < 11 ? 1 : 0;
    }
    const mass = () => sim.dye.reduce((s, v) => s + v, 0);
    const before = mass();
    const report = sim.step(0.05);
    assert.equal(report.status, "ok");
    assert.ok(Math.abs(mass() - before) < 1e-10, `mass ${mass()} vs ${before}`);
    assert.ok(report.divLinf < 1e-8);
  });

  it("keeps a uniform stream uniform under MUSCL-RK2", () => {
    const sim = new MacSolver(periodicBox(10, 0, "muscl-rk2"));
    for (let i = 0; i <= 10; i++) for (let j = 0; j < 10; j++) sim.u[sim.iu(i, j, 0)] = 0.7;
    const report = sim.step(0.02);
    assert.equal(report.status, "ok");
    let err = 0;
    for (let j = 0; j < 10; j++) err = Math.max(err, Math.abs(sim.u[sim.iu(3, j, 0)] - 0.7));
    assert.ok(err < 1e-10, `uniform stream drifted by ${err}`);
  });
});

describe("Taylor–Green", () => {
  it("tracks the periodic vortex; observed spatial rate on 16→32 is about 1.1", () => {
    const coarse = taylorGreen(16, 0.5, 0.2);
    const fine = taylorGreen(32, 0.5, 0.2);
    assert.ok(coarse < 0.04, `n=16 ${coarse}`);
    assert.ok(fine < 0.02, `n=32 ${fine}`);
    assert.ok(coarse / fine > 1.9, `rate ${Math.log2(coarse / fine)}`);
  });
});

describe("case exchange", () => {
  it("round-trips the case document and a restart", () => {
    const sim = new MacSolver(periodicBox(6, 0.01));
    sim.u[sim.iu(1, 2, 0)] = 0.4;
    sim.step(1e-4);
    const copy = MacSolver.fromRestart(sim.toRestart());
    assert.equal(copy.time, sim.time);
    assert.equal(copy.u[copy.iu(1, 2, 0)], sim.u[sim.iu(1, 2, 0)]);
    assert.equal(MacSolver.fromCaseJson(sim.toCaseJson()).spec.pressure, "kinematic");
    assert.ok(sim.toVtk().includes("kinematic_pressure"));
  });
});

function taylorGreen(n, tEnd, cfl) {
  const spec = periodicBox(n, 0.01, "muscl-rk2");
  spec.lx = 2 * Math.PI;
  spec.ly = 2 * Math.PI;
  spec.poisson = { atol: 1e-8, rtol: 1e-6, maxIterations: 800 };
  spec.cfl = cfl;
  const sim = new MacSolver(spec);
  const decay = Math.exp(-2 * 0.01 * tEnd);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i <= n; i++) sim.u[sim.iu(i, j, 0)] = Math.sin(i * sim.dx) * Math.cos((j + 0.5) * sim.dy);
  }
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i < n; i++) sim.v[sim.iv(i, j, 0)] = -Math.cos((i + 0.5) * sim.dx) * Math.sin(j * sim.dy);
  }
  let guard = 0;
  while (sim.time < tEnd - 1e-12 && guard < 40000) {
    const report = sim.step(Math.min(sim.stableDt(), tEnd - sim.time));
    assert.equal(report.status, "ok");
    guard += 1;
  }
  let e2 = 0;
  let nrm = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const expected = Math.sin(i * sim.dx) * Math.cos((j + 0.5) * sim.dy) * decay;
      const du = sim.u[sim.iu(i, j, 0)] - expected;
      e2 += du * du;
      nrm += expected * expected;
    }
  }
  return Math.sqrt(e2 / nrm);
}
