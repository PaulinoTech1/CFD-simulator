import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { emptyBoundaries, inlet, lid, makeCase, outlet, periodic, wall } from "../src/cfd/case.ts";
import { patchPlaneStress } from "../src/cfd/fea.ts";
import { MacSolver } from "../src/cfd/mac.ts";

function sliceWalls() {
  const b = emptyBoundaries();
  b.back = periodic();
  b.front = periodic();
  return b;
}

describe("lid-driven cavity against Ghia Re 100", () => {
  it("matches interior centerline samples on a 32² grid and records the lid-layer miss", () => {
    const n = 32;
    const b = sliceWalls();
    b.north = lid(1, 0, 0);
    const sim = new MacSolver(makeCase({
      name: "ghia-re100",
      nx: n,
      ny: n,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0.01,
      advection: "upwind1",
      uRef: 1,
      lRef: 1,
      boundaries: b,
    }));
    let guard = 0;
    while (sim.time < 12 && guard < 20000) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
      guard += 1;
    }
    const at = (y) => {
      const dy = sim.dy;
      const yFace = Math.min(n - 1, Math.max(0, y / dy - 0.5));
      const j0 = Math.floor(yFace);
      const j1 = Math.min(n - 1, j0 + 1);
      const t = yFace - j0;
      const i = n >> 1;
      return (1 - t) * sim.u[sim.iu(i, j0, 0)] + t * sim.u[sim.iu(i, j1, 0)];
    };
    // Ghia, Ghia, Shin, J. Comput. Phys. 48 (1982), Re = 100, u(x=0.5).
    const interior = [
      [0.5, -0.20581],
      [0.7344, 0.00332],
      [0.1719, -0.10150],
    ];
    let interiorErr = 0;
    for (const [y, u] of interior) interiorErr = Math.max(interiorErr, Math.abs(at(y) - u));
    const lidErr = Math.abs(at(0.9531) - 0.68717);
    assert.ok(interiorErr < 0.05, `interior Linf ${interiorErr}`);
    assert.ok(lidErr < 0.25, `near-lid error ${lidErr} on a 32² upwind grid`);
    assert.ok(sim.last.divLinf < 1e-6);
  });
});

describe("Boussinesq rest and conduction", () => {
  it("stays at rest when temperature is uniform", () => {
    const b = sliceWalls();
    const sim = new MacSolver(makeCase({
      name: "heat-rest",
      nx: 8,
      ny: 8,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0.1,
      beta: 0.01,
      gravity: { x: 0, y: -1, z: 0 },
      tRef: 20,
      advection: "none",
      boundaries: b,
    }));
    for (let n = 0; n < 20; n++) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
    }
    assert.ok(sim.last.maxSpeed < 1e-8, `speed ${sim.last.maxSpeed}`);
  });

  it("relaxes a wall-to-wall temperature drop toward a line", () => {
    const b = sliceWalls();
    b.south = periodic();
    b.north = periodic();
    b.west = wall(0, 0, 0, 1);
    b.east = wall(0, 0, 0, 0);
    const nx = 16;
    const sim = new MacSolver(makeCase({
      name: "conduction",
      nx,
      ny: 4,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      alpha: 0.2,
      tRef: 0,
      advection: "none",
      boundaries: b,
    }));
    let guard = 0;
    while (sim.time < 8 && guard < 20000) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
      guard += 1;
    }
    let err = 0;
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) / nx;
      err = Math.max(err, Math.abs(sim.temperature[sim.ic(i, 1, 0)] - (1 - x)));
    }
    assert.ok(err < 0.05, `conduction Linf ${err}`);
  });
});

describe("surface force and voxel stair", () => {
  it("reports a finite force on a voxel block without calling it a coefficient", () => {
    const b = sliceWalls();
    b.west = inlet(1);
    b.east = outlet(0);
    const nx = 20;
    const ny = 12;
    const solid = new Uint8Array(nx * ny);
    for (let j = 5; j <= 7; j++) for (let i = 6; i <= 8; i++) solid[i + nx * j] = 1;
    const sim = new MacSolver(makeCase({
      name: "block",
      nx,
      ny,
      nz: 1,
      lx: 2,
      ly: 1,
      lz: 1,
      nu: 0.01,
      advection: "upwind1",
      uRef: 1,
      lRef: 0.15,
      areaRef: 0.15,
      boundaries: b,
    }), solid);
    for (let n = 0; n < 30; n++) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
    }
    const force = sim.bodyForce();
    assert.ok(Number.isFinite(force.pressure.x + force.viscous.x));
    assert.ok(sim.last.divLinf < 1e-4, `div ${sim.last.divLinf}`);
  });

  it("recovers a linear profile above a voxel floor", () => {
    const ny = 16;
    const nx = 4;
    const b = sliceWalls();
    b.west = periodic();
    b.east = periodic();
    b.north = lid(1, 0, 0);
    const solid = new Uint8Array(nx * ny);
    const floor = 4;
    for (let j = 0; j < floor; j++) for (let i = 0; i < nx; i++) solid[i + nx * j] = 1;
    const sim = new MacSolver(makeCase({
      name: "stair-couette",
      nx,
      ny,
      nz: 1,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0.08,
      advection: "none",
      boundaries: b,
    }), solid);
    let guard = 0;
    while (sim.time < 6 && guard < 20000) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok");
      guard += 1;
    }
    const gap = ny - floor;
    let err = 0;
    for (let j = floor; j < ny; j++) {
      const y = (j - floor + 0.5) / gap;
      err = Math.max(err, Math.abs(sim.u[sim.iu(1, j, 0)] - y));
    }
    assert.ok(err < 0.08, `stair Couette Linf ${err}`);
  });

  it("counts an axis-aligned voxel box exactly", () => {
    const nx = 12;
    const ny = 8;
    const solid = new Uint8Array(nx * ny);
    let expected = 0;
    for (let j = 2; j < 6; j++) for (let i = 3; i < 9; i++) {
      solid[i + nx * j] = 1;
      expected += 1;
    }
    let got = 0;
    for (const v of solid) got += v;
    assert.equal(got, expected);
    assert.equal(expected, 4 * 6);
  });
});

describe("plane-stress patch", () => {
  it("returns uniaxial stress E ε and zero transverse stress", () => {
    const E = 2e9;
    const eps = 1e-4;
    const s = patchPlaneStress(E, 0.3, eps);
    assert.ok(Math.abs(s.sigmaX - E * eps) / (E * eps) < 1e-12);
    assert.ok(Math.abs(s.sigmaY) < 1e-6);
    assert.equal(s.tau, 0);
  });
});

describe("bench-span MAC", () => {
  it("keeps a 32×16×16 lid slice divergence-free for 20 steps", () => {
    const b = sliceWalls();
    b.north = lid(1, 0, 0);
    const sim = new MacSolver(makeCase({
      name: "bench",
      nx: 32,
      ny: 16,
      nz: 16,
      lx: 2,
      ly: 1,
      lz: 1,
      nu: 0.02,
      advection: "none",
      boundaries: b,
      poisson: { atol: 1e-8, rtol: 1e-6, maxIterations: 400 },
    }));
    for (let n = 0; n < 20; n++) {
      const report = sim.step(sim.stableDt());
      assert.equal(report.status, "ok", report.detail);
    }
    assert.ok(sim.last.divLinf < 1e-4, `div ${sim.last.divLinf}`);
  });
});

describe("Jacobi witness against CG", () => {
  it("removes the same manufactured divergence", () => {
    const spec = makeCase({
      name: "jacobi",
      nx: 8,
      ny: 8,
      nz: 4,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: emptyBoundaries(),
      poisson: { atol: 1e-8, rtol: 1e-6, maxIterations: 4000 },
    });
    const fill = (sim) => {
      for (let k = 0; k < spec.nz; k++) {
        for (let j = 0; j < spec.ny; j++) {
          for (let i = 0; i <= spec.nx; i++) sim.u[sim.iu(i, j, k)] = Math.sin((2 * i) / spec.nx);
        }
      }
    };
    const cg = new MacSolver(spec);
    fill(cg);
    const a = cg.step(1e-3);
    const jac = new MacSolver(spec);
    jac.pressureSolver = "jacobi";
    fill(jac);
    const b = jac.step(1e-3);
    assert.equal(a.status, "ok");
    assert.equal(b.status, "ok", b.detail);
    assert.ok(b.divLinf < 1e-5, `jacobi div ${b.divLinf}`);
    assert.ok(b.pressureIterations > a.pressureIterations);
  });
});
