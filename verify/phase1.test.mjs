import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { CaseError, caseFromJson, emptyBoundaries, inlet, makeCase, outlet, periodic, wall } from "../src/cfd/case.ts";
import { MacSolver } from "../src/cfd/mac.ts";

function periodicAll() {
  const b = emptyBoundaries();
  b.west = periodic();
  b.east = periodic();
  b.south = periodic();
  b.north = periodic();
  b.back = periodic();
  b.front = periodic();
  return b;
}

function shiftError(axis, n, sign) {
  const nx = axis === "x" ? n : 8;
  const ny = axis === "y" ? n : 8;
  const nz = axis === "z" ? n : 4;
  const spec = makeCase({
    name: `shift-${axis}`,
    nx,
    ny,
    nz,
    lx: nx,
    ly: ny,
    lz: nz,
    nu: 0,
    boundaries: periodicAll(),
    advection: "upwind1",
  });
  spec.cfl = 1;
  const sim = new MacSolver(spec);
  const speed = sign;
  if (axis === "x") {
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) sim.u[sim.iu(i, j, k)] = speed;
  } else if (axis === "y") {
    for (let k = 0; k < nz; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) sim.v[sim.iv(i, j, k)] = speed;
  } else {
    for (let k = 0; k <= nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) sim.w[sim.iw(i, j, k)] = speed;
  }
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const s = axis === "x" ? i : axis === "y" ? j : k;
        const transverse = axis === "z" ? i : j;
        const nActive = axis === "x" ? nx : axis === "y" ? ny : nz;
        const nTrans = axis === "z" ? nx : ny;
        sim.dye[sim.ic(i, j, k)] = Math.sin((2 * Math.PI * s) / nActive) * (1 + 0.3 * Math.sin((2 * Math.PI * transverse) / nTrans));
      }
    }
  }
  const before = Float64Array.from(sim.dye);
  const mass0 = before.reduce((s, v) => s + v, 0);
  const report = sim.step(1);
  assert.equal(report.status, "ok", report.detail);
  assert.equal(report.substeps, 1);
  assert.equal(report.accepted, 1);
  const shift = speed > 0 ? 1 : -1;
  let l1 = 0;
  let l2 = 0;
  let n1 = 0;
  let n2 = 0;
  let linf = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const ii = axis === "x" ? (i - shift + nx) % nx : i;
        const jj = axis === "y" ? (j - shift + ny) % ny : j;
        const kk = axis === "z" ? (k - shift + nz) % nz : k;
        const exact = before[sim.ic(ii, jj, kk)];
        const got = sim.dye[sim.ic(i, j, k)];
        const e = got - exact;
        l1 += Math.abs(e);
        n1 += Math.abs(exact);
        l2 += e * e;
        n2 += exact * exact;
        linf = Math.max(linf, Math.abs(e));
      }
    }
  }
  return {
    relL1: l1 / n1,
    relL2: Math.sqrt(l2 / n2),
    linf,
    mass: sim.dye.reduce((s, v) => s + v, 0) - mass0,
    min: Math.min(...sim.dye),
    max: Math.max(...sim.dye),
  };
}

function smoothError(axis, n, scheme) {
  const nx = axis === "x" ? n : 4;
  const ny = axis === "y" ? n : 4;
  const nz = axis === "z" ? n : 4;
  const spec = makeCase({
    name: `smooth-${axis}`,
    nx,
    ny,
    nz,
    lx: 1,
    ly: 1,
    lz: 1,
    nu: 0,
    boundaries: periodicAll(),
    advection: scheme,
  });
  spec.cfl = 0.3;
  const sim = new MacSolver(spec);
  const U = 1;
  if (axis === "x") for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i <= nx; i++) sim.u[sim.iu(i, j, k)] = U;
  if (axis === "y") for (let k = 0; k < nz; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i < nx; i++) sim.v[sim.iv(i, j, k)] = U;
  if (axis === "z") for (let k = 0; k <= nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) sim.w[sim.iw(i, j, k)] = U;
  const coord = (i, j, k) => (axis === "x" ? (i + 0.5) / nx : axis === "y" ? (j + 0.5) / ny : (k + 0.5) / nz);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) sim.dye[sim.ic(i, j, k)] = Math.sin(2 * Math.PI * coord(i, j, k));
  const tEnd = 0.5;
  let guard = 0;
  while (sim.time < tEnd - 1e-12 && guard < 20000) {
    const report = sim.step(Math.min(sim.stableDt(), tEnd - sim.time));
    assert.equal(report.status, "ok", report.detail);
    guard += 1;
  }
  let l1 = 0;
  let l2 = 0;
  let n1 = 0;
  let n2 = 0;
  let linf = 0;
  let mass = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const exact = Math.sin(2 * Math.PI * (coord(i, j, k) - U * tEnd));
        const got = sim.dye[sim.ic(i, j, k)];
        const e = got - exact;
        l1 += Math.abs(e);
        n1 += Math.abs(exact);
        l2 += e * e;
        n2 += exact * exact;
        linf = Math.max(linf, Math.abs(e));
        mass += got;
      }
    }
  }
  return { relL1: l1 / n1, relL2: Math.sqrt(l2 / n2), linf, mass };
}

describe("axis-aware scalar transport", () => {
  for (const axis of ["x", "y", "z"]) {
    for (const sign of [1, -1]) {
      for (const n of [16, 32, 64]) {
        it(`upwind translates a ${axis} field by one cell at n=${n}, sign ${sign}`, () => {
          const err = shiftError(axis, n, sign);
          assert.ok(Math.abs(err.mass) < 1e-8, `mass ${err.mass}`);
          assert.ok(err.relL2 < 1e-8, `${axis} relL2 ${err.relL2}`);
          assert.ok(err.linf < 1e-8, `${axis} linf ${err.linf}`);
        });
      }
    }
  }

  it("reports the measured order of a smooth periodic wave, without assuming 2", () => {
    const rows = [];
    for (const scheme of ["upwind1", "muscl-rk2"]) {
      for (const axis of ["x", "y", "z"]) {
        const e16 = smoothError(axis, 16, scheme);
        const e32 = smoothError(axis, 32, scheme);
        const e64 = smoothError(axis, 64, scheme);
        const order32 = Math.log2(e16.relL2 / e32.relL2);
        const order64 = Math.log2(e32.relL2 / e64.relL2);
        rows.push({ scheme, axis, e16: e16.relL2, e32: e32.relL2, e64: e64.relL2, order32, order64, mass: e64.mass });
        assert.ok(Math.abs(e64.mass) < 1e-8, `${scheme} ${axis} mass ${e64.mass}`);
        assert.ok(e32.relL2 < e16.relL2, `${scheme} ${axis} did not improve 16→32`);
        assert.ok(e64.relL2 < e32.relL2, `${scheme} ${axis} did not improve 32→64`);
      }
    }
    const upwind = rows.filter((r) => r.scheme === "upwind1");
    const muscl = rows.filter((r) => r.scheme === "muscl-rk2");
    for (const row of upwind) {
      assert.ok(row.order64 > 0.8 && row.order64 < 1.25, `upwind ${row.axis} order ${row.order64}`);
    }
    for (const row of muscl) {
      assert.ok(row.e32 < upwind.find((r) => r.axis === row.axis).e32, `muscl ${row.axis} is not more accurate than upwind`);
      // Minmod on a sine is not a pure second-order scheme. Record the measured band.
      assert.ok(row.order64 > 1.2, `muscl ${row.axis} order ${row.order64} (expected above first order)`);
    }
    const x = muscl.find((r) => r.axis === "x");
    for (const row of muscl) {
      const ratio = row.e32 / x.e32;
      assert.ok(ratio > 0.95 && ratio < 1.05, `${row.axis} muscl L2 ${row.e32} disagrees with x ${x.e32}`);
    }
    console.log("smooth-wave errors", JSON.stringify(rows));
  });
});

describe("zero is a valid boundary scalar", () => {
  it("does not replace a zero outflow concentration with the exterior value", () => {
    for (const axis of ["x", "y", "z"]) {
      const b = emptyBoundaries();
      if (axis === "x") {
        b.west = outlet(0);
        b.west.backflow = "allow";
        b.east = outlet(0);
        b.east.backflow = "allow";
      } else if (axis === "y") {
        b.south = outlet(0);
        b.south.backflow = "allow";
        b.north = outlet(0);
        b.north.backflow = "allow";
      } else {
        b.back = outlet(0);
        b.back.backflow = "allow";
        b.front = outlet(0);
        b.front.backflow = "allow";
      }
      const spec = makeCase({
        name: `zero-${axis}`,
        nx: 4,
        ny: 4,
        nz: 4,
        lx: 4,
        ly: 4,
        lz: 4,
        nu: 0,
        boundaries: b,
        advection: "upwind1",
        tRef: 20,
      });
      const sim = new MacSolver(spec);
      sim.inletDye = 5;
      sim.dye.fill(0);
      sim.temperature.fill(0);
      const speed = -1;
      if (axis === "x") for (let k = 0; k < 4; k++) for (let j = 0; j < 4; j++) for (let i = 0; i <= 4; i++) sim.u[sim.iu(i, j, k)] = speed;
      if (axis === "y") for (let k = 0; k < 4; k++) for (let j = 0; j <= 4; j++) for (let i = 0; i < 4; i++) sim.v[sim.iv(i, j, k)] = speed;
      if (axis === "z") for (let k = 0; k <= 4; k++) for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) sim.w[sim.iw(i, j, k)] = speed;
      const report = sim.step(0.2);
      assert.equal(report.status, "ok", `${axis} ${report.detail}`);
      assert.ok(Math.abs(sim.dye[sim.ic(0, 0, 0)]) < 1e-12, `${axis} dye became ${sim.dye[0]}`);
      assert.ok(Math.abs(sim.temperature[sim.ic(0, 0, 0)]) < 1e-12, `${axis} temperature became ${sim.temperature[0]}`);
    }
  });

  it("admits a zero exterior value on an inflow face", () => {
    const b = emptyBoundaries();
    b.west = inlet(0.5, 0, 0, 0);
    b.east = outlet(0);
    b.east.backflow = "allow";
    const spec = makeCase({
      name: "zero-inflow",
      nx: 6,
      ny: 2,
      nz: 2,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: b,
      advection: "upwind1",
      tRef: 9,
    });
    const sim = new MacSolver(spec);
    sim.inletDye = 0;
    sim.dye.fill(4);
    sim.temperature.fill(4);
    const report = sim.step(0.05);
    assert.equal(report.status, "ok", report.detail);
    assert.ok(sim.dye[sim.ic(0, 0, 0)] < 4, `inflow dye did not enter, got ${sim.dye[0]}`);
    assert.ok(sim.temperature[sim.ic(0, 0, 0)] < 4, `zero wall temperature did not enter, got ${sim.temperature[0]}`);
  });
});

describe("transactional steps", () => {
  it("rolls the clock and the fields back when a later substep fails", () => {
    const spec = makeCase({
      name: "rollback",
      nx: 8,
      ny: 8,
      nz: 4,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0.02,
      boundaries: periodicAll(),
    });
    const sim = new MacSolver(spec);
    sim.u[sim.iu(2, 2, 1)] = 0.3;
    sim.dye[sim.ic(2, 2, 1)] = 0.7;
    sim.temperature[sim.ic(1, 1, 1)] = 1.5;
    sim.phi[sim.ic(3, 1, 1)] = 0.25;
    const limit = sim.stableDt();
    const first = sim.step(limit);
    assert.equal(first.status, "ok");
    assert.equal(first.substeps, 1);
    const snap = {
      time: sim.time,
      steps: sim.steps,
      u: Float64Array.from(sim.u),
      v: Float64Array.from(sim.v),
      w: Float64Array.from(sim.w),
      phi: Float64Array.from(sim.phi),
      dye: Float64Array.from(sim.dye),
      temperature: Float64Array.from(sim.temperature),
    };
    sim.failAtSubstep = 1;
    const requested = limit * 2.5;
    const report = sim.step(requested);
    assert.equal(report.status, "rejected");
    assert.equal(report.accepted, 0);
    assert.equal(report.substeps, 0);
    assert.equal(report.attempted, requested);
    assert.match(report.detail, /substep 1/);
    assert.equal(sim.time, snap.time);
    assert.equal(sim.steps, snap.steps);
    assert.deepEqual(sim.u, snap.u);
    assert.deepEqual(sim.v, snap.v);
    assert.deepEqual(sim.w, snap.w);
    assert.deepEqual(sim.phi, snap.phi);
    assert.deepEqual(sim.dye, snap.dye);
    assert.deepEqual(sim.temperature, snap.temperature);
  });

  it("counts every committed substep", () => {
    const spec = makeCase({
      name: "count",
      nx: 6,
      ny: 6,
      nz: 2,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0.02,
      boundaries: periodicAll(),
    });
    const sim = new MacSolver(spec);
    const limit = sim.stableDt();
    const requested = limit * 2.5;
    const report = sim.step(requested);
    assert.equal(report.status, "ok", report.detail);
    assert.equal(report.substeps, 3);
    assert.ok(Math.abs(report.accepted - requested) < 1e-12, `accepted ${report.accepted}`);
    assert.equal(sim.steps, 3);
    assert.ok(Math.abs(sim.time - requested) < 1e-12);
  });
});

describe("case and restart validation", () => {
  it("rejects a malformed case before constructing a solver", () => {
    const good = makeCase({ name: "ok", nx: 4, ny: 4, nz: 2, lx: 1, ly: 1, lz: 1, nu: 0.01, boundaries: emptyBoundaries() });
    const text = JSON.stringify(good);
    assert.equal(caseFromJson(text).pressure, "kinematic");
    const broken = JSON.parse(text);
    broken.boundaries.east = { ...broken.boundaries.east, kind: "periodic" };
    assert.throws(() => caseFromJson(JSON.stringify(broken)), CaseError);
    broken.boundaries.east.kind = "wall";
    broken.nu = -1;
    assert.throws(() => caseFromJson(JSON.stringify(broken)), /viscosity|nu/i);
    const huge = JSON.parse(text);
    huge.nx = 200;
    huge.ny = 200;
    huge.nz = 200;
    assert.throws(() => caseFromJson(JSON.stringify(huge)), /cap|cells/i);
    assert.throws(() => caseFromJson("{"), CaseError);
  });

  it("rejects a restart whose arrays do not match the grid", () => {
    const sim = new MacSolver(makeCase({ name: "r", nx: 4, ny: 3, nz: 2, lx: 1, ly: 1, lz: 1, nu: 0, boundaries: emptyBoundaries() }));
    sim.u[sim.iu(1, 1, 0)] = 0.2;
    sim.step(1e-3);
    const copy = MacSolver.fromRestart(sim.toRestart());
    assert.equal(copy.time, sim.time);
    assert.equal(copy.steps, sim.steps);
    assert.equal(copy.u[copy.iu(1, 1, 0)], sim.u[sim.iu(1, 1, 0)]);
    const bad = JSON.parse(sim.toRestart());
    bad.u[0] = Number.NaN;
    assert.throws(() => MacSolver.fromRestart(JSON.stringify(bad)), /finite/);
    bad.u[0] = 0;
    bad.u.pop();
    assert.throws(() => MacSolver.fromRestart(JSON.stringify(bad)), /length/);
    const badTime = JSON.parse(sim.toRestart());
    badTime.time = -1;
    assert.throws(() => MacSolver.fromRestart(JSON.stringify(badTime)), /time/);
  });
});

describe("projection compatibility", () => {
  it("refuses a closed inlet instead of reporting a converged divergence", () => {
    const b = emptyBoundaries();
    b.west = inlet(0.5, 0, 0);
    const sim = new MacSolver(makeCase({
      name: "closed-inlet",
      nx: 8,
      ny: 6,
      nz: 4,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: b,
    }));
    const report = sim.step(0.01);
    assert.equal(report.status, "rejected");
    assert.equal(sim.time, 0);
    assert.equal(sim.steps, 0);
    assert.match(report.detail, /volume flux|divergence/i);
  });

  it("refuses an outlet clamp that would hide a mass imbalance", () => {
    const b = emptyBoundaries();
    b.west = inlet(-0.25, 0, 0);
    b.east = outlet(0);
    const sim = new MacSolver(makeCase({
      name: "suction",
      nx: 8,
      ny: 3,
      nz: 2,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: b,
    }));
    const report = sim.step(0.01);
    assert.equal(report.status, "rejected");
    assert.equal(sim.time, 0);
    assert.equal(report.accepted, 0);
    assert.match(report.detail, /clamp|volume flux|divergence/i);
    assert.equal(sim.u[sim.iu(0, 0, 0)], -0.25);
  });

  it("keeps a compatible channel divergence-free when backflow is allowed", () => {
    const b = emptyBoundaries();
    b.south = inlet(0, 0.4, 0);
    b.north = outlet(0);
    b.north.backflow = "allow";
    const sim = new MacSolver(makeCase({
      name: "channel",
      nx: 6,
      ny: 8,
      nz: 4,
      lx: 1,
      ly: 1,
      lz: 1,
      nu: 0,
      boundaries: b,
    }));
    const report = sim.step(0.01);
    assert.equal(report.status, "ok", report.detail);
    assert.ok(report.divLinf < 1e-6, report.divLinf);
    assert.ok(Math.abs(report.massImbalance) < 1e-6);
  });
});
