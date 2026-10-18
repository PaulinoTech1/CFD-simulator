import { emptyBoundaries, lid, makeCase, periodic, wall, caseToJson, type CaseSpec } from "./case.ts";
import { MacSolver } from "./mac.ts";

export type CheckResult = {
  name: string;
  pass: boolean;
  value: string;
};

/** One viscous step of a shear mode. Same discrete check as verify/numerics.test.mjs. */
export function runShearCheck(): CheckResult {
  const n = 24;
  const nu = 0.02;
  const spec = makeCase({
    name: "shear",
    nx: n,
    ny: n,
    nz: 1,
    lx: 1,
    ly: 1,
    lz: 1,
    nu,
    advection: "none",
    boundaries: periodicBox(),
  });
  const sim = new MacSolver(spec);
  for (let j = 0; j < n; j++) {
    const value = Math.sin((2 * Math.PI * (j + 0.5)) / n);
    for (let i = 0; i <= n; i++) sim.u[sim.iu(i, j, 0)] = value;
  }
  const kappa = (2 * (Math.cos((2 * Math.PI) / n) - 1)) / sim.dy ** 2;
  const dt = 0.2 * sim.stableDt();
  const report = sim.step(dt);
  let err = 0;
  for (let j = 0; j < n; j++) {
    const expected = Math.sin((2 * Math.PI * (j + 0.5)) / n) * (1 + dt * nu * kappa);
    err = Math.max(err, Math.abs(sim.u[sim.iu(0, j, 0)] - expected));
  }
  const pass = report.status === "ok" && err < 1e-12;
  return { name: "shear", pass, value: `max error ${err.toExponential(2)}` };
}

/** Uniform temperature must not start a Boussinesq current. */
export function runHeatRestCheck(): CheckResult {
  const b = emptyBoundaries();
  b.back = periodic();
  b.front = periodic();
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
  let failed = "";
  for (let n = 0; n < 20; n++) {
    const report = sim.step(sim.stableDt());
    if (report.status !== "ok") failed = report.detail;
  }
  const pass = failed === "" && sim.last.maxSpeed < 1e-8;
  return { name: "heat-rest", pass, value: failed || `max speed ${sim.last.maxSpeed.toExponential(2)}` };
}

export function couetteCase(): CaseSpec {
  const ny = 24;
  const b = emptyBoundaries();
  b.west = periodic();
  b.east = periodic();
  b.back = periodic();
  b.front = periodic();
  b.south = wall(0, 0, 0);
  b.north = lid(1, 0, 0);
  return makeCase({
    name: "couette",
    nx: 4,
    ny,
    nz: 1,
    lx: 1,
    ly: 1,
    lz: 1,
    nu: 0.08,
    boundaries: b,
    uRef: 1,
    lRef: 1,
  });
}

export function couetteJson(): string {
  return caseToJson(couetteCase());
}

export function couetteError(sim: MacSolver): number {
  let err = 0;
  for (let j = 0; j < sim.spec.ny; j++) {
    err = Math.max(err, Math.abs(sim.u[sim.iu(0, j, 0)] - (j + 0.5) * sim.dy));
  }
  return err;
}

function periodicBox(): CaseSpec["boundaries"] {
  const b = emptyBoundaries();
  b.west = periodic();
  b.east = periodic();
  b.south = periodic();
  b.north = periodic();
  b.back = periodic();
  b.front = periodic();
  return b;
}
