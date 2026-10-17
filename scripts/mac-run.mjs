// Headless Couette check against the TypeScript MAC reference.
// Matches the verified case in verify/numerics.test.mjs.
// node --experimental-strip-types scripts/mac-run.mjs
import { emptyBoundaries, lid, makeCase, periodic, wall } from "../src/cfd/case.ts";
import { MacSolver } from "../src/cfd/mac.ts";

const ny = 24;
const speed = 1;
const nu = 0.08;
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
let report = null;
let guard = 0;
while (sim.time < 12 && guard < 60000) {
  report = sim.step(sim.stableDt());
  if (report.status !== "ok") break;
  guard += 1;
}
let err = 0;
for (let j = 0; j < ny; j++) {
  const exact = speed * (j + 0.5) * sim.dy;
  err = Math.max(err, Math.abs(sim.u[sim.iu(0, j, 0)] - exact));
}
console.log(
  `couette n=${ny} steps=${sim.steps} time=${sim.time.toExponential(3)} max|u-exact|=${err.toExponential(3)} status=${report?.status ?? "none"}`,
);
if (report?.status !== "ok" || !(err < 1e-3)) {
  console.error("couette did not reach the verified tolerance");
  process.exit(1);
}
