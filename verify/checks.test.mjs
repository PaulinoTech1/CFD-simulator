import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { couetteCase, couetteError, runHeatRestCheck, runShearCheck } from "../src/cfd/checks.ts";
import { caseFromJson, caseToJson } from "../src/cfd/case.ts";
import { MacSolver } from "../src/cfd/mac.ts";

describe("buttons that claim a check", () => {
  it("shear step matches the discrete decay", () => {
    const r = runShearCheck();
    assert.equal(r.pass, true, r.value);
  });

  it("heat rest stays still", () => {
    const r = runHeatRestCheck();
    assert.equal(r.pass, true, r.value);
  });

  it("the downloadable Couette case is the one the error measures", () => {
    const spec = caseFromJson(caseToJson(couetteCase()));
    const sim = new MacSolver(spec);
    const report = sim.step(sim.stableDt());
    assert.equal(report.status, "ok");
    assert.ok(Number.isFinite(couetteError(sim)));
  });
});
