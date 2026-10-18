import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { runExperimental } from "../src/cfd/experimental.ts";

const base = {
  re: 180,
  uIn: 1,
  maxSpeed: 1.2,
  tIn: 20,
  tWall: 40,
  young: 2,
  vonMises: 1e6,
  meanPressure: 0.1,
};

const ids = [
  "lbm",
  "multipurpose",
  "comfort",
  "compressible",
  "conjugate",
  "multiphase",
  "dynamic",
  "heatSolid",
  "thermomech",
  "harmonic",
  "frequency",
  "nonlinear",
];

describe("experimental analysis models", () => {
  for (const id of ids) {
    it(`${id} returns a finite reading`, () => {
      const r = runExperimental(id, base);
      assert.equal(typeof r.title, "string");
      assert.ok(r.title.length > 0);
      assert.equal(typeof r.value, "string");
      assert.ok(r.value.length > 0);
      assert.equal(typeof r.detail, "string");
      assert.ok(r.detail.length > 0);
      assert.ok(!r.value.includes("NaN"), `${id} value has NaN: ${r.value}`);
      assert.ok(!r.value.includes("Infinity"), `${id} value has Infinity: ${r.value}`);
    });
  }

  it("comfort class changes with speed", () => {
    const slow = runExperimental("comfort", { ...base, maxSpeed: 0.5 });
    const fast = runExperimental("comfort", { ...base, maxSpeed: 8 });
    assert.match(slow.value, /sitting/);
    assert.match(fast.value, /uncomfortable/);
  });

  it("compressible density ratio is near 1 at low Mach", () => {
    const r = runExperimental("compressible", { ...base, uIn: 1 });
    assert.match(r.value, /ρ\/ρ0 ≈ 1\.000/);
  });

  it("nonlinear flags high strain and stays quiet at low strain", () => {
    const low = runExperimental("nonlinear", { ...base, vonMises: 1e6, young: 2 });
    const high = runExperimental("nonlinear", { ...base, vonMises: 5e7, young: 2 });
    assert.match(low.value, /linear range/);
    assert.match(high.value, /flagged/);
  });

  it("unknown id falls back to the checked path", () => {
    const r = runExperimental("nope", base);
    assert.match(r.title, /Incompressible/);
  });

  it("zero Reynolds does not produce NaN in LBM", () => {
    const r = runExperimental("lbm", { ...base, re: 0 });
    assert.ok(!r.value.includes("NaN"));
  });
});
