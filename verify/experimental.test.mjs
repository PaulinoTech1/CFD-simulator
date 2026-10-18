import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { conjugateBalance, firstFrequency, isentropicDensityRatio, runExperimental, sectionStiffness } from "../src/cfd/experimental.ts";

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

  it("compressible density ratio is near 1 at low Mach and flags Mach 0.3", () => {
    const low = runExperimental("compressible", { ...base, uIn: 1 });
    const high = runExperimental("compressible", { ...base, uIn: 200 });
    assert.match(low.value, /ρ\/ρ0 = 1\.0000/);
    assert.match(low.detail, /inside its usual envelope/);
    assert.match(high.detail, /outside its usual envelope/);
    const ratio = isentropicDensityRatio(1);
    assert.ok(Math.abs(ratio - 1.2 ** -2.5) < 1e-12);
  });

  it("conjugate flux matches conduction through the same wall drop", () => {
    const { q, tInterface } = conjugateBalance(20, 40);
    const throughWall = (16 / 0.01) * (40 - tInterface);
    assert.ok(Math.abs(q - throughWall) < 1e-9);
  });

  it("section frequency matches the single-degree-of-freedom formula", () => {
    const k = sectionStiffness(2);
    const f = firstFrequency(2);
    assert.ok(Math.abs(f - Math.sqrt(k) / (2 * Math.PI)) < 1e-12);
  });

  it("nonlinear flags strain strictly above 1 percent", () => {
    const low = runExperimental("nonlinear", { ...base, vonMises: 1e6, young: 2 });
    const edge = runExperimental("nonlinear", { ...base, vonMises: 0.01 * 2e9, young: 2 });
    const high = runExperimental("nonlinear", { ...base, vonMises: 5e7, young: 2 });
    assert.match(low.value, /at or under 1%/);
    assert.match(edge.value, /at or under 1%/);
    assert.match(high.value, /above 1%/);
  });

  it("multiphase does not invent a volume fraction", () => {
    const r = runExperimental("multiphase", base);
    assert.match(r.value, /no phase field/);
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
