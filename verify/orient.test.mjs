import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildScene3 } from "../src/cfd/presets3d.ts";
import { Fluid3 } from "../src/cfd/solver3d.ts";
import { F_BODY } from "../src/cfd/types.ts";

function spread(sim) {
  let di = 0;
  let dj = 0;
  let dk = 0;
  let n = 0;
  const { nx, ny, nz, flags } = sim;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (flags[i + nx * (j + ny * k)] !== F_BODY) continue;
        di += Math.abs(i - sim.cx);
        dj += Math.abs(j - sim.cy);
        dk += Math.abs(k - sim.cz);
        n += 1;
      }
    }
  }
  return { di: di / n, dj: dj / n, dk: dk / n, n };
}

describe("solid orientation", () => {
  it("turns a cylinder from the span axis onto the stream axis", () => {
    const sim = new Fluid3();
    sim.load(buildScene3("ink", 16));
    sim.reshape("cylinder", 2.4);
    const flat = spread(sim);
    sim.setOrientation(90, 0);
    const turned = spread(sim);
    assert.ok(flat.n > 8);
    assert.ok(flat.dk > flat.di, `along z ${flat.dk} vs ${flat.di}`);
    assert.ok(turned.di > turned.dk, `along x ${turned.di} vs ${turned.dk}`);
  });

  it("tilts a cone and a plate", () => {
    const sim = new Fluid3();
    sim.load(buildScene3("ink", 16));
    sim.reshape("cone", 2.4);
    const cone = spread(sim);
    sim.setOrientation(0, 90);
    const tilted = spread(sim);
    assert.ok(cone.di > cone.dj);
    assert.ok(tilted.dj > cone.dj, `pitch did not grow the vertical extent`);
    sim.setOrientation(0, 0);
    sim.reshape("plate", 2.4);
    const plate = spread(sim);
    sim.setOrientation(90, 0);
    const plateTurned = spread(sim);
    assert.notEqual(plate.di.toFixed(2), plateTurned.di.toFixed(2));
  });

  it("rolls a cylinder upright and banks an airfoil", () => {
    const sim = new Fluid3();
    sim.load(buildScene3("ink", 16));
    sim.reshape("cylinder", 2.4);
    const lying = spread(sim);
    sim.setOrientation(0, 0, 90);
    const upright = spread(sim);
    assert.ok(lying.dk > lying.dj);
    assert.ok(upright.dj > upright.dk, `roll left the cylinder along z ${upright.dk} vs ${upright.dj}`);
    sim.setOrientation(0, 0, 0);
    sim.reshape("airfoil", 2.4);
    const flat = spread(sim);
    sim.setOrientation(0, 0, 90);
    const twisted = spread(sim);
    assert.ok(flat.dk > flat.dj);
    assert.ok(twisted.dj > twisted.dk);
  });

  it("leaves a sphere unchanged", () => {
    const sim = new Fluid3();
    sim.load(buildScene3("ink", 16));
    sim.reshape("sphere", 2.4);
    const a = sim.bodyCells();
    sim.setOrientation(90, 45);
    assert.equal(sim.bodyCells(), a);
  });
});
