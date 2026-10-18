import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { buildScene3 } from "../src/cfd/presets3d.ts";
import { samplePreview } from "../src/cfd/profiles.ts";
import { Fluid3 } from "../src/cfd/solver3d.ts";
import { F_BODY } from "../src/cfd/types.ts";

describe("preview charts", () => {
  it("reads the mid-span speed and leaves a gap in a solid", () => {
    const sim = new Fluid3();
    sim.load(buildScene3("ink", 8));
    const { nx, ny, nz, u, flags } = sim;
    const k = (nz / 2) | 0;
    const iMid = (nx / 2) | 0;
    for (let j = 0; j < ny; j++) u[iMid + nx * (j + ny * k)] = j / ny;
    const jSolid = Math.min(ny - 2, ((ny / 2) | 0) + 1);
    flags[iMid + nx * (jSolid + ny * k)] = F_BODY;
    const cut = samplePreview(sim);
    assert.ok(Math.abs(cut.speed[1] - 1 / ny) < 1e-6);
    assert.ok(Number.isNaN(cut.speed[jSolid]));
    assert.equal(cut.pressure.length, nx);
    assert.equal(cut.height[ny - 1], 1);
  });
});
