import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { cellBudget } from "../src/cfd/device.ts";

describe("cell budget", () => {
  it("marks the bench grid as inside preview comfort and L2", () => {
    const b = cellBudget(32, 16, 16);
    assert.equal(b.cells, 8192);
    assert.equal(b.previewOk, true);
    assert.equal(b.withinReference, true);
    assert.equal(b.fitsVram, true);
  });

  it("flags a grid above the reference cap", () => {
    const b = cellBudget(200, 200, 100);
    assert.ok(b.cells > 2_000_000);
    assert.equal(b.withinReference, false);
    assert.match(b.note, /reference cap/);
  });

  it("stays finite and names the hardware in the note", () => {
    const b = cellBudget(128, 64, 64);
    assert.ok(Number.isFinite(b.previewBytes));
    assert.ok(Number.isFinite(b.referenceBytes));
    assert.ok(b.note.length > 0);
  });
});
