import type { Fluid3 } from "./solver3d.ts";
import { F_BODY, F_SOLID } from "./types.ts";

export type PreviewProfiles = {
  /** Cell-center height, 0 at the bottom fluid row. */
  height: Float32Array;
  /** Streamwise speed at mid-length, mid-span. NaN inside a solid. */
  speed: Float32Array;
  /** Temperature on the same line. */
  temperature: Float32Array;
  /** Streamwise position, 0 at the inlet side. */
  station: Float32Array;
  /** Pressure at mid-height, mid-span. NaN inside a solid. */
  pressure: Float32Array;
};

function blocked(flag: number): boolean {
  return flag === F_SOLID || flag === F_BODY;
}

/** Mid-span cuts. These are samples of the collocated preview, not the MAC grid. */
export function samplePreview(sim: Fluid3): PreviewProfiles {
  const { nx, ny, nz, flags, u, temp, p } = sim;
  const k = Math.max(0, Math.min(nz - 1, (nz / 2) | 0));
  const iMid = Math.max(0, Math.min(nx - 1, (nx / 2) | 0));
  const jMid = Math.max(0, Math.min(ny - 1, (ny / 2) | 0));
  const height = new Float32Array(ny);
  const speed = new Float32Array(ny);
  const temperature = new Float32Array(ny);
  for (let j = 0; j < ny; j++) {
    height[j] = ny <= 1 ? 0 : j / (ny - 1);
    const id = iMid + nx * (j + ny * k);
    if (blocked(flags[id]!)) {
      speed[j] = Number.NaN;
      temperature[j] = Number.NaN;
    } else {
      speed[j] = u[id]!;
      temperature[j] = temp[id]!;
    }
  }
  const station = new Float32Array(nx);
  const pressure = new Float32Array(nx);
  for (let i = 0; i < nx; i++) {
    station[i] = nx <= 1 ? 0 : i / (nx - 1);
    const id = i + nx * (jMid + ny * k);
    pressure[i] = blocked(flags[id]!) ? Number.NaN : p[id]!;
  }
  return { height, speed, temperature, station, pressure };
}
