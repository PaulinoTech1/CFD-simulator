/** 256×3 RGB lookup, scientific ramps kept inside the instrument palette. */

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function sampleStops(stops: ReadonlyArray<readonly [number, number, number, number]>, t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  let i = 0;
  while (i < stops.length - 2 && x > stops[i + 1]![0]) i++;
  const a = stops[i]!;
  const b = stops[i + 1]!;
  const span = b[0] - a[0] || 1;
  const u = (x - a[0]) / span;
  return [lerp(a[1], b[1], u), lerp(a[2], b[2], u), lerp(a[3], b[3], u)];
}

export function buildLut(stops: ReadonlyArray<readonly [number, number, number, number]>): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = sampleStops(stops, i / 255);
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}

const INK: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, 14, 18, 16],
  [0.12, 24, 38, 32],
  [0.38, 168, 96, 28],
  [0.68, 230, 176, 72],
  [1, 244, 240, 228],
];

const DIVERGE: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, 28, 92, 96],
  [0.35, 18, 40, 40],
  [0.5, 16, 20, 18],
  [0.65, 92, 52, 18],
  [1, 232, 168, 48],
];

const SPEED: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, 12, 16, 14],
  [0.25, 32, 58, 48],
  [0.55, 196, 140, 42],
  [1, 248, 244, 232],
];

export const LUT_INK = buildLut(INK);
export const LUT_DIVERGE = buildLut(DIVERGE);
export const LUT_SPEED = buildLut(SPEED);

export function lutAt(lut: Uint8ClampedArray, t: number, out: { r: number; g: number; b: number }): void {
  const i = Math.max(0, Math.min(255, (t * 255) | 0)) * 3;
  out.r = lut[i]!;
  out.g = lut[i + 1]!;
  out.b = lut[i + 2]!;
}
