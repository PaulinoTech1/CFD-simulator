/** Hardware the bench is written against. The live preview does not launch the GPU. */

export const GPU = {
  name: "GeForce RTX 3090 Ti",
  chip: "GA102",
  arch: "Ampere",
  cc: "8.6",
  sms: 84,
  fp32: 10752,
  boostGhz: 1.86,
  l2Mb: 6,
  busBits: 384,
  vramGb: 24,
  bandwidthGBps: 1008,
  tgpW: 450,
  block: 16,
} as const;

export const CPU = {
  name: "Ryzen 9 5900X",
  arch: "Zen 3",
  cores: 12,
  threads: 24,
  l3Mb: 64,
  ramGb: 64,
} as const;

export type LaunchPlan = {
  block: number;
  gridX: number;
  gridY: number;
  gridZ: number;
  threads: number;
  bytes: number;
  fitsL2: boolean;
  jacobiBytes: number;
  dramMs: number;
  note: string;
};

export function planLaunch(nx: number, ny: number): LaunchPlan {
  const block = GPU.block;
  const gridX = Math.max(1, Math.ceil(nx / block));
  const gridY = Math.max(1, Math.ceil(ny / block));
  const threads = gridX * gridY * block * block;
  // u, v, uT, vT, dye, dyeT, p, div, omega  + flags
  const bytes = nx * ny * (9 * 4 + 1);
  const fitsL2 = bytes < GPU.l2Mb * 1024 * 1024;
  // One Jacobi sweep: read p (with halo, counted once) and write p.
  const jacobiBytes = nx * ny * 4 * 2;
  const dramMs = jacobiBytes / (GPU.bandwidthGBps * 1e9) * 1e3;
  const note = fitsL2
    ? "Working set fits in the 6 MB L2. A 2D slice this size is latency-bound on GA102 — the 1008 GB/s bus shows up once the grid leaves L2 (about 1024², or any 3D volume)."
    : "Working set is larger than L2. Pressure Jacobi is DRAM-bound; the useful ceiling is the 1008 GB/s GDDR6X bus, not the 40 TFLOP FP32 peak.";
  return { block, gridX, gridY, gridZ: 1, threads, bytes, fitsL2, jacobiBytes, dramMs, note };
}

/** 8×4×8 thread blocks. The preview does not launch these; the count is the GA102 map of this volume. */
export function planVolume(nx: number, ny: number, nz: number): LaunchPlan {
  const gridX = Math.max(1, Math.ceil(nx / 8));
  const gridY = Math.max(1, Math.ceil(ny / 4));
  const gridZ = Math.max(1, Math.ceil(nz / 8));
  const threads = gridX * gridY * gridZ * 256;
  const bytes = nx * ny * nz * (10 * 4 + 1);
  const fitsL2 = bytes < GPU.l2Mb * 1024 * 1024;
  const jacobiBytes = nx * ny * nz * 4 * 2;
  const dramMs = jacobiBytes / (GPU.bandwidthGBps * 1e9) * 1e3;
  const note = fitsL2
    ? "The 3D working set still fits in the 6 MB L2. A volume this small is launch-latency bound on GA102. The 1008 GB/s bus matters once the grid leaves L2."
    : "The volume is larger than L2. Pressure Jacobi is then limited by the GDDR6X bus, not by the 40 TFLOP FP32 peak.";
  return { block: 8, gridX, gridY, gridZ, threads, bytes, fitsL2, jacobiBytes, dramMs, note };
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

/** Cell-count boundaries for the stated hardware. Estimates, not measurements. */
export type CellBudget = {
  cells: number;
  previewBytes: number;
  referenceBytes: number;
  fitsL2: boolean;
  fitsVram: boolean;
  fitsRam: boolean;
  withinReference: boolean;
  previewOk: boolean;
  note: string;
};

const PREVIEW_BYTES_PER_CELL = 10 * 4 + 1;
const REFERENCE_BYTES_PER_CELL = 160;
const PREVIEW_COMFORT_CELLS = 200_000;

export function cellBudget(nx: number, ny: number, nz: number): CellBudget {
  const cells = nx * ny * nz;
  const previewBytes = cells * PREVIEW_BYTES_PER_CELL;
  const referenceBytes = cells * REFERENCE_BYTES_PER_CELL;
  const l2 = GPU.l2Mb * 1024 * 1024;
  const vram = GPU.vramGb * 1024 * 1024 * 1024 * 0.7;
  const ram = CPU.ramGb * 1024 * 1024 * 1024 * 0.5;
  const fitsL2 = previewBytes < l2;
  const fitsVram = referenceBytes < vram;
  const fitsRam = referenceBytes < ram;
  const withinReference = cells <= 2_000_000;
  const previewOk = cells <= PREVIEW_COMFORT_CELLS;
  let note = "Inside the browser preview comfort zone.";
  if (!previewOk && withinReference) note = "Above the browser comfort zone. The preview will slow down. The reference solver still accepts this size.";
  if (!withinReference && fitsVram) note = "Above the 2,000,000-cell reference cap. The checked MAC solver will refuse this grid.";
  if (!fitsVram) note = "Working set estimate exceeds 70% of the 24 GB VRAM. Do not target this size on the 3090 Ti.";
  if (!fitsRam) note = "Working set estimate exceeds half of the 64 GB system memory.";
  return { cells, previewBytes, referenceBytes, fitsL2, fitsVram, fitsRam, withinReference, previewOk, note };
}
