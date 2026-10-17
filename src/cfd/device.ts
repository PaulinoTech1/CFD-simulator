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
