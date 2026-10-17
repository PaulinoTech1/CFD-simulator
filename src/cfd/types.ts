export const F_FLUID = 0;
export const F_SOLID = 1;
export const F_INLET = 2;
export const F_LID = 3;
export const F_BODY = 4;

export type ViewMode = "dye" | "vorticity" | "speed" | "pressure";

export type ToolMode = "stir" | "wall" | "erase" | "probe";

export type PresetId =
  | "karman"
  | "airfoil"
  | "cavity"
  | "plume"
  | "kelvin"
  | "jet"
  | "ink";

export type Injector = { i: number; j: number; rate: number };

export type Scene = {
  id: PresetId;
  nx: number;
  ny: number;
  flags: Uint8Array;
  u: Float32Array;
  v: Float32Array;
  dye: Float32Array;
  inject: Injector[];
  periodicX: boolean;
  uIn: number;
  charLength: number;
  hasBody: boolean;
  dyeInlet: boolean;
  dyeDecay: number;
  buoyancy: number;
  vorticity: number;
  re: number;
  view: ViewMode;
  vectors: boolean;
  driven: boolean;
};

export type SimParams = {
  re: number;
  uIn: number;
  vorticity: number;
  buoyancy: number;
  iterations: number;
  dyeDecay: number;
};

export type SimStats = {
  time: number;
  dt: number;
  ke: number;
  maxSpeed: number;
  maxAbsOmega: number;
  fx: number;
  fy: number;
  cd: number;
  cl: number;
  re: number;
  cfl: number;
  strouhal: number;
  steps: number;
};

export type Splat = {
  x: number;
  y: number;
  du: number;
  dv: number;
  dye: number;
  radius: number;
};
