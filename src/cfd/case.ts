/** Versioned incompressible case shared by the browser reference and kernels/karman3d.cu. */

export const CASE_VERSION = 1 as const;

/**
 * Stored pressure is kinematic: φ = p / ρ, units m²/s².
 * The projection is u := u − Δt ∇φ. Physical pressure is ρ φ.
 * An imposed streamwise gradient is a body acceleration, not an extra term inside φ:
 *   ∂u/∂t += fx,  fx = −(1/ρ) ∂p/∂x = −∂φ/∂x.
 */
export type PressureConvention = "kinematic";

export type BoundaryKind = "wall" | "inlet" | "lid" | "outlet" | "periodic";

/** Velocity components are m/s. Outlet φ is kinematic pressure outside the domain. */
export type FaceBoundary = {
  kind: BoundaryKind;
  u: number;
  v: number;
  w: number;
  phi: number;
  /** Outlet only. "clamp" sets reversed outlet velocity to 0 after the projection. */
  backflow: "clamp" | "allow";
  /** Scalar Dirichlet. Null means adiabatic (zero normal derivative) for diffusion. */
  temperature: number | null;
};

export type Vec3 = { x: number; y: number; z: number };

export type CaseSpec = {
  version: typeof CASE_VERSION;
  name: string;
  nx: number;
  ny: number;
  nz: number;
  lx: number;
  ly: number;
  lz: number;
  /** kg/m³. Multiplies φ and viscous stress when forces are reported in Newtons. */
  rho: number;
  /** m²/s. Never replaced by a clipped product dt·ν. */
  nu: number;
  pressure: PressureConvention;
  /** Body acceleration added before the projection, m/s². */
  force: Vec3;
  /** Gravitational acceleration, m/s². */
  gravity: Vec3;
  /** Thermal expansion, 1/K. Buoyancy acceleration is −β (T−Tref) g. */
  beta: number;
  tRef: number;
  /** Thermal diffusivity m²/s. */
  alpha: number;
  /** Passive dye diffusivity. Zero keeps dye conservative aside from the inlet. */
  dyeDiffusivity: number;
  dyeCouplesDensity: false;
  boundaries: {
    west: FaceBoundary;
    east: FaceBoundary;
    south: FaceBoundary;
    north: FaceBoundary;
    back: FaceBoundary;
    front: FaceBoundary;
  };
  uRef: number;
  lRef: number;
  /** Null means force coefficients are not defined for this case. */
  areaRef: number | null;
  cfl: number;
  diffCfl: number;
  poisson: { atol: number; rtol: number; maxIterations: number };
  /** "none" is for viscous and operator tests. muscl-rk2 is limited linear reconstruction. */
  advection: "none" | "upwind1" | "muscl-rk2";
};

export type BoundaryName = keyof CaseSpec["boundaries"];

export class CaseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CaseError";
  }
}

/** Reference solver refuses grids above this cell count before it allocates. */
export const MAX_CELLS = 2_000_000;

const KINDS = new Set<BoundaryKind>(["wall", "inlet", "lid", "outlet", "periodic"]);
const ADVECTION = new Set<CaseSpec["advection"]>(["none", "upwind1", "muscl-rk2"]);

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new CaseError(`${label} must be a finite number`);
  return value;
}

function positiveFinite(value: unknown, label: string): number {
  const n = finiteNumber(value, label);
  if (!(n > 0)) throw new CaseError(`${label} must be positive`);
  return n;
}

function nonnegativeFinite(value: unknown, label: string): number {
  const n = finiteNumber(value, label);
  if (n < 0) throw new CaseError(`${label} must be nonnegative`);
  return n;
}

function positiveInt(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new CaseError(`${label} must be a positive integer`);
  }
  return value;
}

function vec3(value: unknown, label: string): Vec3 {
  if (!value || typeof value !== "object") throw new CaseError(`${label} must be a vector`);
  const v = value as Vec3;
  return { x: finiteNumber(v.x, `${label}.x`), y: finiteNumber(v.y, `${label}.y`), z: finiteNumber(v.z, `${label}.z`) };
}

function validateFace(name: string, face: unknown): FaceBoundary {
  if (!face || typeof face !== "object") throw new CaseError(`${name} boundary is missing`);
  const f = face as FaceBoundary;
  if (!KINDS.has(f.kind)) throw new CaseError(`${name} has unsupported boundary kind ${String(f.kind)}`);
  if (f.backflow !== "clamp" && f.backflow !== "allow") throw new CaseError(`${name}.backflow must be clamp or allow`);
  return {
    kind: f.kind,
    u: finiteNumber(f.u, `${name}.u`),
    v: finiteNumber(f.v, `${name}.v`),
    w: finiteNumber(f.w, `${name}.w`),
    phi: finiteNumber(f.phi, `${name}.phi`),
    backflow: f.backflow,
    temperature: f.temperature === null ? null : finiteNumber(f.temperature, `${name}.temperature`),
  };
}

/**
 * Strict check for a version-1 case. Documents produced by `makeCase` pass.
 * Malformed input is rejected before a solver allocates. A stored scalar of zero
 * is valid and must not be treated as a missing value.
 */
export function validateCase(input: unknown): CaseSpec {
  if (!input || typeof input !== "object") throw new CaseError("case must be an object");
  const raw = input as CaseSpec;
  if (raw.version !== CASE_VERSION) throw new CaseError(`unsupported case version ${String(raw.version)}`);
  if (typeof raw.name !== "string" || raw.name.length === 0) throw new CaseError("case name is required");
  if (raw.pressure !== "kinematic") throw new CaseError("only kinematic pressure (phi = p/rho) is defined");
  if (raw.dyeCouplesDensity !== false) throw new CaseError("dye is passive in this case version");
  const nx = positiveInt(raw.nx, "nx");
  const ny = positiveInt(raw.ny, "ny");
  const nz = positiveInt(raw.nz, "nz");
  if (nx > 4096 || ny > 4096 || nz > 4096) throw new CaseError("a single grid dimension exceeds 4096");
  if (nx * ny * nz > MAX_CELLS) throw new CaseError(`grid has ${nx * ny * nz} cells; the reference cap is ${MAX_CELLS}`);
  const boundaries = {
    west: validateFace("west", raw.boundaries?.west),
    east: validateFace("east", raw.boundaries?.east),
    south: validateFace("south", raw.boundaries?.south),
    north: validateFace("north", raw.boundaries?.north),
    back: validateFace("back", raw.boundaries?.back),
    front: validateFace("front", raw.boundaries?.front),
  };
  const pairs: Array<[BoundaryName, BoundaryName]> = [
    ["west", "east"],
    ["south", "north"],
    ["back", "front"],
  ];
  for (const [a, b] of pairs) {
    if ((boundaries[a].kind === "periodic") !== (boundaries[b].kind === "periodic")) {
      throw new CaseError(`${a} and ${b} must both be periodic, or neither`);
    }
  }
  const cfl = positiveFinite(raw.cfl, "cfl");
  const diffCfl = positiveFinite(raw.diffCfl, "diffCfl");
  if (cfl > 1) throw new CaseError("cfl must be in (0, 1]");
  if (diffCfl > 1) throw new CaseError("diffCfl must be in (0, 1]");
  if (!raw.poisson || typeof raw.poisson !== "object") throw new CaseError("poisson settings are required");
  if (!ADVECTION.has(raw.advection)) throw new CaseError(`unsupported advection scheme ${String(raw.advection)}`);
  let areaRef: number | null;
  if (raw.areaRef === null) areaRef = null;
  else areaRef = positiveFinite(raw.areaRef, "areaRef");
  return {
    version: CASE_VERSION,
    name: raw.name,
    nx,
    ny,
    nz,
    lx: positiveFinite(raw.lx, "lx"),
    ly: positiveFinite(raw.ly, "ly"),
    lz: positiveFinite(raw.lz, "lz"),
    rho: positiveFinite(raw.rho, "rho"),
    nu: nonnegativeFinite(raw.nu, "nu"),
    pressure: "kinematic",
    force: vec3(raw.force, "force"),
    gravity: vec3(raw.gravity, "gravity"),
    beta: finiteNumber(raw.beta, "beta"),
    tRef: finiteNumber(raw.tRef, "tRef"),
    alpha: nonnegativeFinite(raw.alpha, "alpha"),
    dyeDiffusivity: nonnegativeFinite(raw.dyeDiffusivity, "dyeDiffusivity"),
    dyeCouplesDensity: false,
    boundaries,
    uRef: finiteNumber(raw.uRef, "uRef"),
    lRef: positiveFinite(raw.lRef, "lRef"),
    areaRef,
    cfl,
    diffCfl,
    poisson: {
      atol: positiveFinite(raw.poisson.atol, "poisson.atol"),
      rtol: positiveFinite(raw.poisson.rtol, "poisson.rtol"),
      maxIterations: positiveInt(raw.poisson.maxIterations, "poisson.maxIterations"),
    },
    advection: raw.advection,
  };
}

export function wall(u = 0, v = 0, w = 0, temperature: number | null = null): FaceBoundary {
  return { kind: "wall", u, v, w, phi: 0, backflow: "clamp", temperature };
}

export function lid(u: number, v = 0, w = 0): FaceBoundary {
  return { kind: "lid", u, v, w, phi: 0, backflow: "clamp", temperature: null };
}

export function inlet(u: number, v = 0, w = 0, temperature: number | null = null): FaceBoundary {
  return { kind: "inlet", u, v, w, phi: 0, backflow: "clamp", temperature };
}

export function outlet(phi = 0): FaceBoundary {
  return { kind: "outlet", u: 0, v: 0, w: 0, phi, backflow: "clamp", temperature: null };
}

export function periodic(): FaceBoundary {
  return { kind: "periodic", u: 0, v: 0, w: 0, phi: 0, backflow: "allow", temperature: null };
}

export function emptyBoundaries(): CaseSpec["boundaries"] {
  return {
    west: wall(),
    east: wall(),
    south: wall(),
    north: wall(),
    back: wall(),
    front: wall(),
  };
}

const ZERO: Vec3 = { x: 0, y: 0, z: 0 };

export function makeCase(partial: {
  name: string;
  nx: number;
  ny: number;
  nz: number;
  lx: number;
  ly: number;
  lz: number;
  nu: number;
  boundaries: CaseSpec["boundaries"];
  rho?: number;
  force?: Vec3;
  gravity?: Vec3;
  beta?: number;
  tRef?: number;
  alpha?: number;
  uRef?: number;
  lRef?: number;
  areaRef?: number | null;
  advection?: CaseSpec["advection"];
  poisson?: CaseSpec["poisson"];
}): CaseSpec {
  return validateCase({
    version: CASE_VERSION,
    name: partial.name,
    nx: partial.nx,
    ny: partial.ny,
    nz: partial.nz,
    lx: partial.lx,
    ly: partial.ly,
    lz: partial.lz,
    rho: partial.rho ?? 1,
    nu: partial.nu,
    pressure: "kinematic",
    force: partial.force ?? ZERO,
    gravity: partial.gravity ?? ZERO,
    beta: partial.beta ?? 0,
    tRef: partial.tRef ?? 0,
    alpha: partial.alpha ?? 0,
    dyeDiffusivity: 0,
    dyeCouplesDensity: false,
    boundaries: partial.boundaries,
    uRef: partial.uRef ?? 1,
    lRef: partial.lRef ?? partial.ly,
    areaRef: partial.areaRef === undefined ? null : partial.areaRef,
    cfl: 0.45,
    diffCfl: 0.85,
    poisson: partial.poisson ?? { atol: 1e-10, rtol: 1e-8, maxIterations: 4000 },
    advection: partial.advection ?? "none",
  });
}

export function caseToJson(spec: CaseSpec): string {
  return JSON.stringify(spec);
}

export function caseFromJson(text: string): CaseSpec {
  let raw: unknown;
  try {
    raw = JSON.parse(text) as unknown;
  } catch {
    throw new CaseError("case JSON is not parseable");
  }
  return validateCase(raw);
}
