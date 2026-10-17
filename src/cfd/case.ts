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
  /** Scalar Dirichlet. Null means adiabatic (zero normal derivative). */
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
  /** "none" is for viscous and operator tests. muscl-rk2 is the verified transport. */
  advection: "none" | "upwind1" | "muscl-rk2";
};

export type BoundaryName = keyof CaseSpec["boundaries"];

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
  return {
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
  };
}

export function caseToJson(spec: CaseSpec): string {
  return JSON.stringify(spec);
}

export function caseFromJson(text: string): CaseSpec {
  const raw = JSON.parse(text) as CaseSpec;
  if (raw.version !== CASE_VERSION) throw new Error(`unsupported case version ${String(raw.version)}`);
  if (raw.pressure !== "kinematic") throw new Error("only kinematic pressure is defined");
  if (raw.dyeCouplesDensity !== false) throw new Error("dye is passive in this case version");
  if (!(raw.nu >= 0) || !(raw.rho > 0)) throw new Error("density and viscosity must be physical");
  return raw;
}
