// Screening models for analysis types that are not the checked incompressible path.
// Each formula is defined and tested. None is a field solver for that physics.

import { latticeRun } from "./lattice.ts";

export type ExpInput = {
  re: number;
  uIn: number;
  maxSpeed: number;
  tIn: number;
  tWall: number;
  young: number; // GPa
  vonMises: number;
  meanPressure: number;
};

export type ExpResult = {
  title: string;
  value: string;
  detail: string;
};

/** Constants for the screening models. Units are SI unless noted. */
export const SCREEN = {
  sound: 340,
  gamma: 1.4,
  machLimit: 0.3,
  kSolid: 16,
  thickness: 0.01,
  hFluid: 10,
  alpha: 1.2e-5,
  area: 1e-4,
  length: 1,
  mass: 1,
  strainLimit: 0.01,
} as const;

export function isentropicDensityRatio(mach: number): number {
  const m2 = mach * mach;
  return (1 + 0.5 * (SCREEN.gamma - 1) * m2) ** (-1 / (SCREEN.gamma - 1));
}

export function activityClass(speed: number): "sitting" | "standing" | "walking" | "uncomfortable" {
  if (speed < 4) return "sitting";
  if (speed < 6) return "standing";
  if (speed < 8) return "walking";
  return "uncomfortable";
}

export function conjugateBalance(tIn: number, tWall: number): { q: number; tInterface: number } {
  const rFluid = 1 / SCREEN.hFluid;
  const rSolid = SCREEN.thickness / SCREEN.kSolid;
  const q = (tWall - tIn) / (rFluid + rSolid);
  return { q, tInterface: tIn + q * rFluid };
}

export function conductionFlux(tIn: number, tWall: number): number {
  return (SCREEN.kSolid / SCREEN.thickness) * (tWall - tIn);
}

export function sectionStiffness(youngGpa: number): number {
  return Math.max(0, youngGpa) * 1e9 * (SCREEN.area / SCREEN.length);
}

export function firstFrequency(youngGpa: number): number {
  const k = sectionStiffness(youngGpa);
  return Math.sqrt(k / SCREEN.mass) / (2 * Math.PI);
}

export function harmonicGain(youngGpa: number, hz = 1): number {
  const k = sectionStiffness(youngGpa);
  const omega = 2 * Math.PI * hz;
  const denom = Math.abs(k - SCREEN.mass * omega * omega);
  return denom > 0 ? 1 / denom : Number.POSITIVE_INFINITY;
}

export function elasticStrain(vonMises: number, youngGpa: number): number {
  const e = youngGpa * 1e9;
  return e > 0 ? vonMises / e : 0;
}

const A_SOUND = SCREEN.sound;

export function runExperimental(id: string, p: ExpInput): ExpResult {
  switch (id) {
    case "lbm":
      return lbm(p);
    case "multipurpose":
      return multipurpose(p);
    case "comfort":
      return comfort(p);
    case "compressible":
      return compressible(p);
    case "conjugate":
      return conjugate(p);
    case "multiphase":
      return multiphase();
    case "dynamic":
      return dynamic(p);
    case "heatSolid":
      return heatSolid(p);
    case "thermomech":
      return thermomech(p);
    case "harmonic":
      return harmonic(p);
    case "frequency":
      return frequency(p);
    case "nonlinear":
      return nonlinear(p);
    default:
      return { title: "Incompressible", value: "checked path", detail: "No experimental model." };
  }
}

function lbm(_p: ExpInput): ExpResult {
  const tau = 0.8;
  const ux0 = 0.05;
  const run = latticeRun(16, 16, 8, tau, ux0);
  const nu = (1 / 3) * (tau - 0.5);
  const densityError = Math.abs(run.meanDensity - 1);
  const momentumError = Math.abs(run.meanUx - ux0);
  return {
    title: "LBM step",
    value: `ρ̄ = ${run.meanDensity.toFixed(6)} · ūx = ${run.meanUx.toFixed(4)}`,
    detail: `Periodic D2Q9, ${run.steps} steps, τ = ${tau}, ν_lb = ${nu.toExponential(2)}. Density error ${densityError.toExponential(1)}, momentum error ${momentumError.toExponential(1)}. Not coupled to the volume and not a 3D lattice.`,
  };
}

function multipurpose(p: ExpInput): ExpResult {
  return {
    title: "Multi-purpose",
    value: `Re ${p.re.toFixed(0)} · |u|max ${p.maxSpeed.toFixed(2)}`,
    detail: "Same incompressible equations. Heat and the section stay available in Suite. No extra coupling is solved.",
  };
}

function comfort(p: ExpInput): ExpResult {
  const cls = activityClass(p.maxSpeed);
  return {
    title: "Wind comfort",
    value: `${cls} · ${p.maxSpeed.toFixed(2)} m/s`,
    detail: "Activity class of the preview max speed at 4, 6, and 8 m/s. No exceedance probability, so this is not a Lawson study.",
  };
}

function compressible(p: ExpInput): ExpResult {
  const mach = Math.abs(p.uIn) / A_SOUND;
  const ratio = isentropicDensityRatio(mach);
  const inside = mach < SCREEN.machLimit;
  return {
    title: "Compressible screen",
    value: `M = ${mach.toExponential(2)} · ρ/ρ0 = ${ratio.toFixed(4)}`,
    detail: inside
      ? "Isentropic density ratio. Mach is under 0.3, so the constant-density path is inside its usual envelope. Not a compressible solve."
      : "Isentropic density ratio. Mach is at or above 0.3, so the constant-density path is outside its usual envelope. Not a compressible solve.",
  };
}

function conjugate(p: ExpInput): ExpResult {
  const { q, tInterface } = conjugateBalance(p.tIn, p.tWall);
  return {
    title: "Conjugate screen",
    value: `T_interface = ${tInterface.toFixed(2)} · q = ${q.toExponential(2)}`,
    detail: `Series resistance, h = ${SCREEN.hFluid} W/m²K, k = ${SCREEN.kSolid} W/mK, L = ${SCREEN.thickness} m. Fluid and solid fluxes match by construction. Not a conjugate field solve.`,
  };
}

function multiphase(): ExpResult {
  return {
    title: "Multiphase",
    value: "single fluid · no phase field",
    detail: "Dye is a passive scalar. It is not a volume fraction, so no mixture density is reported.",
  };
}

function dynamic(p: ExpInput): ExpResult {
  const f = firstFrequency(p.young);
  const period = f > 0 ? 1 / f : Number.POSITIVE_INFINITY;
  return {
    title: "Dynamic screen",
    value: `T = ${period.toExponential(2)} s`,
    detail: `Undamped period of a ${SCREEN.mass} kg mass on k = E A/L with A = ${SCREEN.area} m² and L = ${SCREEN.length} m. Not a dynamic field solve.`,
  };
}

function heatSolid(p: ExpInput): ExpResult {
  const q = conductionFlux(p.tIn, p.tWall);
  return {
    title: "Solid conduction",
    value: `q = ${q.toExponential(2)} W/m²`,
    detail: `Steady flux k ΔT/L with k = ${SCREEN.kSolid} W/mK and L = ${SCREEN.thickness} m. Not a conduction field solve.`,
  };
}

function thermomech(p: ExpInput): ExpResult {
  const dT = p.tWall - p.tIn;
  const epsT = SCREEN.alpha * dT;
  const sigT = p.young * 1e9 * epsT;
  return {
    title: "Thermomechanical screen",
    value: `ε_T = ${epsT.toExponential(2)} · σ_T = ${sigT.toExponential(2)}`,
    detail: `Free thermal strain α ΔT with α = ${SCREEN.alpha} /K, then E ε. Not added to the section stress and not coupled to the flow.`,
  };
}

function harmonic(p: ExpInput): ExpResult {
  const f1 = firstFrequency(p.young);
  const gain = harmonicGain(p.young, 1);
  return {
    title: "Harmonic screen",
    value: `f1 = ${f1.toExponential(2)} Hz · |H(1 Hz)| = ${gain.toExponential(2)}`,
    detail: "Undamped single-degree-of-freedom gain at 1 Hz. No damping and no spatial mode.",
  };
}

function frequency(p: ExpInput): ExpResult {
  const f1 = firstFrequency(p.young);
  return {
    title: "Frequency screen",
    value: `f1 = ${f1.toExponential(2)} Hz`,
    detail: "First frequency of the same single-degree-of-freedom oscillator. Not a modal analysis of the section.",
  };
}

function nonlinear(p: ExpInput): ExpResult {
  const eps = elasticStrain(p.vonMises, p.young);
  const flag = eps > SCREEN.strainLimit;
  return {
    title: "Nonlinear screen",
    value: flag ? `strain ${eps.toExponential(2)} · above 1%` : `strain ${eps.toExponential(2)} · at or under 1%`,
    detail: "Compares σ/E with 1%. Above that, a linear section is outside its usual range. Not a nonlinear solve.",
  };
}
