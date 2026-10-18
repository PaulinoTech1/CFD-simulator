// Experimental models for analysis types that are not the checked incompressible path.
// Each function runs a small calculation. None is a verified solver.

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

const A_SOUND = 340;

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
      return multiphase(p);
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

function lbm(p: ExpInput): ExpResult {
  const cs2 = 1 / 3;
  const tau = 0.5 + (3 * p.re > 0 ? 1 / p.re : 0.1);
  const nu = cs2 * (tau - 0.5);
  const mach = Math.abs(p.uIn) / A_SOUND;
  return {
    title: "LBM estimate",
    value: `τ = ${tau.toFixed(3)} · ν_lb = ${nu.toExponential(2)}`,
    detail: `D3Q19 viscosity from τ. Mach ${mach.toExponential(2)} at the inlet. This is not a lattice Boltzmann solve.`,
  };
}

function multipurpose(p: ExpInput): ExpResult {
  return {
    title: "Multi-purpose",
    value: `Re ${p.re.toFixed(0)} · |u|max ${p.maxSpeed.toFixed(2)}`,
    detail: "Uses the incompressible fields. Heat and the section are available in Suite. Not a coupled multiphysics solve.",
  };
}

function comfort(p: ExpInput): ExpResult {
  const u = p.maxSpeed;
  const cls = u < 1.5 ? "sitting" : u < 3 ? "standing" : u < 5 ? "walking" : "uncomfortable";
  return {
    title: "Wind comfort",
    value: `${cls} · ${u.toFixed(2)} m/s`,
    detail: "Lawson-style class from the current max speed. Not a pedestrian comfort study.",
  };
}

function compressible(p: ExpInput): ExpResult {
  const mach = Math.abs(p.uIn) / A_SOUND;
  const rhoRatio = 1 / (1 + 0.5 * (1.4 - 1) * mach * mach);
  return {
    title: "Compressible estimate",
    value: `M = ${mach.toExponential(2)} · ρ/ρ0 ≈ ${rhoRatio.toFixed(4)}`,
    detail: "Isentropic density ratio at the inlet Mach number. Not a compressible solver.",
  };
}

function conjugate(p: ExpInput): ExpResult {
  const h = 10;
  const kSolid = 1;
  const L = 0.01;
  const rFluid = 1 / h;
  const rSolid = L / kSolid;
  const q = (p.tWall - p.tIn) / (rFluid + rSolid);
  const tInterface = p.tIn + q * rFluid;
  return {
    title: "Conjugate estimate",
    value: `T_interface ≈ ${tInterface.toFixed(2)} · q ≈ ${q.toFixed(2)}`,
    detail: "One-dimensional resistance network. Not conjugate heat transfer.",
  };
}

function multiphase(p: ExpInput): ExpResult {
  const alpha = Math.min(1, Math.max(0, p.maxSpeed / 5));
  const rhoMix = 1000 * (1 - alpha) + 1.2 * alpha;
  return {
    title: "Multiphase estimate",
    value: `α ≈ ${alpha.toFixed(2)} · ρ_mix ≈ ${rhoMix.toFixed(1)}`,
    detail: "Volume fraction stand-in from the speed scale. Not a multiphase solver.",
  };
}

function dynamic(p: ExpInput): ExpResult {
  const k = p.young * 1e9 * 1e-4;
  const m = 1;
  const omega = Math.sqrt(k / m);
  const period = (2 * Math.PI) / omega;
  return {
    title: "Dynamic estimate",
    value: `T ≈ ${period.toExponential(2)} s`,
    detail: "Single-degree-of-freedom period from section stiffness and unit mass. Not a dynamic solve.",
  };
}

function heatSolid(p: ExpInput): ExpResult {
  const k = 16;
  const L = 0.01;
  const q = (k / L) * (p.tWall - p.tIn);
  return {
    title: "Solid conduction",
    value: `q ≈ ${q.toExponential(2)} W/m²`,
    detail: "Steady conduction across 10 mm of steel-like conductivity. Not a conduction solver.",
  };
}

function thermomech(p: ExpInput): ExpResult {
  const alpha = 1.2e-5;
  const dT = p.tWall - p.tIn;
  const epsT = alpha * dT;
  const sigT = p.young * 1e9 * epsT;
  return {
    title: "Thermomechanical estimate",
    value: `ε_T ≈ ${epsT.toExponential(2)} · σ_T ≈ ${sigT.toExponential(2)}`,
    detail: "Free thermal strain at the section. Not coupled to the flow.",
  };
}

function harmonic(p: ExpInput): ExpResult {
  const k = p.young * 1e9 * 1e-4;
  const m = 1;
  const omega0 = Math.sqrt(k / m);
  const omega = 2 * Math.PI * 1;
  const amp = 1 / Math.abs(k - m * omega * omega);
  return {
    title: "Harmonic estimate",
    value: `ω0 ≈ ${omega0.toExponential(2)} · |H(1 Hz)| ≈ ${amp.toExponential(2)}`,
    detail: "Undamped single-degree-of-freedom response at 1 Hz. Not a harmonic solver.",
  };
}

function frequency(p: ExpInput): ExpResult {
  const k = p.young * 1e9 * 1e-4;
  const m = 1;
  const f1 = Math.sqrt(k / m) / (2 * Math.PI);
  return {
    title: "Frequency estimate",
    value: `f1 ≈ ${f1.toExponential(2)} Hz`,
    detail: "First frequency of a unit-mass oscillator with the section stiffness. Not a modal analysis.",
  };
}

function nonlinear(p: ExpInput): ExpResult {
  const eps = p.young > 0 ? p.vonMises / (p.young * 1e9) : 0;
  const flag = eps > 0.01;
  return {
    title: "Nonlinear check",
    value: flag ? `strain ${eps.toExponential(2)} · flagged` : `strain ${eps.toExponential(2)} · linear range`,
    detail: "Geometric nonlinearity flagged above 1% strain. Not a nonlinear solve.",
  };
}
