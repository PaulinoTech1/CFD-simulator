import type { FluidSim } from "./solver";
import type { Fluid3 } from "./solver3d";
import type { Preset3 } from "./presets3d";
import { F_BODY, F_INLET, F_SOLID, type PresetId } from "./types";

export type SkillId = "algebra" | "group" | "topology" | "thermo" | "blackscholes" | "heston";

export type SkillInfo = {
  id: SkillId;
  name: string;
  subject: string;
  claim: string;
  machine: string;
};

export const SKILLS: readonly SkillInfo[] = [
  {
    id: "algebra",
    name: "Abstract algebra",
    subject: "The grid is a vector space",
    claim: "A continuum derivative is replaced by a linear map on R^N. Projection is the statement that velocity is put back in the kernel of divergence.",
    machine: "N = nx·ny numbers. Gradient, divergence, and the Laplacian are matrices. The machine only adds and scales them.",
  },
  {
    id: "group",
    name: "Group theory",
    subject: "Symmetry before it breaks",
    claim: "The empty tunnel is invariant under reflection through mid-height. The cylinder keeps that Z2 symmetry until the street breaks it. The airfoil does not: 8° already picks a sign.",
    machine: "The order parameter is lift. The defect is how far v is from being odd under j → ny−1−j.",
  },
  {
    id: "topology",
    name: "Topology",
    subject: "What the grid cannot untie",
    claim: "A cylinder or an airfoil punches a hole in the plane. Loops around it cannot contract, so circulation is a winding. The plume has no hole.",
    machine: "One component count, and one line integral of velocity around the body. On the plume that loop is contractible.",
  },
  {
    id: "thermo",
    name: "Thermodynamics",
    subject: "Energy has a ledger",
    claim: "Kinetic energy is the internal energy of this model. Viscosity is the irreversible leak. Buoyancy and the inlet are the only other work terms.",
    machine: "KE = ½ Σ |u|². Dissipation is ν times the squared velocity gradients. Inlet flux is the energy arriving through the upstream face.",
  },
  {
    id: "blackscholes",
    name: "Black–Scholes",
    subject: "The same parabolic machine",
    claim: "An option price obeys a linear convection–diffusion equation. After x = log S it is a heat equation with a drift — the dye step, with a different coefficient.",
    machine: "A 1D grid in log-price, explicit in time, checked against the closed-form call.",
  },
  {
    id: "heston",
    name: "Heston",
    subject: "Volatility of volatility",
    claim: "Variance is its own state. ξ is how hard that state is kicked. ξ = 0 collapses the second axis and the price returns to Black–Scholes.",
    machine: "Euler–Maruyama on (S, v), then an average of the discounted payoff. The second Wiener increment is the vol-of-vol.",
  },
];

export type SkillReading = {
  meter: string;
  value: string;
  note: string;
};

export function readSkill(
  sim: FluidSim,
  live: { ke: number; re: number; preset: PresetId },
  id: SkillId,
): SkillReading {
  if (id === "algebra") {
    const div = meanAbsDiv(sim);
    const name =
      live.preset === "karman"
        ? "Kármán street"
        : live.preset === "airfoil"
          ? "NACA section"
          : live.preset === "plume"
            ? "plume"
            : "field";
    return {
      meter: "mean |∇·u| after projection",
      value: div.toExponential(2),
      note: `Dimension N = ${(sim.nx * sim.ny).toLocaleString("en-US")}. On the ${name} this is the linear map that puts velocity back in ker(div). SOR is truncated, so the residual is small rather than zero.`,
    };
  }
  if (id === "group") {
    const defect = reflectionDefect(sim);
    let note = "The meter is |v + σv| under reflection through mid-height.";
    if (live.preset === "karman") {
      note =
        defect < 0.08
          ? "The cylinder is still symmetric. The street has not broken the reflection yet."
          : "The street has left the symmetric subspace. Lift is the sign of that broken Z2.";
    } else if (live.preset === "airfoil") {
      note = "Eight degrees already selects a sign, so this defect is forced by the angle of attack. It is not a spontaneous bifurcation.";
    } else if (live.preset === "plume") {
      note = "The floor picks a direction. A rising plume is not odd in v, so the defect stays large even when the plume is symmetric left to right.";
    }
    return { meter: "reflection defect of v", value: defect.toFixed(3), note };
  }
  if (id === "topology") {
    const loop = circulation(sim);
    let note = "No hole in the domain. The loop can shrink to a point.";
    if (live.preset === "karman") {
      note = "The cylinder makes the plane’s complement not simply connected. This circulation is the winding around it.";
    } else if (live.preset === "airfoil") {
      note = "The section is a hole in the plane. Γ is the bound vortex of the airfoil, the same winding the cylinder carries.";
    } else if (live.preset === "plume") {
      note = "The plume has no body. Γ here is just the buoyant vortex pair, and the loop can slide off.";
    }
    return {
      meter: "circulation Γ",
      value: loop.gamma.toFixed(2),
      note: `${loop.components} solid component${loop.components === 1 ? "" : "s"}. ${note}`,
    };
  }
  const heat = dissipation(sim, live.re);
  return {
    meter: "KE · dissipation · inlet flux",
    value: `${live.ke.toExponential(2)} · ${heat.phi.toExponential(2)} · ${heat.flux.toExponential(2)}`,
    note:
      live.preset === "plume"
        ? "On the plume, buoyancy is the work term and dissipation is the leak. There is no free stream doing the heating."
        : live.preset === "karman" || live.preset === "airfoil"
          ? "Wake dissipation is the irreversible part. Drag is the momentum the body takes out of that ledger."
          : "Dissipation is non-negative. A closed box has no inlet flux, so kinetic energy only grows from the lid or from buoyancy.",
  };
}

function blocked(flag: number): boolean {
  return flag === F_SOLID || flag === F_BODY;
}

function meanAbsDiv(sim: FluidSim): number {
  const { u, v, flags, nx, ny, periodicX } = sim;
  const i0 = periodicX ? 0 : 1;
  const i1 = periodicX ? nx : nx - 1;
  let acc = 0;
  let n = 0;
  for (let j = 1; j < ny - 1; j++) {
    for (let i = i0; i < i1; i++) {
      const id = i + j * nx;
      if (blocked(flags[id]!)) continue;
      const im = periodicX && i === 0 ? nx - 1 : i - 1;
      const ip = periodicX && i === nx - 1 ? 0 : i + 1;
      const d = 0.5 * ((u[ip + j * nx] ?? 0) - (u[im + j * nx] ?? 0) + ((v[id + nx] ?? 0) - (v[id - nx] ?? 0)));
      acc += Math.abs(d);
      n++;
    }
  }
  return n > 0 ? acc / n : 0;
}

function reflectionDefect(sim: FluidSim): number {
  const { v, flags, nx, ny } = sim;
  const mid = ny >> 1;
  let num = 0;
  let den = 0;
  for (let j = 1; j < mid; j += 2) {
    const jm = ny - 1 - j;
    for (let i = 1; i < nx - 1; i += 2) {
      const a = i + j * nx;
      const b = i + jm * nx;
      if (blocked(flags[a]!) || blocked(flags[b]!)) continue;
      const va = v[a]!;
      const vb = v[b]!;
      num += Math.abs(va + vb);
      den += Math.abs(va) + Math.abs(vb);
    }
  }
  return den > 1e-6 ? num / den : 0;
}

function circulation(sim: FluidSim): { gamma: number; components: number } {
  const { u, v, flags, nx, ny } = sim;
  let i0 = nx;
  let i1 = 0;
  let j0 = ny;
  let j1 = 0;
  let components = 0;
  const seen = new Uint8Array(flags.length);
  const stack: number[] = [];
  for (let start = 0; start < flags.length; start++) {
    if (flags[start] !== F_BODY || seen[start]) continue;
    components++;
    stack.push(start);
    seen[start] = 1;
    while (stack.length) {
      const id = stack.pop()!;
      const i = id % nx;
      const j = (id / nx) | 0;
      if (i < i0) i0 = i;
      if (i > i1) i1 = i;
      if (j < j0) j0 = j;
      if (j > j1) j1 = j;
      const nbr = [id - 1, id + 1, id - nx, id + nx];
      for (let s = 0; s < 4; s++) {
        const nb = nbr[s]!;
        if (nb < 0 || nb >= flags.length || seen[nb] || flags[nb] !== F_BODY) continue;
        seen[nb] = 1;
        stack.push(nb);
      }
    }
  }
  const loop = (a: number, b: number, c: number, d: number) => {
    let gamma = 0;
    for (let i = a; i < b; i++) gamma += u[i + c * nx] ?? 0;
    for (let j = c; j < d; j++) gamma += v[b + j * nx] ?? 0;
    for (let i = b; i > a; i--) gamma -= u[i + d * nx] ?? 0;
    for (let j = d; j > c; j--) gamma -= v[a + j * nx] ?? 0;
    return gamma;
  };
  if (components === 0) {
    const a = Math.max(1, (nx * 0.35) | 0);
    const b = Math.min(nx - 2, (nx * 0.65) | 0);
    const c = Math.max(1, (ny * 0.3) | 0);
    const d = Math.min(ny - 2, (ny * 0.7) | 0);
    return { gamma: loop(a, b, c, d), components: 0 };
  }
  return {
    gamma: loop(Math.max(1, i0 - 2), Math.min(nx - 2, i1 + 2), Math.max(1, j0 - 2), Math.min(ny - 2, j1 + 2)),
    components,
  };
}

function dissipation(sim: FluidSim, re: number): { phi: number; flux: number } {
  const { u, v, flags, nx, ny, charLength, uIn } = sim;
  const nu = (Math.max(Math.abs(uIn), 6) * charLength) / Math.max(re, 1);
  let phi = 0;
  let flux = 0;
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const id = i + j * nx;
      const f = flags[id]!;
      if (blocked(f)) continue;
      const dudx = 0.5 * ((u[id + 1] ?? 0) - (u[id - 1] ?? 0));
      const dudy = 0.5 * ((u[id + nx] ?? 0) - (u[id - nx] ?? 0));
      const dvdx = 0.5 * ((v[id + 1] ?? 0) - (v[id - 1] ?? 0));
      const dvdy = 0.5 * ((v[id + nx] ?? 0) - (v[id - nx] ?? 0));
      phi += nu * (dudx * dudx + dudy * dudy + dvdx * dvdx + dvdy * dvdy);
      if (f === F_INLET) flux += 0.5 * ((u[id] ?? 0) ** 2 + (v[id] ?? 0) ** 2) * Math.max(u[id] ?? 0, 0);
    }
  }
  return { phi, flux };
}

export function readVolume(sim: Fluid3, live: { re: number; preset: Preset3 }, id: SkillId): SkillReading {
  if (id === "algebra") {
    return {
      meter: "mean |∇·u|",
      value: volumeDiv(sim).toExponential(2),
      note: `N = ${(sim.nx * sim.ny * sim.nz).toLocaleString("en-US")}. The 3D projection is the same linear map as the street in 2D: subtract a gradient so the velocity lands in ker(div).`,
    };
  }
  if (id === "group") {
    const defect = volumeDefect(sim);
    const note =
      live.preset === "karman"
        ? defect < 0.08
          ? "The spanwise cylinder is still symmetric through mid-height. The street has not broken that Z2 yet."
          : "The street has left the symmetric subspace. Lift is the sign of the broken reflection."
        : live.preset === "airfoil"
          ? "Eight degrees already selects a sign along the whole span. The defect is the angle, not a bifurcation."
          : live.preset === "plume"
            ? "The floor picks up. A rising plume is not odd in v."
            : "The meter is |v + σv| under reflection through mid-height.";
    return { meter: "reflection defect of v", value: defect.toFixed(3), note };
  }
  if (id === "topology") {
    const gamma = volumeGamma(sim);
    const note =
      live.preset === "karman"
        ? "The cylinder meets both spanwise walls, so this mid-span loop cannot contract. Γ is the winding of the street."
        : live.preset === "airfoil"
          ? "The extruded section is a hole through the span. Γ is the bound vortex, the same winding the cylinder carries."
          : "No hole through the span. A loop in the mid-span plane can slip off, so this number is not a topological charge.";
    return { meter: "mid-span circulation Γ", value: gamma.toFixed(2), note };
  }
  const heat = volumeHeat(sim, live.re);
  return {
    meter: "KE · dissipation · inlet flux",
    value: `${sim.ke.toExponential(2)} · ${heat.phi.toExponential(2)} · ${heat.flux.toExponential(2)}`,
    note:
      live.preset === "plume"
        ? "On the plume, buoyancy is the work and dissipation is the leak."
        : "Wake dissipation is the irreversible part. Drag is the momentum the body takes out of the ledger.",
  };
}

function volumeDiv(sim: Fluid3): number {
  const { u, v, w, flags, nx, ny, nz } = sim;
  const nxy = nx * ny;
  let acc = 0;
  let n = 0;
  for (let k = 1; k < nz - 1; k += 2) {
    for (let j = 1; j < ny - 1; j += 2) {
      for (let i = 1; i < nx - 1; i += 2) {
        const id = i + nx * j + nxy * k;
        if (flags[id] === F_SOLID || flags[id] === F_BODY) continue;
        const d =
          0.5 *
          ((u[id + 1] ?? 0) -
            (u[id - 1] ?? 0) +
            ((v[id + nx] ?? 0) - (v[id - nx] ?? 0)) +
            ((w[id + nxy] ?? 0) - (w[id - nxy] ?? 0)));
        acc += Math.abs(d);
        n++;
      }
    }
  }
  return n > 0 ? acc / n : 0;
}

function volumeDefect(sim: Fluid3): number {
  const { v, flags, nx, ny, nz } = sim;
  const nxy = nx * ny;
  const mid = ny >> 1;
  let num = 0;
  let den = 0;
  for (let k = 1; k < nz - 1; k += 2) {
    for (let j = 1; j < mid; j += 2) {
      for (let i = 1; i < nx - 1; i += 2) {
        const a = i + nx * j + nxy * k;
        const b = i + nx * (ny - 1 - j) + nxy * k;
        if (flags[a] === F_BODY || flags[b] === F_BODY) continue;
        num += Math.abs(v[a]! + v[b]!);
        den += Math.abs(v[a]!) + Math.abs(v[b]!);
      }
    }
  }
  return den > 1e-6 ? num / den : 0;
}

function volumeGamma(sim: Fluid3): number {
  const { u, v, flags, nx, ny, nz, cx, cy, radius } = sim;
  const k = (nz / 2) | 0;
  const nxy = nx * ny;
  const r = Math.max(radius + 2, 3);
  const i0 = Math.max(1, Math.round(cx - r));
  const i1 = Math.min(nx - 2, Math.round(cx + r));
  const j0 = Math.max(1, Math.round(cy - r));
  const j1 = Math.min(ny - 2, Math.round(cy + r));
  let gamma = 0;
  for (let i = i0; i < i1; i++) gamma += u[i + nx * j0 + nxy * k] ?? 0;
  for (let j = j0; j < j1; j++) gamma += v[i1 + nx * j + nxy * k] ?? 0;
  for (let i = i1; i > i0; i--) gamma -= u[i + nx * j1 + nxy * k] ?? 0;
  for (let j = j1; j > j0; j--) gamma -= v[i0 + nx * j + nxy * k] ?? 0;
  return gamma;
}

function volumeHeat(sim: Fluid3, re: number): { phi: number; flux: number } {
  const { u, v, w, flags, nx, ny, nz, charLength, uIn } = sim;
  const nu = (Math.max(Math.abs(uIn), 0.35) * charLength) / Math.max(re, 1);
  const nxy = nx * ny;
  let phi = 0;
  let flux = 0;
  for (let k = 1; k < nz - 1; k += 2) {
    for (let j = 1; j < ny - 1; j += 2) {
      for (let i = 1; i < nx - 1; i += 2) {
        const id = i + nx * j + nxy * k;
        const f = flags[id]!;
        if (f === F_SOLID || f === F_BODY) continue;
        const dudx = 0.5 * ((u[id + 1] ?? 0) - (u[id - 1] ?? 0));
        const dvdy = 0.5 * ((v[id + nx] ?? 0) - (v[id - nx] ?? 0));
        const dwdz = 0.5 * ((w[id + nxy] ?? 0) - (w[id - nxy] ?? 0));
        phi += nu * (dudx * dudx + dvdy * dvdy + dwdz * dwdz);
        if (f === F_INLET) {
          const uu = u[id] ?? 0;
          flux += 0.5 * (uu * uu + (v[id] ?? 0) ** 2 + (w[id] ?? 0) ** 2) * Math.max(uu, 0);
        }
      }
    }
  }
  return { phi, flux };
}

function normCdf(x: number): number {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-0.5 * x * x);
  const p = d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

export function bsCall(spot: number, strike: number, t: number, rate: number, sigma: number): number {
  if (t <= 1e-8 || sigma <= 1e-8) return Math.max(spot - strike * Math.exp(-rate * t), 0);
  const vol = sigma * Math.sqrt(t);
  const d1 = (Math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / vol;
  const d2 = d1 - vol;
  return spot * normCdf(d1) - strike * Math.exp(-rate * t) * normCdf(d2);
}

export type HeatSolve = {
  analytic: number;
  numerical: number;
  error: number;
  curveN: Float32Array;
  curveA: Float32Array;
};

/** Explicit heat solve in log-price. Returns the call at spot 100. */
export function solveBlackScholes(sigma: number): HeatSolve {
  const spot = 100;
  const strike = 100;
  const rate = 0.02;
  const maturity = 1;
  const n = 160;
  const xMin = Math.log(strike) - 1.6;
  const xMax = Math.log(strike) + 1.6;
  const dx = (xMax - xMin) / (n - 1);
  const sig2 = Math.max(sigma, 0.02) ** 2;
  const drift = rate - 0.5 * sig2;
  const dt = Math.min(0.001, 0.25 * dx * dx / sig2);
  const steps = Math.max(1, Math.ceil(maturity / dt));
  const h = maturity / steps;
  const c = new Float32Array(n);
  const next = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const s = Math.exp(xMin + i * dx);
    c[i] = Math.max(s - strike, 0);
  }
  const ax = 0.5 * sig2 / (dx * dx);
  const bx = drift / (2 * dx);
  for (let step = 1; step <= steps; step++) {
    const tau = h * step;
    next[0] = 0;
    const sMax = Math.exp(xMax);
    next[n - 1] = sMax - strike * Math.exp(-rate * tau);
    for (let i = 1; i < n - 1; i++) {
      const diff = ax * (c[i + 1]! - 2 * c[i]! + c[i - 1]!);
      const adv = bx * (c[i + 1]! - c[i - 1]!);
      next[i] = c[i]! + h * (diff + adv - rate * c[i]!);
      if (next[i]! < 0) next[i] = 0;
    }
    c.set(next);
  }
  const curveN = new Float32Array(n);
  const curveA = new Float32Array(n);
  let numerical = 0;
  let best = 1e9;
  for (let i = 0; i < n; i++) {
    const s = Math.exp(xMin + i * dx);
    curveN[i] = c[i]!;
    curveA[i] = bsCall(s, strike, maturity, rate, sigma);
    const dist = Math.abs(s - spot);
    if (dist < best) {
      best = dist;
      numerical = c[i]!;
    }
  }
  const analytic = bsCall(spot, strike, maturity, rate, sigma);
  return { analytic, numerical, error: numerical - analytic, curveN, curveA };
}

function mulberry(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function gauss(rng: () => number): number {
  const u = Math.max(1e-9, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export type HestonSolve = {
  heston: number;
  blackScholes: number;
  gap: number;
  variance: Float32Array;
};

/** Full-truncation Euler on the Heston SDE. ξ is volatility of volatility. */
export function solveHeston(xi: number): HestonSolve {
  const spot = 100;
  const strike = 100;
  const rate = 0.02;
  const maturity = 1;
  const kappa = 2;
  const theta = 0.04;
  const v0 = 0.04;
  const rho = -0.6;
  const paths = 900;
  const steps = 32;
  const dt = maturity / steps;
  const rng = mulberry(7);
  let payoff = 0;
  const variance = new Float32Array(steps + 1);
  variance[0] = v0;
  for (let p = 0; p < paths; p++) {
    let s = spot;
    let vol = v0;
    for (let n = 0; n < steps; n++) {
      const z1 = gauss(rng);
      const z2 = rho * z1 + Math.sqrt(Math.max(0, 1 - rho * rho)) * gauss(rng);
      const root = Math.sqrt(Math.max(vol, 0) * dt);
      vol = Math.max(0, vol + kappa * (theta - Math.max(vol, 0)) * dt + xi * root * z2);
      s = s * Math.exp((rate - 0.5 * Math.max(vol, 0)) * dt + root * z1);
      if (p === 0) variance[n + 1] = vol;
    }
    payoff += Math.max(s - strike, 0);
  }
  const heston = Math.exp(-rate * maturity) * (payoff / paths);
  const blackScholes = bsCall(spot, strike, maturity, rate, Math.sqrt(theta));
  return { heston, blackScholes, gap: heston - blackScholes, variance };
}
