import {
  type BoundaryName,
  type CaseSpec,
  CaseError,
  type FaceBoundary,
  caseFromJson,
  caseToJson,
  validateCase,
} from "./case.ts";
import { dissipationDensity, type StrainSample } from "./diagnostics.ts";

export type StepStatus = "ok" | "pressure-not-converged" | "nonfinite" | "rejected";

export type BoundaryFlux = Record<BoundaryName, number>;

export type StepReport = {
  status: StepStatus;
  time: number;
  /** Stable substep size. Zero when the call is rejected before a substep. */
  dt: number;
  /** Interval the caller asked to advance. */
  attempted: number;
  /** Interval actually committed. Zero when the call rolls back. */
  accepted: number;
  /** Substeps still present in the state. Zero after a rollback. */
  substeps: number;
  kineticEnergy: number;
  divL2: number;
  divLinf: number;
  massImbalance: number;
  fluxes: BoundaryFlux;
  pressureIterations: number;
  pressureResidual: number;
  pressureConverged: boolean;
  dissipation: number;
  maxSpeed: number;
  detail: string;
};

type Component = { cells: number[]; pin: number; closed: boolean };

const SIDES: BoundaryName[] = ["west", "east", "south", "north", "back", "front"];

/**
 * Staggered MAC reference for constant-density Newtonian flow.
 * φ = p/ρ is the only pressure that is stored.
 * The discrete Laplacian is div(grad) with the same face sets used to correct velocity.
 * Transport, when enabled, is flux-form. muscl-rk2 is limited linear reconstruction and Heun's method.
 * Spatial design order is 2 away from limiters; the limiter drops that locally to 1.
 */
export class MacSolver {
  readonly spec: CaseSpec;
  readonly dx: number;
  readonly dy: number;
  readonly dz: number;
  readonly u: Float64Array;
  readonly v: Float64Array;
  readonly w: Float64Array;
  readonly phi: Float64Array;
  readonly temperature: Float64Array;
  readonly dye: Float64Array;
  readonly solid: Uint8Array;
  inletDye = 1;
  time = 0;
  steps = 0;
  /**
   * Verification hook. When set to a substep index, that substep fails before
   * it updates the fields and the whole requested advance rolls back.
   * Not a physical model.
   */
  failAtSubstep: number | null = null;
  last: StepReport;

  private readonly nuX: number;
  private readonly nU: number;
  private readonly nV: number;
  private readonly nW: number;
  private readonly nC: number;
  private readonly uT: Float64Array;
  private readonly vT: Float64Array;
  private readonly wT: Float64Array;
  private readonly phiT: Float64Array;
  private readonly dyeT: Float64Array;
  private readonly tempT: Float64Array;
  private readonly u0: Float64Array;
  private readonly v0: Float64Array;
  private readonly w0: Float64Array;
  private readonly phi0: Float64Array;
  private readonly dye0: Float64Array;
  private readonly temp0: Float64Array;
  private readonly ru: Float64Array;
  private readonly rv: Float64Array;
  private readonly rw: Float64Array;
  private readonly su: Float64Array;
  private readonly sv: Float64Array;
  private readonly sw: Float64Array;
  private readonly div: Float64Array;
  private readonly rhs: Float64Array;
  private readonly x: Float64Array;
  private readonly r: Float64Array;
  private readonly pvec: Float64Array;
  private readonly ap: Float64Array;
  private readonly z: Float64Array;
  private freeU: Uint8Array;
  private freeV: Uint8Array;
  private freeW: Uint8Array;
  private components: Component[] = [];
  private unknown: number[] = [];
  private time0 = 0;
  private steps0 = 0;
  private readonly heldU: number[] = [];
  private readonly heldV: number[] = [];
  private readonly heldW: number[] = [];

  constructor(spec: CaseSpec, solid?: Uint8Array) {
    this.spec = validateCase(spec);
    const { nx, ny, nz } = spec;
    this.dx = spec.lx / nx;
    this.dy = spec.ly / ny;
    this.dz = spec.lz / nz;
    this.nuX = nx + 1;
    this.nU = this.nuX * ny * nz;
    this.nV = nx * (ny + 1) * nz;
    this.nW = nx * ny * (nz + 1);
    this.nC = nx * ny * nz;
    this.u = new Float64Array(this.nU);
    this.v = new Float64Array(this.nV);
    this.w = new Float64Array(this.nW);
    this.phi = new Float64Array(this.nC);
    this.temperature = new Float64Array(this.nC);
    this.dye = new Float64Array(this.nC);
    this.temperature.fill(spec.tRef);
    this.solid = solid ? Uint8Array.from(solid) : new Uint8Array(this.nC);
    if (this.solid.length !== this.nC) throw new Error("solid mask does not match the cell count");
    this.uT = new Float64Array(this.nU);
    this.vT = new Float64Array(this.nV);
    this.wT = new Float64Array(this.nW);
    this.phiT = new Float64Array(this.nC);
    this.dyeT = new Float64Array(this.nC);
    this.tempT = new Float64Array(this.nC);
    this.u0 = new Float64Array(this.nU);
    this.v0 = new Float64Array(this.nV);
    this.w0 = new Float64Array(this.nW);
    this.phi0 = new Float64Array(this.nC);
    this.dye0 = new Float64Array(this.nC);
    this.temp0 = new Float64Array(this.nC);
    this.ru = new Float64Array(this.nU);
    this.rv = new Float64Array(this.nV);
    this.rw = new Float64Array(this.nW);
    this.su = new Float64Array(this.nU);
    this.sv = new Float64Array(this.nV);
    this.sw = new Float64Array(this.nW);
    this.div = new Float64Array(this.nC);
    this.rhs = new Float64Array(this.nC);
    this.x = new Float64Array(this.nC);
    this.r = new Float64Array(this.nC);
    this.pvec = new Float64Array(this.nC);
    this.ap = new Float64Array(this.nC);
    this.z = new Float64Array(this.nC);
    this.freeU = new Uint8Array(this.nU);
    this.freeV = new Uint8Array(this.nV);
    this.freeW = new Uint8Array(this.nW);
    this.rebuildFaces();
    this.applyVelocityBC();
    this.last = this.blank("ok", 0, 0, "initialized");
  }

  iu(i: number, j: number, k: number): number {
    return i + this.nuX * (j + this.spec.ny * k);
  }
  iv(i: number, j: number, k: number): number {
    return i + this.spec.nx * (j + (this.spec.ny + 1) * k);
  }
  iw(i: number, j: number, k: number): number {
    return i + this.spec.nx * (j + this.spec.ny * k);
  }
  ic(i: number, j: number, k: number): number {
    return i + this.spec.nx * (j + this.spec.ny * k);
  }

  setSolid(mask: Uint8Array): void {
    if (mask.length !== this.nC) throw new Error("solid mask does not match the cell count");
    this.solid.set(mask);
    this.rebuildFaces();
    this.applyVelocityBC();
  }

  maxWaveSpeed(): number {
    const { dx, dy, dz } = this;
    let m = 0;
    for (let i = 0; i < this.u.length; i++) m = Math.max(m, Math.abs(this.u[i]!) / dx);
    for (let i = 0; i < this.v.length; i++) m = Math.max(m, Math.abs(this.v[i]!) / dy);
    for (let i = 0; i < this.w.length; i++) m = Math.max(m, Math.abs(this.w[i]!) / dz);
    return m;
  }

  /** Largest stable step for the active advection and explicit diffusion schemes. */
  stableDt(): number {
    const { nu, alpha, cfl, diffCfl } = this.spec;
    const wave = this.maxWaveSpeed();
    const adv = this.spec.advection === "none" || wave === 0 ? Number.POSITIVE_INFINITY : cfl / wave;
    const invH = 2 / (this.dx * this.dx) + 2 / (this.dy * this.dy) + 2 / (this.dz * this.dz);
    const visc = nu > 0 ? diffCfl / (nu * invH) : Number.POSITIVE_INFINITY;
    const heat = alpha > 0 ? diffCfl / (alpha * invH) : Number.POSITIVE_INFINITY;
    return Math.min(adv, visc, heat);
  }

  step(dtRequested: number): StepReport {
    if (!(dtRequested > 0) || !Number.isFinite(dtRequested)) {
      this.last = this.report("rejected", 0, dtRequested, 0, 0, 0, 0, false, "timestep must be a positive finite number");
      return this.last;
    }
    if (!this.fieldsFinite()) {
      this.last = this.report("nonfinite", 0, dtRequested, 0, 0, 0, 0, false, "refusing to advance a nonfinite field");
      return this.last;
    }
    this.save();
    this.applyVelocityBC();
    const limit = this.stableDt();
    const dt = Math.min(dtRequested, limit);
    if (!(dt > 0) || !Number.isFinite(dt)) {
      this.restore();
      this.last = this.report("rejected", 0, dtRequested, 0, 0, 0, 0, false, "stable timestep is not positive");
      return this.last;
    }
    const substeps = Math.max(1, Math.ceil(dtRequested / dt - 1e-12));
    let pressureIterations = 0;
    let pressureResidual = 0;
    let accepted = 0;
    for (let s = 0; s < substeps; s++) {
      if (this.failAtSubstep === s) {
        this.restore();
        this.last = this.report(
          "rejected",
          0,
          dtRequested,
          0,
          0,
          pressureIterations,
          pressureResidual,
          false,
          `injected failure before substep ${s}; rolled back ${s} completed substep(s)`,
        );
        return this.last;
      }
      const h = s === substeps - 1 ? dtRequested - dt * (substeps - 1) : dt;
      const used = Math.min(h, dt);
      this.addForces(used);
      if (this.spec.advection !== "none") this.advectMomentum(used);
      this.diffuseVelocity(used);
      const projected = this.project(used);
      pressureIterations += projected.iterations;
      pressureResidual = projected.residual;
      this.applyVelocityBC();
      const clamped = this.holdBackflow();
      if (!projected.converged || !this.fieldsFinite()) {
        this.restore();
        const status: StepStatus = !this.fieldsFinite() ? "nonfinite" : "pressure-not-converged";
        this.last = this.report(status, 0, dtRequested, 0, 0, pressureIterations, pressureResidual, false, projected.detail);
        return this.last;
      }
      if (clamped > 0) {
        const defect = this.volumeFlux();
        if (Math.abs(defect) > this.fluxTolerance()) {
          this.restore();
          this.last = this.report(
            "rejected",
            0,
            dtRequested,
            0,
            0,
            pressureIterations,
            pressureResidual,
            true,
            `outlet clamp would leave volume flux ${defect.toExponential(2)}; step refused`,
          );
          return this.last;
        }
        const again = this.project(used);
        pressureIterations += again.iterations;
        pressureResidual = again.residual;
        this.releaseHeld();
        this.applyVelocityBC();
        if (!again.converged || !this.fieldsFinite()) {
          this.restore();
          const status: StepStatus = !this.fieldsFinite() ? "nonfinite" : "pressure-not-converged";
          this.last = this.report(status, 0, dtRequested, 0, 0, pressureIterations, pressureResidual, false, again.detail);
          return this.last;
        }
      }
      const defect = this.volumeFlux();
      const div = this.divergenceNorms().linf;
      if (Math.abs(defect) > this.fluxTolerance() || div > 1e-4) {
        this.restore();
        this.last = this.report(
          "rejected",
          0,
          dtRequested,
          0,
          0,
          pressureIterations,
          pressureResidual,
          projected.converged,
          `projection left divergence ${div.toExponential(2)} and volume flux ${defect.toExponential(2)}`,
        );
        return this.last;
      }
      this.advanceScalars(used);
      if (!this.fieldsFinite()) {
        this.restore();
        this.last = this.report(
          "nonfinite",
          0,
          dtRequested,
          0,
          0,
          pressureIterations,
          pressureResidual,
          false,
          "scalar transport became nonfinite",
        );
        return this.last;
      }
      this.time += used;
      this.steps += 1;
      accepted += used;
    }
    this.last = this.report("ok", dt, dtRequested, accepted, substeps, pressureIterations, pressureResidual, true, "accepted");
    return this.last;
  }

  kineticEnergy(): number {
    const { rho, nx, ny, nz } = this.spec;
    const { dx, dy, dz } = this;
    const vol = dx * dy * dz;
    let ke = 0;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          if (this.solid[this.ic(i, j, k)] === 1) continue;
          const uu = 0.5 * (this.u[this.iu(i, j, k)]! ** 2 + this.u[this.iu(i + 1, j, k)]! ** 2);
          const vv = 0.5 * (this.v[this.iv(i, j, k)]! ** 2 + this.v[this.iv(i, j + 1, k)]! ** 2);
          const ww = 0.5 * (this.w[this.iw(i, j, k)]! ** 2 + this.w[this.iw(i, j, k + 1)]! ** 2);
          ke += 0.5 * rho * (uu + vv + ww) * vol;
        }
      }
    }
    return ke;
  }

  divergenceNorms(): { l2: number; linf: number } {
    this.writeDivergence(this.div);
    let s = 0;
    let n = 0;
    let linf = 0;
    for (let c = 0; c < this.nC; c++) {
      if (this.solid[c] === 1) continue;
      const d = this.div[c]!;
      s += d * d;
      n += 1;
      linf = Math.max(linf, Math.abs(d));
    }
    return { l2: n > 0 ? Math.sqrt(s / n) : 0, linf };
  }

  /** Independent of the pressure matrix: face gradient of the current φ, written over free faces. */
  writeGradient(scale: number, targetU: Float64Array, targetV: Float64Array, targetW: Float64Array): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    const { dx, dy, dz } = this;
    targetU.fill(0);
    targetV.fill(0);
    targetW.fill(0);
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i <= nx; i++) {
          const id = this.iu(i, j, k);
          if (this.freeU[id] !== 1) continue;
          const left =
            i === 0
              ? b.west.kind === "periodic"
                ? this.phi[this.ic(nx - 1, j, k)]!
                : b.west.phi
              : this.phi[this.ic(i - 1, j, k)]!;
          const right =
            i === nx
              ? b.east.kind === "periodic"
                ? this.phi[this.ic(0, j, k)]!
                : b.east.phi
              : this.phi[this.ic(i, j, k)]!;
          targetU[id] = (scale * (right - left)) / dx;
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j <= ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iv(i, j, k);
          if (this.freeV[id] !== 1) continue;
          const south =
            j === 0
              ? b.south.kind === "periodic"
                ? this.phi[this.ic(i, ny - 1, k)]!
                : b.south.phi
              : this.phi[this.ic(i, j - 1, k)]!;
          const north =
            j === ny
              ? b.north.kind === "periodic"
                ? this.phi[this.ic(i, 0, k)]!
                : b.north.phi
              : this.phi[this.ic(i, j, k)]!;
          targetV[id] = (scale * (north - south)) / dy;
        }
      }
    }
    for (let k = 0; k <= nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iw(i, j, k);
          if (this.freeW[id] !== 1) continue;
          const back =
            k === 0
              ? b.back.kind === "periodic"
                ? this.phi[this.ic(i, j, nz - 1)]!
                : b.back.phi
              : this.phi[this.ic(i, j, k - 1)]!;
          const front =
            k === nz
              ? b.front.kind === "periodic"
                ? this.phi[this.ic(i, j, 0)]!
                : b.front.phi
              : this.phi[this.ic(i, j, k)]!;
          targetW[id] = (scale * (front - back)) / dz;
        }
      }
    }
    this.syncCopies(targetU, targetV, targetW);
  }

  bodyForce(): { pressure: { x: number; y: number; z: number }; viscous: { x: number; y: number; z: number } } {
    const { rho, nu, nx, ny, nz } = this.spec;
    const { dx, dy, dz } = this;
    const mu = rho * nu;
    const pressure = { x: 0, y: 0, z: 0 };
    const viscous = { x: 0, y: 0, z: 0 };
    const areaX = dy * dz;
    const areaY = dx * dz;
    const areaZ = dx * dy;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          if (this.solid[this.ic(i, j, k)] !== 1) continue;
          if (i > 0 && this.solid[this.ic(i - 1, j, k)] !== 1) {
            const s = this.strainAtFace("x", i, j, k);
            pressure.x += rho * this.phi[this.ic(i - 1, j, k)]! * areaX;
            viscous.x += -2 * mu * s.sxx * areaX;
            viscous.y += -2 * mu * s.sxy * areaX;
            viscous.z += -2 * mu * s.sxz * areaX;
          }
          if (i + 1 < nx && this.solid[this.ic(i + 1, j, k)] !== 1) {
            const s = this.strainAtFace("x", i + 1, j, k);
            pressure.x -= rho * this.phi[this.ic(i + 1, j, k)]! * areaX;
            viscous.x += 2 * mu * s.sxx * areaX;
            viscous.y += 2 * mu * s.sxy * areaX;
            viscous.z += 2 * mu * s.sxz * areaX;
          }
          if (j > 0 && this.solid[this.ic(i, j - 1, k)] !== 1) {
            const s = this.strainAtFace("y", i, j, k);
            pressure.y += rho * this.phi[this.ic(i, j - 1, k)]! * areaY;
            viscous.x += -2 * mu * s.sxy * areaY;
            viscous.y += -2 * mu * s.syy * areaY;
            viscous.z += -2 * mu * s.syz * areaY;
          }
          if (j + 1 < ny && this.solid[this.ic(i, j + 1, k)] !== 1) {
            const s = this.strainAtFace("y", i, j + 1, k);
            pressure.y -= rho * this.phi[this.ic(i, j + 1, k)]! * areaY;
            viscous.x += 2 * mu * s.sxy * areaY;
            viscous.y += 2 * mu * s.syy * areaY;
            viscous.z += 2 * mu * s.syz * areaY;
          }
          if (k > 0 && this.solid[this.ic(i, j, k - 1)] !== 1) {
            const s = this.strainAtFace("z", i, j, k);
            pressure.z += rho * this.phi[this.ic(i, j, k - 1)]! * areaZ;
            viscous.x += -2 * mu * s.sxz * areaZ;
            viscous.y += -2 * mu * s.syz * areaZ;
            viscous.z += -2 * mu * s.szz * areaZ;
          }
          if (k + 1 < nz && this.solid[this.ic(i, j, k + 1)] !== 1) {
            const s = this.strainAtFace("z", i, j, k + 1);
            pressure.z -= rho * this.phi[this.ic(i, j, k + 1)]! * areaZ;
            viscous.x += 2 * mu * s.sxz * areaZ;
            viscous.y += 2 * mu * s.syz * areaZ;
            viscous.z += 2 * mu * s.szz * areaZ;
          }
        }
      }
    }
    return { pressure, viscous };
  }

  /** Traction the fluid exerts on a no-slip wall, in Newtons. Couette: south.x = μ (U/H) Area. */
  wallShear(): BoundaryFlux {
    const { rho, nu, nx, ny, nz, boundaries: b } = this.spec;
    const mu = rho * nu;
    const out: BoundaryFlux = { west: 0, east: 0, south: 0, north: 0, back: 0, front: 0 };
    if (b.south.kind === "wall" || b.south.kind === "lid") {
      let f = 0;
      for (let k = 0; k < nz; k++) {
        for (let i = 0; i < nx; i++) {
          if (this.solid[this.ic(i, 0, k)] === 1) continue;
          const dudy = (2 * (this.u[this.iu(i, 0, k)]! - b.south.u)) / this.dy;
          f += mu * dudy * this.dx * this.dz;
        }
      }
      out.south = f;
    }
    if (b.north.kind === "wall" || b.north.kind === "lid") {
      let f = 0;
      for (let k = 0; k < nz; k++) {
        for (let i = 0; i < nx; i++) {
          if (this.solid[this.ic(i, ny - 1, k)] === 1) continue;
          const dudy = (2 * (b.north.u - this.u[this.iu(i, ny - 1, k)]!)) / this.dy;
          f += mu * dudy * this.dx * this.dz;
        }
      }
      out.north = f;
    }
    return out;
  }

  dissipationPower(): number {
    const { rho, nu, nx, ny, nz } = this.spec;
    const vol = this.dx * this.dy * this.dz;
    let p = 0;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          if (this.solid[this.ic(i, j, k)] === 1) continue;
          p += rho * dissipationDensity(nu, this.strainAtCell(i, j, k)) * vol;
        }
      }
    }
    return p;
  }

  curlAt(i: number, j: number, k: number): { x: number; y: number; z: number } {
    const dvdz = this.dVdz(i, j, k);
    const dwdy = this.dWdy(i, j, k);
    const dudz = this.dUdz(i, j, k);
    const dwdx = this.dWdx(i, j, k);
    const dudy = this.dUdy(i, j, k);
    const dvdx = this.dVdx(i, j, k);
    return { x: dwdy - dvdz, y: dudz - dwdx, z: dvdx - dudy };
  }

  outwardFluxes(): BoundaryFlux {
    const { rho, nx, ny, nz, boundaries: b } = this.spec;
    const ax = this.dy * this.dz;
    const ay = this.dx * this.dz;
    const az = this.dx * this.dy;
    const flux: BoundaryFlux = { west: 0, east: 0, south: 0, north: 0, back: 0, front: 0 };
    if (b.west.kind !== "periodic") {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) flux.west += -rho * this.u[this.iu(0, j, k)]! * ax;
    }
    if (b.east.kind !== "periodic") {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) flux.east += rho * this.u[this.iu(nx, j, k)]! * ax;
    }
    if (b.south.kind !== "periodic") {
      for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) flux.south += -rho * this.v[this.iv(i, 0, k)]! * ay;
    }
    if (b.north.kind !== "periodic") {
      for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) flux.north += rho * this.v[this.iv(i, ny, k)]! * ay;
    }
    if (b.back.kind !== "periodic") {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) flux.back += -rho * this.w[this.iw(i, j, 0)]! * az;
    }
    if (b.front.kind !== "periodic") {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) flux.front += rho * this.w[this.iw(i, j, nz)]! * az;
    }
    return flux;
  }

  cellVelocity(i: number, j: number, k: number): { u: number; v: number; w: number } {
    return {
      u: 0.5 * (this.u[this.iu(i, j, k)]! + this.u[this.iu(i + 1, j, k)]!),
      v: 0.5 * (this.v[this.iv(i, j, k)]! + this.v[this.iv(i, j + 1, k)]!),
      w: 0.5 * (this.w[this.iw(i, j, k)]! + this.w[this.iw(i, j, k + 1)]!),
    };
  }

  toCaseJson(): string {
    return caseToJson(this.spec);
  }

  static fromCaseJson(text: string, solid?: Uint8Array): MacSolver {
    return new MacSolver(caseFromJson(text), solid);
  }

  /** VTK structured-points, cell-centered velocity and kinematic pressure. */
  toVtk(): string {
    const { nx, ny, nz, name } = this.spec;
    const lines = [
      "# vtk DataFile Version 3.0",
      `karman ${name} cell-centered samples of the MAC fields. phi is p/rho.`,
      "ASCII",
      "DATASET STRUCTURED_POINTS",
      `DIMENSIONS ${nx} ${ny} ${nz}`,
      "ORIGIN 0 0 0",
      `SPACING ${this.dx} ${this.dy} ${this.dz}`,
      `POINT_DATA ${nx * ny * nz}`,
      "SCALARS kinematic_pressure double 1",
      "LOOKUP_TABLE default",
    ];
    for (let c = 0; c < this.nC; c++) lines.push(String(this.phi[c]));
    lines.push("SCALARS temperature double 1", "LOOKUP_TABLE default");
    for (let c = 0; c < this.nC; c++) lines.push(String(this.temperature[c]));
    lines.push("VECTORS velocity double");
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const q = this.cellVelocity(i, j, k);
          lines.push(`${q.u} ${q.v} ${q.w}`);
        }
      }
    }
    return lines.join("\n") + "\n";
  }

  toRestart(): string {
    return JSON.stringify({
      version: 1,
      time: this.time,
      steps: this.steps,
      case: this.spec,
      u: Array.from(this.u),
      v: Array.from(this.v),
      w: Array.from(this.w),
      phi: Array.from(this.phi),
      temperature: Array.from(this.temperature),
      dye: Array.from(this.dye),
      solid: Array.from(this.solid),
    });
  }

  static fromRestart(text: string): MacSolver {
    let raw: unknown;
    try {
      raw = JSON.parse(text) as unknown;
    } catch {
      throw new CaseError("restart JSON is not parseable");
    }
    if (!raw || typeof raw !== "object") throw new CaseError("restart must be an object");
    const doc = raw as {
      version?: unknown;
      time?: unknown;
      steps?: unknown;
      case?: unknown;
      u?: unknown;
      v?: unknown;
      w?: unknown;
      phi?: unknown;
      temperature?: unknown;
      dye?: unknown;
      solid?: unknown;
    };
    if (doc.version !== 1) throw new CaseError(`unsupported restart version ${String(doc.version)}`);
    const spec = validateCase(doc.case);
    const time = doc.time;
    const steps = doc.steps;
    if (typeof time !== "number" || !Number.isFinite(time) || time < 0) throw new CaseError("restart time must be a nonnegative finite number");
    if (typeof steps !== "number" || !Number.isInteger(steps) || steps < 0) throw new CaseError("restart step counter must be a nonnegative integer");
    const nC = spec.nx * spec.ny * spec.nz;
    const nU = (spec.nx + 1) * spec.ny * spec.nz;
    const nV = spec.nx * (spec.ny + 1) * spec.nz;
    const nW = spec.nx * spec.ny * (spec.nz + 1);
    const finiteArray = (value: unknown, length: number, label: string): number[] => {
      if (!Array.isArray(value) || value.length !== length) {
        throw new CaseError(`${label} must be an array of length ${length}`);
      }
      for (let i = 0; i < value.length; i++) {
        if (typeof value[i] !== "number" || !Number.isFinite(value[i])) throw new CaseError(`${label}[${i}] must be finite`);
      }
      return value as number[];
    };
    const u = finiteArray(doc.u, nU, "u");
    const v = finiteArray(doc.v, nV, "v");
    const w = finiteArray(doc.w, nW, "w");
    const phi = finiteArray(doc.phi, nC, "phi");
    const temperature = finiteArray(doc.temperature, nC, "temperature");
    const dye = finiteArray(doc.dye, nC, "dye");
    if (!Array.isArray(doc.solid) || doc.solid.length !== nC) throw new CaseError(`solid must be an array of length ${nC}`);
    const solid = new Uint8Array(nC);
    for (let i = 0; i < nC; i++) {
      if (doc.solid[i] !== 0 && doc.solid[i] !== 1) throw new CaseError(`solid[${i}] must be 0 or 1`);
      solid[i] = doc.solid[i];
    }
    const sim = new MacSolver(spec, solid);
    sim.u.set(u);
    sim.v.set(v);
    sim.w.set(w);
    sim.phi.set(phi);
    sim.temperature.set(temperature);
    sim.dye.set(dye);
    sim.time = time;
    sim.steps = steps;
    return sim;
  }

  private rebuildFaces(): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    this.freeU.fill(0);
    this.freeV.fill(0);
    this.freeW.fill(0);
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i <= nx; i++) {
          const id = this.iu(i, j, k);
          if (b.west.kind === "periodic" && i === nx) continue;
          const blocked =
            (i > 0 && this.solid[this.ic(i - 1, j, k)] === 1) || (i < nx && this.solid[this.ic(i, j, k)] === 1);
          const prescribed =
            (i === 0 && isVelocityBC(b.west)) || (i === nx && isVelocityBC(b.east)) || blocked;
          this.freeU[id] = prescribed ? 0 : 1;
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j <= ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iv(i, j, k);
          if (b.south.kind === "periodic" && j === ny) continue;
          const blocked =
            (j > 0 && this.solid[this.ic(i, j - 1, k)] === 1) || (j < ny && this.solid[this.ic(i, j, k)] === 1);
          const prescribed =
            (j === 0 && isVelocityBC(b.south)) || (j === ny && isVelocityBC(b.north)) || blocked;
          this.freeV[id] = prescribed ? 0 : 1;
        }
      }
    }
    for (let k = 0; k <= nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iw(i, j, k);
          if (b.back.kind === "periodic" && k === nz) continue;
          const blocked =
            (k > 0 && this.solid[this.ic(i, j, k - 1)] === 1) || (k < nz && this.solid[this.ic(i, j, k)] === 1);
          const prescribed =
            (k === 0 && isVelocityBC(b.back)) || (k === nz && isVelocityBC(b.front)) || blocked;
          this.freeW[id] = prescribed ? 0 : 1;
        }
      }
    }
    this.components = this.findComponents();
    this.unknown = [];
    for (const comp of this.components) for (const c of comp.cells) this.unknown.push(c);
  }

  private findComponents(): Component[] {
    const { nx, ny, nz, boundaries: b } = this.spec;
    const seen = new Int16Array(this.nC).fill(-1);
    const out: Component[] = [];
    for (let start = 0; start < this.nC; start++) {
      if (this.solid[start] === 1 || seen[start] !== -1) continue;
      const cells: number[] = [];
      const stack = [start];
      seen[start] = out.length;
      let closed = true;
      while (stack.length > 0) {
        const c = stack.pop()!;
        cells.push(c);
        const i = c % nx;
        const j = Math.floor(c / nx) % ny;
        const k = Math.floor(c / (nx * ny));
        const link = (ni: number, nj: number, nk: number, side: BoundaryName | null) => {
          if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
            if (side && b[side].kind === "outlet") closed = false;
            if (side && b[side].kind === "periodic") {
              const wi = (ni + nx) % nx;
              const wj = (nj + ny) % ny;
              const wk = (nk + nz) % nz;
              const id = this.ic(wi, wj, wk);
              if (this.solid[id] !== 1 && seen[id] === -1) {
                seen[id] = out.length;
                stack.push(id);
              }
            }
            return;
          }
          const id = this.ic(ni, nj, nk);
          if (this.solid[id] === 1 || seen[id] !== -1) return;
          seen[id] = out.length;
          stack.push(id);
        };
        link(i - 1, j, k, "west");
        link(i + 1, j, k, "east");
        link(i, j - 1, k, "south");
        link(i, j + 1, k, "north");
        link(i, j, k - 1, "back");
        link(i, j, k + 1, "front");
      }
      out.push({ cells, pin: closed ? cells[0]! : -1, closed });
    }
    return out;
  }

  private applyVelocityBC(): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        if (isVelocityBC(b.west)) this.u[this.iu(0, j, k)] = b.west.u;
        if (isVelocityBC(b.east)) this.u[this.iu(nx, j, k)] = b.east.u;
        for (let i = 0; i <= nx; i++) {
          const id = this.iu(i, j, k);
          const blocked =
            (i > 0 && this.solid[this.ic(i - 1, j, k)] === 1) || (i < nx && this.solid[this.ic(i, j, k)] === 1);
          if (blocked) this.u[id] = 0;
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let i = 0; i < nx; i++) {
        if (isVelocityBC(b.south)) this.v[this.iv(i, 0, k)] = b.south.v;
        if (isVelocityBC(b.north)) this.v[this.iv(i, ny, k)] = b.north.v;
        for (let j = 0; j <= ny; j++) {
          const id = this.iv(i, j, k);
          const blocked =
            (j > 0 && this.solid[this.ic(i, j - 1, k)] === 1) || (j < ny && this.solid[this.ic(i, j, k)] === 1);
          if (blocked) this.v[id] = 0;
        }
      }
    }
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (isVelocityBC(b.back)) this.w[this.iw(i, j, 0)] = b.back.w;
        if (isVelocityBC(b.front)) this.w[this.iw(i, j, nz)] = b.front.w;
        for (let k = 0; k <= nz; k++) {
          const id = this.iw(i, j, k);
          const blocked =
            (k > 0 && this.solid[this.ic(i, j, k - 1)] === 1) || (k < nz && this.solid[this.ic(i, j, k)] === 1);
          if (blocked) this.w[id] = 0;
        }
      }
    }
    this.syncPeriodic();
  }

  /**
   * Zero reversed outlet faces and freeze them for a second projection.
   * Returns how many faces changed. Call `releaseHeld` before the next step
   * so those faces are free again; `restore` does that itself.
   */
  private holdBackflow(): number {
    const { nx, ny, nz, boundaries: b } = this.spec;
    let changed = 0;
    const take = (
      side: FaceBoundary,
      inward: boolean,
      id: number,
      vel: Float64Array,
      free: Uint8Array,
      held: number[],
    ) => {
      if (side.kind !== "outlet" || side.backflow !== "clamp" || !inward) return;
      if (vel[id] === 0) return;
      vel[id] = 0;
      changed += 1;
      if (free[id] === 1) {
        free[id] = 0;
        held.push(id);
      }
    };
    if (b.west.kind === "outlet" || b.east.kind === "outlet") {
      for (let k = 0; k < nz; k++) {
        for (let j = 0; j < ny; j++) {
          take(b.west, this.u[this.iu(0, j, k)]! > 0, this.iu(0, j, k), this.u, this.freeU, this.heldU);
          take(b.east, this.u[this.iu(nx, j, k)]! < 0, this.iu(nx, j, k), this.u, this.freeU, this.heldU);
        }
      }
    }
    if (b.south.kind === "outlet" || b.north.kind === "outlet") {
      for (let k = 0; k < nz; k++) {
        for (let i = 0; i < nx; i++) {
          take(b.south, this.v[this.iv(i, 0, k)]! > 0, this.iv(i, 0, k), this.v, this.freeV, this.heldV);
          take(b.north, this.v[this.iv(i, ny, k)]! < 0, this.iv(i, ny, k), this.v, this.freeV, this.heldV);
        }
      }
    }
    if (b.back.kind === "outlet" || b.front.kind === "outlet") {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          take(b.back, this.w[this.iw(i, j, 0)]! > 0, this.iw(i, j, 0), this.w, this.freeW, this.heldW);
          take(b.front, this.w[this.iw(i, j, nz)]! < 0, this.iw(i, j, nz), this.w, this.freeW, this.heldW);
        }
      }
    }
    return changed;
  }

  private releaseHeld(): void {
    for (const id of this.heldU) this.freeU[id] = 1;
    for (const id of this.heldV) this.freeV[id] = 1;
    for (const id of this.heldW) this.freeW[id] = 1;
    this.heldU.length = 0;
    this.heldV.length = 0;
    this.heldW.length = 0;
  }

  /** Net volume flux through every fluid cell, m³/s. Zero on a discrete divergence-free field. */
  private volumeFlux(): number {
    this.writeDivergence(this.div);
    const vol = this.dx * this.dy * this.dz;
    let net = 0;
    for (let c = 0; c < this.nC; c++) if (this.solid[c] !== 1) net += this.div[c]! * vol;
    return net;
  }

  private fluxTolerance(): number {
    return 1e-6 * Math.max(this.spec.lx * this.spec.ly * this.spec.lz, this.dx * this.dy * this.dz);
  }

  private syncPeriodic(): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    if (b.west.kind === "periodic") {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) this.u[this.iu(nx, j, k)] = this.u[this.iu(0, j, k)]!;
    }
    if (b.south.kind === "periodic") {
      for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) this.v[this.iv(i, ny, k)] = this.v[this.iv(i, 0, k)]!;
    }
    if (b.back.kind === "periodic") {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) this.w[this.iw(i, j, nz)] = this.w[this.iw(i, j, 0)]!;
    }
  }

  private syncCopies(uu: Float64Array, vv: Float64Array, ww: Float64Array): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    if (b.west.kind === "periodic") {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) uu[this.iu(nx, j, k)] = uu[this.iu(0, j, k)]!;
    }
    if (b.south.kind === "periodic") {
      for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) vv[this.iv(i, ny, k)] = vv[this.iv(i, 0, k)]!;
    }
    if (b.back.kind === "periodic") {
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) ww[this.iw(i, j, nz)] = ww[this.iw(i, j, 0)]!;
    }
  }

  private addForces(dt: number): void {
    const { nx, ny, nz, force: f, gravity: g, beta, tRef } = this.spec;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i <= nx; i++) {
          const id = this.iu(i, j, k);
          if (this.freeU[id] !== 1) continue;
          const t = this.faceTemperature("u", i, j, k);
          const ax = f.x - beta * (t - tRef) * g.x;
          this.u[id] = this.u[id]! + dt * ax;
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j <= ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iv(i, j, k);
          if (this.freeV[id] !== 1) continue;
          const t = this.faceTemperature("v", i, j, k);
          const ay = f.y - beta * (t - tRef) * g.y;
          this.v[id] = this.v[id]! + dt * ay;
        }
      }
    }
    for (let k = 0; k <= nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.iw(i, j, k);
          if (this.freeW[id] !== 1) continue;
          const t = this.faceTemperature("w", i, j, k);
          const az = f.z - beta * (t - tRef) * g.z;
          this.w[id] = this.w[id]! + dt * az;
        }
      }
    }
    this.syncPeriodic();
  }

  private faceTemperature(axis: "u" | "v" | "w", i: number, j: number, k: number): number {
    const { nx, ny, nz, tRef } = this.spec;
    if (axis === "u") {
      const a = i > 0 ? this.temperature[this.ic(i - 1, j, k)]! : tRef;
      const b = i < nx ? this.temperature[this.ic(i, j, k)]! : tRef;
      if (i === 0) return b;
      if (i === nx) return a;
      return 0.5 * (a + b);
    }
    if (axis === "v") {
      const a = j > 0 ? this.temperature[this.ic(i, j - 1, k)]! : tRef;
      const b = j < ny ? this.temperature[this.ic(i, j, k)]! : tRef;
      if (j === 0) return b;
      if (j === ny) return a;
      return 0.5 * (a + b);
    }
    const a = k > 0 ? this.temperature[this.ic(i, j, k - 1)]! : tRef;
    const c = k < nz ? this.temperature[this.ic(i, j, k)]! : tRef;
    if (k === 0) return c;
    if (k === nz) return a;
    return 0.5 * (a + c);
  }

  private diffuseVelocity(dt: number): void {
    const limit = this.diffusionLimit(this.spec.nu);
    if (!(this.spec.nu > 0) || !(dt > 0)) return;
    let left = dt;
    while (left > 1e-15) {
      const h = Math.min(left, limit);
      this.diffuseVelocityOnce(h);
      left -= h;
    }
  }

  private diffusionLimit(coeff: number): number {
    if (!(coeff > 0)) return Number.POSITIVE_INFINITY;
    const invH = 2 / (this.dx * this.dx) + 2 / (this.dy * this.dy) + 2 / (this.dz * this.dz);
    return this.spec.diffCfl / (coeff * invH);
  }

  private diffuseVelocityOnce(dt: number): void {
    const nu = this.spec.nu;
    const { nx, ny, nz } = this.spec;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        const iMax = this.spec.boundaries.west.kind === "periodic" ? nx - 1 : nx;
        for (let i = 0; i <= iMax; i++) {
          const id = this.iu(i, j, k);
          if (this.freeU[id] !== 1) {
            this.uT[id] = this.u[id]!;
            continue;
          }
          const lap = this.lapU(i, j, k);
          this.uT[id] = this.u[id]! + dt * nu * lap;
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let i = 0; i < nx; i++) {
        const jMax = this.spec.boundaries.south.kind === "periodic" ? ny - 1 : ny;
        for (let j = 0; j <= jMax; j++) {
          const id = this.iv(i, j, k);
          if (this.freeV[id] !== 1) {
            this.vT[id] = this.v[id]!;
            continue;
          }
          this.vT[id] = this.v[id]! + dt * nu * this.lapV(i, j, k);
        }
      }
    }
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const kMax = this.spec.boundaries.back.kind === "periodic" ? nz - 1 : nz;
        for (let k = 0; k <= kMax; k++) {
          const id = this.iw(i, j, k);
          if (this.freeW[id] !== 1) {
            this.wT[id] = this.w[id]!;
            continue;
          }
          this.wT[id] = this.w[id]! + dt * nu * this.lapW(i, j, k);
        }
      }
    }
    this.u.set(this.uT);
    this.v.set(this.vT);
    this.w.set(this.wT);
    this.syncPeriodic();
  }

  private lapU(i: number, j: number, k: number): number {
    const c = this.u[this.iu(i, j, k)]!;
    const { dx, dy, dz } = this;
    return (
      (this.uGhostX(i - 1, j, k) - 2 * c + this.uGhostX(i + 1, j, k)) / (dx * dx) +
      (this.uGhostY(i, j - 1, k) - 2 * c + this.uGhostY(i, j + 1, k)) / (dy * dy) +
      (this.uGhostZ(i, j, k - 1) - 2 * c + this.uGhostZ(i, j, k + 1)) / (dz * dz)
    );
  }

  private lapV(i: number, j: number, k: number): number {
    const c = this.v[this.iv(i, j, k)]!;
    const { dx, dy, dz } = this;
    return (
      (this.vGhostX(i - 1, j, k) - 2 * c + this.vGhostX(i + 1, j, k)) / (dx * dx) +
      (this.vGhostY(i, j - 1, k) - 2 * c + this.vGhostY(i, j + 1, k)) / (dy * dy) +
      (this.vGhostZ(i, j, k - 1) - 2 * c + this.vGhostZ(i, j, k + 1)) / (dz * dz)
    );
  }

  private lapW(i: number, j: number, k: number): number {
    const c = this.w[this.iw(i, j, k)]!;
    const { dx, dy, dz } = this;
    return (
      (this.wGhostX(i - 1, j, k) - 2 * c + this.wGhostX(i + 1, j, k)) / (dx * dx) +
      (this.wGhostY(i, j - 1, k) - 2 * c + this.wGhostY(i, j + 1, k)) / (dy * dy) +
      (this.wGhostZ(i, j, k - 1) - 2 * c + this.wGhostZ(i, j, k + 1)) / (dz * dz)
    );
  }

  private uGhostX(i: number, j: number, k: number): number {
    const { nx, boundaries: b } = this.spec;
    if (b.west.kind === "periodic") return this.u[this.iu(((i % nx) + nx) % nx, j, k)]!;
    if (i < 0) return isVelocityBC(b.west) ? b.west.u : this.u[this.iu(0, j, k)]!;
    if (i > nx) return isVelocityBC(b.east) ? b.east.u : this.u[this.iu(nx, j, k)]!;
    return this.u[this.iu(i, j, k)]!;
  }
  private uGhostY(i: number, j: number, k: number): number {
    const { ny, boundaries: b } = this.spec;
    if (j >= 0 && j < ny) return this.u[this.iu(this.wrapI(i), j, k)]!;
    if (b.south.kind === "periodic") return this.u[this.iu(this.wrapI(i), (j + ny) % ny, k)]!;
    const bc = j < 0 ? b.south : b.north;
    const inner = j < 0 ? 0 : ny - 1;
    return dirichletTangential(bc) ? 2 * bc.u - this.u[this.iu(this.wrapI(i), inner, k)]! : this.u[this.iu(this.wrapI(i), inner, k)]!;
  }
  private uGhostZ(i: number, j: number, k: number): number {
    const { nz, boundaries: b } = this.spec;
    if (k >= 0 && k < nz) return this.u[this.iu(this.wrapI(i), j, k)]!;
    if (b.back.kind === "periodic") return this.u[this.iu(this.wrapI(i), j, (k + nz) % nz)]!;
    const bc = k < 0 ? b.back : b.front;
    const inner = k < 0 ? 0 : nz - 1;
    return dirichletTangential(bc) ? 2 * bc.u - this.u[this.iu(this.wrapI(i), j, inner)]! : this.u[this.iu(this.wrapI(i), j, inner)]!;
  }
  private vGhostY(i: number, j: number, k: number): number {
    const { ny, boundaries: b } = this.spec;
    if (b.south.kind === "periodic") return this.v[this.iv(i, ((j % ny) + ny) % ny, k)]!;
    if (j < 0) return isVelocityBC(b.south) ? b.south.v : this.v[this.iv(i, 0, k)]!;
    if (j > ny) return isVelocityBC(b.north) ? b.north.v : this.v[this.iv(i, ny, k)]!;
    return this.v[this.iv(i, j, k)]!;
  }
  private vGhostX(i: number, j: number, k: number): number {
    const { nx, boundaries: b } = this.spec;
    if (i >= 0 && i < nx) return this.v[this.iv(i, this.wrapJ(j), k)]!;
    if (b.west.kind === "periodic") return this.v[this.iv((i + nx) % nx, this.wrapJ(j), k)]!;
    const bc = i < 0 ? b.west : b.east;
    const inner = i < 0 ? 0 : nx - 1;
    return dirichletTangential(bc) ? 2 * bc.v - this.v[this.iv(inner, this.wrapJ(j), k)]! : this.v[this.iv(inner, this.wrapJ(j), k)]!;
  }
  private vGhostZ(i: number, j: number, k: number): number {
    const { nz, boundaries: b } = this.spec;
    if (k >= 0 && k < nz) return this.v[this.iv(i, this.wrapJ(j), k)]!;
    if (b.back.kind === "periodic") return this.v[this.iv(i, this.wrapJ(j), (k + nz) % nz)]!;
    const bc = k < 0 ? b.back : b.front;
    const inner = k < 0 ? 0 : nz - 1;
    return dirichletTangential(bc) ? 2 * bc.v - this.v[this.iv(i, this.wrapJ(j), inner)]! : this.v[this.iv(i, this.wrapJ(j), inner)]!;
  }
  private wGhostZ(i: number, j: number, k: number): number {
    const { nz, boundaries: b } = this.spec;
    if (b.back.kind === "periodic") return this.w[this.iw(i, j, ((k % nz) + nz) % nz)]!;
    if (k < 0) return isVelocityBC(b.back) ? b.back.w : this.w[this.iw(i, j, 0)]!;
    if (k > nz) return isVelocityBC(b.front) ? b.front.w : this.w[this.iw(i, j, nz)]!;
    return this.w[this.iw(i, j, k)]!;
  }
  private wGhostX(i: number, j: number, k: number): number {
    const { nx, boundaries: b } = this.spec;
    if (i >= 0 && i < nx) return this.w[this.iw(i, j, this.wrapK(k))]!;
    if (b.west.kind === "periodic") return this.w[this.iw((i + nx) % nx, j, this.wrapK(k))]!;
    const bc = i < 0 ? b.west : b.east;
    const inner = i < 0 ? 0 : nx - 1;
    return dirichletTangential(bc) ? 2 * bc.w - this.w[this.iw(inner, j, this.wrapK(k))]! : this.w[this.iw(inner, j, this.wrapK(k))]!;
  }
  private wGhostY(i: number, j: number, k: number): number {
    const { ny, boundaries: b } = this.spec;
    if (j >= 0 && j < ny) return this.w[this.iw(i, j, this.wrapK(k))]!;
    if (b.south.kind === "periodic") return this.w[this.iw(i, (j + ny) % ny, this.wrapK(k))]!;
    const bc = j < 0 ? b.south : b.north;
    const inner = j < 0 ? 0 : ny - 1;
    return dirichletTangential(bc) ? 2 * bc.w - this.w[this.iw(i, inner, this.wrapK(k))]! : this.w[this.iw(i, inner, this.wrapK(k))]!;
  }

  private wrapI(i: number): number {
    const { nx, boundaries: b } = this.spec;
    if (i >= 0 && i <= nx) return i;
    if (b.west.kind === "periodic") return ((i % nx) + nx) % nx;
    return Math.max(0, Math.min(nx, i));
  }
  private wrapJ(j: number): number {
    const { ny, boundaries: b } = this.spec;
    if (j >= 0 && j <= ny) return j;
    if (b.south.kind === "periodic") return ((j % ny) + ny) % ny;
    return Math.max(0, Math.min(ny, j));
  }
  private wrapK(k: number): number {
    const { nz, boundaries: b } = this.spec;
    if (k >= 0 && k <= nz) return k;
    if (b.back.kind === "periodic") return ((k % nz) + nz) % nz;
    return Math.max(0, Math.min(nz, k));
  }

  private advectMomentum(dt: number): void {
    const order = this.spec.advection === "muscl-rk2" ? 2 : 1;
    const stages = this.spec.advection === "muscl-rk2" ? 2 : 1;
    this.uT.set(this.u);
    this.vT.set(this.v);
    this.wT.set(this.w);
    this.momentumRhs(order);
    this.su.set(this.ru);
    this.sv.set(this.rv);
    this.sw.set(this.rw);
    if (stages === 2) {
      this.axpyVelocity(dt, this.ru, this.rv, this.rw, this.uT, this.vT, this.wT);
      this.applyVelocityBC();
      this.momentumRhs(order);
      for (let i = 0; i < this.nU; i++) this.ru[i] = 0.5 * (this.su[i]! + this.ru[i]!);
      for (let i = 0; i < this.nV; i++) this.rv[i] = 0.5 * (this.sv[i]! + this.rv[i]!);
      for (let i = 0; i < this.nW; i++) this.rw[i] = 0.5 * (this.sw[i]! + this.rw[i]!);
    }
    this.axpyVelocity(dt, this.ru, this.rv, this.rw, this.uT, this.vT, this.wT);
    this.applyVelocityBC();
  }

  private axpyVelocity(
    dt: number,
    ru: Float64Array,
    rv: Float64Array,
    rw: Float64Array,
    baseU: Float64Array,
    baseV: Float64Array,
    baseW: Float64Array,
  ): void {
    for (let i = 0; i < this.nU; i++) this.u[i] = this.freeU[i] === 1 ? baseU[i]! + dt * ru[i]! : baseU[i]!;
    for (let i = 0; i < this.nV; i++) this.v[i] = this.freeV[i] === 1 ? baseV[i]! + dt * rv[i]! : baseV[i]!;
    for (let i = 0; i < this.nW; i++) this.w[i] = this.freeW[i] === 1 ? baseW[i]! + dt * rw[i]! : baseW[i]!;
    this.syncPeriodic();
  }

  private momentumRhs(order: number): void {
    const { nx, ny, nz } = this.spec;
    const { dx, dy, dz } = this;
    this.ru.fill(0);
    this.rv.fill(0);
    this.rw.fill(0);
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        const iMax = this.spec.boundaries.west.kind === "periodic" ? nx - 1 : nx;
        for (let i = 0; i <= iMax; i++) {
          const id = this.iu(i, j, k);
          if (this.freeU[id] !== 1) continue;
          const fe = this.uFluxX(i, j, k, order);
          const fw = this.uFluxX(i - 1, j, k, order);
          const fn = this.uFluxY(i, j, k, order);
          const fs = this.uFluxY(i, j - 1, k, order);
          const ff = this.uFluxZ(i, j, k, order);
          const fb = this.uFluxZ(i, j, k - 1, order);
          this.ru[id] = -((fe - fw) / dx + (fn - fs) / dy + (ff - fb) / dz);
        }
      }
    }
    for (let k = 0; k < nz; k++) {
      for (let i = 0; i < nx; i++) {
        const jMax = this.spec.boundaries.south.kind === "periodic" ? ny - 1 : ny;
        for (let j = 0; j <= jMax; j++) {
          const id = this.iv(i, j, k);
          if (this.freeV[id] !== 1) continue;
          const fe = this.vFluxX(i, j, k, order);
          const fw = this.vFluxX(i - 1, j, k, order);
          const fn = this.vFluxY(i, j, k, order);
          const fs = this.vFluxY(i, j - 1, k, order);
          const ff = this.vFluxZ(i, j, k, order);
          const fb = this.vFluxZ(i, j, k - 1, order);
          this.rv[id] = -((fe - fw) / dx + (fn - fs) / dy + (ff - fb) / dz);
        }
      }
    }
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const kMax = this.spec.boundaries.back.kind === "periodic" ? nz - 1 : nz;
        for (let k = 0; k <= kMax; k++) {
          const id = this.iw(i, j, k);
          if (this.freeW[id] !== 1) continue;
          const fe = this.wFluxX(i, j, k, order);
          const fw = this.wFluxX(i - 1, j, k, order);
          const fn = this.wFluxY(i, j, k, order);
          const fs = this.wFluxY(i, j - 1, k, order);
          const ff = this.wFluxZ(i, j, k, order);
          const fb = this.wFluxZ(i, j, k - 1, order);
          this.rw[id] = -((fe - fw) / dx + (fn - fs) / dy + (ff - fb) / dz);
        }
      }
    }
  }

  private uFluxX(i: number, j: number, k: number, order: number): number {
    const left0 = this.sampleU(i, j, k);
    const right0 = this.sampleU(i + 1, j, k);
    const adv = 0.5 * (left0 + right0);
    if (order < 2) return adv >= 0 ? adv * left0 : adv * right0;
    const left = left0 + 0.5 * minmod(left0 - this.sampleU(i - 1, j, k), right0 - left0);
    const right = right0 - 0.5 * minmod(right0 - left0, this.sampleU(i + 2, j, k) - right0);
    return adv >= 0 ? adv * left : adv * right;
  }

  private uFluxY(i: number, j: number, k: number, order: number): number {
    const vel = this.vAtU(i, j + 1, k);
    return this.transverse(vel, order, (dj) => this.sampleU(i, j + dj, k));
  }

  private uFluxZ(i: number, j: number, k: number, order: number): number {
    const vel = this.wAtU(i, j, k + 1);
    return this.transverse(vel, order, (dk) => this.sampleU(i, j, k + dk));
  }

  private vFluxY(i: number, j: number, k: number, order: number): number {
    const left0 = this.sampleV(i, j, k);
    const right0 = this.sampleV(i, j + 1, k);
    const adv = 0.5 * (left0 + right0);
    if (order < 2) return adv >= 0 ? adv * left0 : adv * right0;
    const left = left0 + 0.5 * minmod(left0 - this.sampleV(i, j - 1, k), right0 - left0);
    const right = right0 - 0.5 * minmod(right0 - left0, this.sampleV(i, j + 2, k) - right0);
    return adv >= 0 ? adv * left : adv * right;
  }

  private vFluxX(i: number, j: number, k: number, order: number): number {
    const vel = this.uAtV(i + 1, j, k);
    return this.transverse(vel, order, (di) => this.sampleV(i + di, j, k));
  }

  private vFluxZ(i: number, j: number, k: number, order: number): number {
    const vel = this.wAtV(i, j, k + 1);
    return this.transverse(vel, order, (dk) => this.sampleV(i, j, k + dk));
  }

  private wFluxZ(i: number, j: number, k: number, order: number): number {
    const left0 = this.sampleW(i, j, k);
    const right0 = this.sampleW(i, j, k + 1);
    const adv = 0.5 * (left0 + right0);
    if (order < 2) return adv >= 0 ? adv * left0 : adv * right0;
    const left = left0 + 0.5 * minmod(left0 - this.sampleW(i, j, k - 1), right0 - left0);
    const right = right0 - 0.5 * minmod(right0 - left0, this.sampleW(i, j, k + 2) - right0);
    return adv >= 0 ? adv * left : adv * right;
  }

  private wFluxX(i: number, j: number, k: number, order: number): number {
    const vel = this.uAtW(i + 1, j, k);
    return this.transverse(vel, order, (di) => this.sampleW(i + di, j, k));
  }

  private wFluxY(i: number, j: number, k: number, order: number): number {
    const vel = this.vAtW(i, j + 1, k);
    return this.transverse(vel, order, (dj) => this.sampleW(i, j + dj, k));
  }

  private transverse(vel: number, order: number, at: (delta: number) => number): number {
    const left0 = at(0);
    const right0 = at(1);
    if (order < 2) return vel >= 0 ? vel * left0 : vel * right0;
    const left = left0 + 0.5 * minmod(left0 - at(-1), right0 - left0);
    const right = right0 - 0.5 * minmod(right0 - left0, at(2) - right0);
    return vel >= 0 ? vel * left : vel * right;
  }

  private sampleU(i: number, j: number, k: number): number {
    const { ny, nz } = this.spec;
    if (j < 0 || j >= ny) return this.uGhostY(i, j, clamp(k, 0, nz - 1));
    if (k < 0 || k >= nz) return this.uGhostZ(i, j, k);
    return this.uGhostX(i, j, k);
  }

  private sampleV(i: number, j: number, k: number): number {
    const { nx, ny, nz } = this.spec;
    if (j < 0 || j > ny) return this.vGhostY(clamp(i, 0, nx - 1), j, clamp(k, 0, nz - 1));
    if (i < 0 || i >= nx) return this.vGhostX(i, j, clamp(k, 0, nz - 1));
    if (k < 0 || k >= nz) return this.vGhostZ(i, j, k);
    return this.v[this.iv(i, j, k)]!;
  }

  private sampleW(i: number, j: number, k: number): number {
    const { nx, ny, nz } = this.spec;
    if (k < 0 || k > nz) return this.wGhostZ(clamp(i, 0, nx - 1), clamp(j, 0, ny - 1), k);
    if (i < 0 || i >= nx) return this.wGhostX(i, j, k);
    if (j < 0 || j >= ny) return this.wGhostY(i, j, k);
    return this.w[this.iw(i, j, k)]!;
  }

  private vAtU(i: number, jV: number, k: number): number {
    const { nx, boundaries: b } = this.spec;
    if ((i === 0 && b.west.kind !== "periodic") || (i === nx && b.east.kind !== "periodic")) {
      const ii = i === 0 ? 0 : nx - 1;
      return this.sampleV(ii, jV, k);
    }
    const iL = b.west.kind === "periodic" ? ((i - 1) % nx + nx) % nx : i - 1;
    const iR = b.west.kind === "periodic" ? i % nx : i;
    return 0.5 * (this.sampleV(iL, jV, k) + this.sampleV(iR, jV, k));
  }

  private wAtU(i: number, j: number, kW: number): number {
    const { nx, boundaries: b } = this.spec;
    if ((i === 0 && b.west.kind !== "periodic") || (i === nx && b.east.kind !== "periodic")) {
      const ii = i === 0 ? 0 : nx - 1;
      return this.sampleW(ii, j, kW);
    }
    const iL = b.west.kind === "periodic" ? ((i - 1) % nx + nx) % nx : i - 1;
    const iR = b.west.kind === "periodic" ? i % nx : i;
    return 0.5 * (this.sampleW(iL, j, kW) + this.sampleW(iR, j, kW));
  }

  private uAtV(iU: number, j: number, k: number): number {
    const { ny, boundaries: b } = this.spec;
    if ((j === 0 && b.south.kind !== "periodic") || (j === ny && b.north.kind !== "periodic")) {
      const jj = j === 0 ? 0 : ny - 1;
      return this.sampleU(iU, jj, k);
    }
    const jL = b.south.kind === "periodic" ? ((j - 1) % ny + ny) % ny : j - 1;
    const jR = b.south.kind === "periodic" ? j % ny : j;
    return 0.5 * (this.sampleU(iU, jL, k) + this.sampleU(iU, jR, k));
  }

  private wAtV(i: number, j: number, kW: number): number {
    const { ny, boundaries: b } = this.spec;
    if ((j === 0 && b.south.kind !== "periodic") || (j === ny && b.north.kind !== "periodic")) {
      const jj = j === 0 ? 0 : ny - 1;
      return this.sampleW(i, jj, kW);
    }
    const jL = b.south.kind === "periodic" ? ((j - 1) % ny + ny) % ny : j - 1;
    const jR = b.south.kind === "periodic" ? j % ny : j;
    return 0.5 * (this.sampleW(i, jL, kW) + this.sampleW(i, jR, kW));
  }

  private uAtW(iU: number, j: number, k: number): number {
    const { nz, boundaries: b } = this.spec;
    if ((k === 0 && b.back.kind !== "periodic") || (k === nz && b.front.kind !== "periodic")) {
      const kk = k === 0 ? 0 : nz - 1;
      return this.sampleU(iU, j, kk);
    }
    const kL = b.back.kind === "periodic" ? ((k - 1) % nz + nz) % nz : k - 1;
    const kR = b.back.kind === "periodic" ? k % nz : k;
    return 0.5 * (this.sampleU(iU, j, kL) + this.sampleU(iU, j, kR));
  }

  private vAtW(i: number, jV: number, k: number): number {
    const { nz, boundaries: b } = this.spec;
    if ((k === 0 && b.back.kind !== "periodic") || (k === nz && b.front.kind !== "periodic")) {
      const kk = k === 0 ? 0 : nz - 1;
      return this.sampleV(i, jV, kk);
    }
    const kL = b.back.kind === "periodic" ? ((k - 1) % nz + nz) % nz : k - 1;
    const kR = b.back.kind === "periodic" ? k % nz : k;
    return 0.5 * (this.sampleV(i, jV, kL) + this.sampleV(i, jV, kR));
  }

  private project(dt: number): { converged: boolean; iterations: number; residual: number; detail: string } {
    this.syncPeriodic();
    this.writeDivergence(this.div);
    const { atol, rtol, maxIterations } = this.spec.poisson;
    const invDt = 1 / dt;
    this.rhs.fill(0);
    for (let c = 0; c < this.nC; c++) {
      if (this.solid[c] === 1) continue;
      this.rhs[c] = -this.div[c]! * invDt + this.outletRhs(c);
    }
    for (const comp of this.components) {
      if (!comp.closed) continue;
      let sum = 0;
      for (const c of comp.cells) sum += this.rhs[c]!;
      const mean = sum / comp.cells.length;
      for (const c of comp.cells) this.rhs[c] = this.rhs[c]! - mean;
    }
    this.x.set(this.phi);
    this.deflate(this.x);
    this.applyA(this.x, this.ap);
    let rr = 0;
    let bb = 0;
    for (const c of this.unknown) {
      this.r[c] = this.rhs[c]! - this.ap[c]!;
      rr += this.r[c]! * this.r[c]!;
      bb += this.rhs[c]! * this.rhs[c]!;
    }
    this.deflate(this.r);
    rr = 0;
    for (const c of this.unknown) {
      this.pvec[c] = this.r[c]!;
      rr += this.r[c]! * this.r[c]!;
    }
    const bnorm = Math.sqrt(bb);
    const tol = atol + rtol * bnorm;
    let iterations = 0;
    let residual = Math.sqrt(rr);
    let converged = residual <= tol;
    while (!converged && iterations < maxIterations) {
      this.deflate(this.pvec);
      this.applyA(this.pvec, this.ap);
      let pap = 0;
      for (const c of this.unknown) pap += this.pvec[c]! * this.ap[c]!;
      if (!(pap > 0)) break;
      const alpha = rr / pap;
      let rr2 = 0;
      for (const c of this.unknown) {
        this.x[c] = this.x[c]! + alpha * this.pvec[c]!;
        this.r[c] = this.r[c]! - alpha * this.ap[c]!;
        rr2 += this.r[c]! * this.r[c]!;
      }
      this.deflate(this.r);
      this.deflate(this.x);
      rr2 = 0;
      for (const c of this.unknown) rr2 += this.r[c]! * this.r[c]!;
      iterations += 1;
      residual = Math.sqrt(rr2);
      if (residual <= tol) {
        converged = true;
        rr = rr2;
        break;
      }
      const beta = rr2 / rr;
      for (const c of this.unknown) this.pvec[c] = this.r[c]! + beta * this.pvec[c]!;
      rr = rr2;
    }
    for (let c = 0; c < this.nC; c++) this.phi[c] = this.solid[c] === 1 ? 0 : this.x[c]!;
    for (const comp of this.components) {
      if (!comp.closed) continue;
      let sum = 0;
      for (const c of comp.cells) sum += this.phi[c]!;
      const mean = sum / comp.cells.length;
      for (const c of comp.cells) this.phi[c] = this.phi[c]! - mean;
    }
    this.writeGradient(dt, this.uT, this.vT, this.wT);
    for (let i = 0; i < this.nU; i++) if (this.freeU[i] === 1) this.u[i] = this.u[i]! - this.uT[i]!;
    for (let i = 0; i < this.nV; i++) if (this.freeV[i] === 1) this.v[i] = this.v[i]! - this.vT[i]!;
    for (let i = 0; i < this.nW; i++) if (this.freeW[i] === 1) this.w[i] = this.w[i]! - this.wT[i]!;
    this.syncPeriodic();
    const detail = converged
      ? `CG residual ${residual.toExponential(2)} in ${iterations}`
      : `pressure CG did not converge: residual ${residual.toExponential(2)} after ${iterations}, tolerance ${tol.toExponential(2)}`;
    return { converged, iterations, residual, detail };
  }

  private deflate(vec: Float64Array): void {
    for (const comp of this.components) {
      if (!comp.closed) continue;
      let sum = 0;
      for (const c of comp.cells) sum += vec[c]!;
      const mean = sum / comp.cells.length;
      for (const c of comp.cells) vec[c] = vec[c]! - mean;
    }
  }

  private outletRhs(c: number): number {
    const { nx, ny, nz, boundaries: b } = this.spec;
    const i = c % nx;
    const j = Math.floor(c / nx) % ny;
    const k = Math.floor(c / (nx * ny));
    let s = 0;
    if (i === 0 && b.west.kind === "outlet") s += b.west.phi / (this.dx * this.dx);
    if (i === nx - 1 && b.east.kind === "outlet") s += b.east.phi / (this.dx * this.dx);
    if (j === 0 && b.south.kind === "outlet") s += b.south.phi / (this.dy * this.dy);
    if (j === ny - 1 && b.north.kind === "outlet") s += b.north.phi / (this.dy * this.dy);
    if (k === 0 && b.back.kind === "outlet") s += b.back.phi / (this.dz * this.dz);
    if (k === nz - 1 && b.front.kind === "outlet") s += b.front.phi / (this.dz * this.dz);
    return s;
  }

  private applyA(x: Float64Array, y: Float64Array): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    const invX = 1 / (this.dx * this.dx);
    const invY = 1 / (this.dy * this.dy);
    const invZ = 1 / (this.dz * this.dz);
    y.fill(0);
    for (const c of this.unknown) {
      const i = c % nx;
      const j = Math.floor(c / nx) % ny;
      const k = Math.floor(c / (nx * ny));
      let diag = 0;
      let off = 0;
      const take = (ni: number, nj: number, nk: number, inv: number, side: BoundaryName | null) => {
        if (ni < 0 || nj < 0 || nk < 0 || ni >= nx || nj >= ny || nk >= nz) {
          if (side && b[side].kind === "outlet") diag += inv;
          else if (side && b[side].kind === "periodic") {
            const wi = (ni + nx) % nx;
            const wj = (nj + ny) % ny;
            const wk = (nk + nz) % nz;
            const id = this.ic(wi, wj, wk);
            if (this.solid[id] !== 1) {
              diag += inv;
              off += inv * x[id]!;
            }
          }
          return;
        }
        const id = this.ic(ni, nj, nk);
        if (this.solid[id] === 1) return;
        diag += inv;
        off += inv * x[id]!;
      };
      take(i - 1, j, k, invX, "west");
      take(i + 1, j, k, invX, "east");
      take(i, j - 1, k, invY, "south");
      take(i, j + 1, k, invY, "north");
      take(i, j, k - 1, invZ, "back");
      take(i, j, k + 1, invZ, "front");
      y[c] = diag * x[c]! - off;
    }
  }

  private writeDivergence(out: Float64Array): void {
    const { nx, ny, nz } = this.spec;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.ic(i, j, k);
          if (this.solid[id] === 1) {
            out[id] = 0;
            continue;
          }
          out[id] =
            (this.u[this.iu(i + 1, j, k)]! - this.u[this.iu(i, j, k)]!) / this.dx +
            (this.v[this.iv(i, j + 1, k)]! - this.v[this.iv(i, j, k)]!) / this.dy +
            (this.w[this.iw(i, j, k + 1)]! - this.w[this.iw(i, j, k)]!) / this.dz;
        }
      }
    }
  }

  private advanceScalars(dt: number): void {
    if (this.spec.advection !== "none") this.advectScalar(this.dye, this.dyeT, dt, this.inletDye, null);
    this.diffuseScalar(this.dye, this.dyeT, dt, this.spec.dyeDiffusivity, null);
    const dirichlet = (side: FaceBoundary) => side.temperature;
    if (this.spec.advection !== "none") this.advectScalar(this.temperature, this.tempT, dt, this.spec.tRef, dirichlet);
    this.diffuseScalar(this.temperature, this.tempT, dt, this.spec.alpha, dirichlet);
  }

  private advectScalar(
    field: Float64Array,
    out: Float64Array,
    dt: number,
    inletValue: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
  ): void {
    const order = this.spec.advection === "muscl-rk2" ? 2 : 1;
    const stages = this.spec.advection === "muscl-rk2" ? 2 : 1;
    this.z.set(field);
    this.scalarRhs(field, this.rhs, order, inletValue, dirichlet);
    this.r.set(this.rhs);
    if (stages === 2) {
      for (let c = 0; c < this.nC; c++) out[c] = this.solid[c] === 1 ? 0 : field[c]! + dt * this.rhs[c]!;
      this.scalarRhs(out, this.rhs, order, inletValue, dirichlet);
      for (let c = 0; c < this.nC; c++) this.rhs[c] = 0.5 * (this.r[c]! + this.rhs[c]!);
    }
    for (let c = 0; c < this.nC; c++) field[c] = this.solid[c] === 1 ? 0 : this.z[c]! + dt * this.rhs[c]!;
  }

  private scalarRhs(
    field: Float64Array,
    out: Float64Array,
    order: number,
    inletValue: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
  ): void {
    const { nx, ny, nz, boundaries: b } = this.spec;
    const { dx, dy, dz } = this;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.ic(i, j, k);
          if (this.solid[id] === 1) {
            out[id] = 0;
            continue;
          }
          const fe = this.scalarFlux("u", i + 1, j, k, field, order, inletValue, dirichlet, b.east);
          const fw = this.scalarFlux("u", i, j, k, field, order, inletValue, dirichlet, b.west);
          const fn = this.scalarFlux("v", i, j + 1, k, field, order, inletValue, dirichlet, b.north);
          const fs = this.scalarFlux("v", i, j, k, field, order, inletValue, dirichlet, b.south);
          const ff = this.scalarFlux("w", i, j, k + 1, field, order, inletValue, dirichlet, b.front);
          const fb = this.scalarFlux("w", i, j, k, field, order, inletValue, dirichlet, b.back);
          out[id] = -((fe - fw) / dx + (fn - fs) / dy + (ff - fb) / dz);
        }
      }
    }
  }

  private scalarFlux(
    axis: "u" | "v" | "w",
    i: number,
    j: number,
    k: number,
    field: Float64Array,
    order: number,
    inletValue: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
    boundary: FaceBoundary,
  ): number {
    const { nx, ny, nz } = this.spec;
    const vel = axis === "u" ? this.u[this.iu(i, j, k)]! : axis === "v" ? this.v[this.iv(i, j, k)]! : this.w[this.iw(i, j, k)]!;
    const face = axis === "u" ? i : axis === "v" ? j : k;
    const n = axis === "u" ? nx : axis === "v" ? ny : nz;
    const at = (delta: number): number => {
      const along = face - 1 + delta;
      if (axis === "u") return this.sampleScalar(field, along, j, k, inletValue, dirichlet);
      if (axis === "v") return this.sampleScalar(field, i, along, k, inletValue, dirichlet);
      return this.sampleScalar(field, i, j, along, inletValue, dirichlet);
    };
    const onBoundary = face === 0 || face === n;
    if (onBoundary && boundary.kind !== "periodic") {
      const prescribed = dirichlet ? dirichlet(boundary) : null;
      const exterior = prescribed === null ? inletValue : prescribed;
      const interiorIndex = face === 0 ? 0 : n - 1;
      const ii = axis === "u" ? interiorIndex : i;
      const jj = axis === "v" ? interiorIndex : j;
      const kk = axis === "w" ? interiorIndex : k;
      const id = this.ic(ii, jj, kk);
      const inside = this.solid[id] === 1 ? 0 : field[id]!;
      const entering =
        (axis === "u" && i === 0 && vel > 0) ||
        (axis === "u" && i === nx && vel < 0) ||
        (axis === "v" && j === 0 && vel > 0) ||
        (axis === "v" && j === ny && vel < 0) ||
        (axis === "w" && k === 0 && vel > 0) ||
        (axis === "w" && k === nz && vel < 0);
      // Zero is a real scalar. Do not replace it with another sample.
      return vel * (entering ? exterior : inside);
    }
    const low = at(0);
    const high = at(1);
    if (order < 2) return vel >= 0 ? vel * low : vel * high;
    const slopeLow = minmod(low - at(-1), high - low);
    const slopeHigh = minmod(high - low, at(2) - high);
    const left = low + 0.5 * slopeLow;
    const right = high - 0.5 * slopeHigh;
    return vel >= 0 ? vel * left : vel * right;
  }

  private sampleScalar(
    field: Float64Array,
    i: number,
    j: number,
    k: number,
    inletValue: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
  ): number {
    const { nx, ny, nz, boundaries: b } = this.spec;
    let ii = i;
    let jj = j;
    let kk = k;
    if (b.west.kind === "periodic") ii = ((ii % nx) + nx) % nx;
    if (b.south.kind === "periodic") jj = ((jj % ny) + ny) % ny;
    if (b.back.kind === "periodic") kk = ((kk % nz) + nz) % nz;
    if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) {
      const side: FaceBoundary =
        ii < 0 ? b.west : ii >= nx ? b.east : jj < 0 ? b.south : jj >= ny ? b.north : kk < 0 ? b.back : b.front;
      if (dirichlet) {
        const prescribed = dirichlet(side);
        if (prescribed === null) return inletValue;
        return 2 * prescribed - inletValue;
      }
      return inletValue;
    }
    return this.solid[this.ic(ii, jj, kk)] === 1 ? 0 : field[this.ic(ii, jj, kk)]!;
  }

  private diffuseScalar(
    field: Float64Array,
    out: Float64Array,
    dt: number,
    coeff: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
  ): void {
    if (!(coeff > 0) || !(dt > 0)) return;
    const limit = this.diffusionLimit(coeff);
    let left = dt;
    while (left > 1e-15) {
      const h = Math.min(left, limit);
      this.diffuseScalarOnce(field, out, h, coeff, dirichlet);
      left -= h;
    }
  }

  private diffuseScalarOnce(
    field: Float64Array,
    out: Float64Array,
    dt: number,
    coeff: number,
    dirichlet: ((side: FaceBoundary) => number | null) | null,
  ): void {
    const { nx, ny, nz } = this.spec;
    const { dx, dy, dz } = this;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = this.ic(i, j, k);
          if (this.solid[id] === 1) {
            out[id] = 0;
            continue;
          }
          const c = field[id]!;
          const lap =
            (this.sampleScalar(field, i - 1, j, k, c, dirichlet) - 2 * c + this.sampleScalar(field, i + 1, j, k, c, dirichlet)) / (dx * dx) +
            (this.sampleScalar(field, i, j - 1, k, c, dirichlet) - 2 * c + this.sampleScalar(field, i, j + 1, k, c, dirichlet)) / (dy * dy) +
            (this.sampleScalar(field, i, j, k - 1, c, dirichlet) - 2 * c + this.sampleScalar(field, i, j, k + 1, c, dirichlet)) / (dz * dz);
          out[id] = c + dt * coeff * lap;
        }
      }
    }
    field.set(out);
  }

  private strainAtCell(i: number, j: number, k: number): StrainSample {
    return {
      sxx: (this.u[this.iu(i + 1, j, k)]! - this.u[this.iu(i, j, k)]!) / this.dx,
      syy: (this.v[this.iv(i, j + 1, k)]! - this.v[this.iv(i, j, k)]!) / this.dy,
      szz: (this.w[this.iw(i, j, k + 1)]! - this.w[this.iw(i, j, k)]!) / this.dz,
      sxy: 0.5 * (this.dUdy(i, j, k) + this.dVdx(i, j, k)),
      sxz: 0.5 * (this.dUdz(i, j, k) + this.dWdx(i, j, k)),
      syz: 0.5 * (this.dVdz(i, j, k) + this.dWdy(i, j, k)),
    };
  }

  private strainAtFace(axis: "x" | "y" | "z", i: number, j: number, k: number): StrainSample {
    if (axis === "x") {
      const ii = Math.max(0, Math.min(this.spec.nx - 1, i - (i === this.spec.nx ? 1 : 0)));
      return this.strainAtCell(ii === i ? i : Math.max(0, i - 1), j, k);
    }
    if (axis === "y") {
      const jj = Math.max(0, Math.min(this.spec.ny - 1, j === this.spec.ny ? j - 1 : j));
      return this.strainAtCell(i, jj, k);
    }
    const kk = Math.max(0, Math.min(this.spec.nz - 1, k === this.spec.nz ? k - 1 : k));
    return this.strainAtCell(i, j, kk);
  }

  private dUdy(i: number, j: number, k: number): number {
    const south = 0.5 * (this.uGhostY(i, j - 1, k) + this.uGhostY(i + 1, j - 1, k));
    const north = 0.5 * (this.uGhostY(i, j + 1, k) + this.uGhostY(i + 1, j + 1, k));
    const here = 0.5 * (this.u[this.iu(i, j, k)]! + this.u[this.iu(i + 1, j, k)]!);
    return (0.5 * (north - here) + 0.5 * (here - south)) / this.dy;
  }
  private dUdz(i: number, j: number, k: number): number {
    const back = 0.5 * (this.uGhostZ(i, j, k - 1) + this.uGhostZ(i + 1, j, k - 1));
    const front = 0.5 * (this.uGhostZ(i, j, k + 1) + this.uGhostZ(i + 1, j, k + 1));
    const here = 0.5 * (this.u[this.iu(i, j, k)]! + this.u[this.iu(i + 1, j, k)]!);
    return (0.5 * (front - here) + 0.5 * (here - back)) / this.dz;
  }
  private dVdx(i: number, j: number, k: number): number {
    const west = 0.5 * (this.vGhostX(i - 1, j, k) + this.vGhostX(i - 1, j + 1, k));
    const east = 0.5 * (this.vGhostX(i + 1, j, k) + this.vGhostX(i + 1, j + 1, k));
    const here = 0.5 * (this.v[this.iv(i, j, k)]! + this.v[this.iv(i, j + 1, k)]!);
    return (0.5 * (east - here) + 0.5 * (here - west)) / this.dx;
  }
  private dVdz(i: number, j: number, k: number): number {
    const back = 0.5 * (this.vGhostZ(i, j, k - 1) + this.vGhostZ(i, j + 1, k - 1));
    const front = 0.5 * (this.vGhostZ(i, j, k + 1) + this.vGhostZ(i, j + 1, k + 1));
    const here = 0.5 * (this.v[this.iv(i, j, k)]! + this.v[this.iv(i, j + 1, k)]!);
    return (0.5 * (front - here) + 0.5 * (here - back)) / this.dz;
  }
  private dWdx(i: number, j: number, k: number): number {
    const west = 0.5 * (this.wGhostX(i - 1, j, k) + this.wGhostX(i - 1, j, k + 1));
    const east = 0.5 * (this.wGhostX(i + 1, j, k) + this.wGhostX(i + 1, j, k + 1));
    const here = 0.5 * (this.w[this.iw(i, j, k)]! + this.w[this.iw(i, j, k + 1)]!);
    return (0.5 * (east - here) + 0.5 * (here - west)) / this.dx;
  }
  private dWdy(i: number, j: number, k: number): number {
    const south = 0.5 * (this.wGhostY(i, j - 1, k) + this.wGhostY(i, j - 1, k + 1));
    const north = 0.5 * (this.wGhostY(i, j + 1, k) + this.wGhostY(i, j + 1, k + 1));
    const here = 0.5 * (this.w[this.iw(i, j, k)]! + this.w[this.iw(i, j, k + 1)]!);
    return (0.5 * (north - here) + 0.5 * (here - south)) / this.dy;
  }

  private save(): void {
    this.u0.set(this.u);
    this.v0.set(this.v);
    this.w0.set(this.w);
    this.phi0.set(this.phi);
    this.dye0.set(this.dye);
    this.temp0.set(this.temperature);
    this.time0 = this.time;
    this.steps0 = this.steps;
  }

  private restore(): void {
    this.releaseHeld();
    this.u.set(this.u0);
    this.v.set(this.v0);
    this.w.set(this.w0);
    this.phi.set(this.phi0);
    this.dye.set(this.dye0);
    this.temperature.set(this.temp0);
    this.time = this.time0;
    this.steps = this.steps0;
  }

  private fieldsFinite(): boolean {
    for (let i = 0; i < this.nU; i++) if (!Number.isFinite(this.u[i]!)) return false;
    for (let i = 0; i < this.nV; i++) if (!Number.isFinite(this.v[i]!)) return false;
    for (let i = 0; i < this.nW; i++) if (!Number.isFinite(this.w[i]!)) return false;
    for (let i = 0; i < this.nC; i++) {
      if (!Number.isFinite(this.phi[i]!) || !Number.isFinite(this.temperature[i]!) || !Number.isFinite(this.dye[i]!)) return false;
    }
    return true;
  }

  private report(
    status: StepStatus,
    dt: number,
    attempted: number,
    accepted: number,
    substeps: number,
    pressureIterations: number,
    pressureResidual: number,
    pressureConverged: boolean,
    detail: string,
  ): StepReport {
    const norms = this.divergenceNorms();
    const fluxes = this.outwardFluxes();
    let imbalance = 0;
    for (const side of SIDES) imbalance += fluxes[side];
    let maxSpeed = 0;
    for (let i = 0; i < this.nU; i++) maxSpeed = Math.max(maxSpeed, Math.abs(this.u[i]!));
    for (let i = 0; i < this.nV; i++) maxSpeed = Math.max(maxSpeed, Math.abs(this.v[i]!));
    for (let i = 0; i < this.nW; i++) maxSpeed = Math.max(maxSpeed, Math.abs(this.w[i]!));
    return {
      status,
      time: this.time,
      dt,
      attempted,
      accepted,
      substeps,
      kineticEnergy: this.kineticEnergy(),
      divL2: norms.l2,
      divLinf: norms.linf,
      massImbalance: imbalance,
      fluxes,
      pressureIterations,
      pressureResidual,
      pressureConverged,
      dissipation: this.dissipationPower(),
      maxSpeed,
      detail,
    };
  }

  private blank(status: StepStatus, time: number, dt: number, detail: string): StepReport {
    void time;
    return this.report(status, dt, 0, 0, 0, 0, 0, status === "ok", detail);
  }
}

function isVelocityBC(side: FaceBoundary): boolean {
  return side.kind === "wall" || side.kind === "inlet" || side.kind === "lid";
}

function dirichletTangential(side: FaceBoundary): boolean {
  return side.kind === "wall" || side.kind === "inlet" || side.kind === "lid";
}

function minmod(a: number, b: number): number {
  if (a * b <= 0) return 0;
  return Math.abs(a) < Math.abs(b) ? a : b;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}
