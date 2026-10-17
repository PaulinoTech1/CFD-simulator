import {
  F_BODY,
  F_INLET,
  F_LID,
  F_SOLID,
  type Scene,
  type SimParams,
  type SimStats,
  type Splat,
} from "./types";

function isBlocked(flag: number): boolean {
  return flag === F_SOLID || flag === F_BODY;
}

function sample(
  field: Float32Array,
  x: number,
  y: number,
  nx: number,
  ny: number,
  periodicX: boolean,
): number {
  let xx = x;
  if (periodicX) {
    xx = xx % nx;
    if (xx < 0) xx += nx;
  } else {
    if (xx < 0) xx = 0;
    else if (xx > nx - 1) xx = nx - 1;
  }
  let yy = y;
  if (yy < 0) yy = 0;
  else if (yy > ny - 1) yy = ny - 1;

  const i0 = Math.floor(xx);
  const j0 = Math.floor(yy);
  const tx = xx - i0;
  const ty = yy - j0;
  let i1 = i0 + 1;
  let j1 = j0 + 1;
  if (j1 >= ny) j1 = ny - 1;
  if (periodicX) {
    if (i1 >= nx) i1 = 0;
  } else if (i1 >= nx) i1 = nx - 1;

  const n = nx;
  const a = field[i0 + j0 * n]!;
  const b = field[i1 + j0 * n]!;
  const c = field[i0 + j1 * n]!;
  const d = field[i1 + j1 * n]!;
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

const EMPTY_STATS: SimStats = {
  time: 0,
  dt: 0,
  ke: 0,
  maxSpeed: 0,
  maxAbsOmega: 0,
  fx: 0,
  fy: 0,
  cd: 0,
  cl: 0,
  re: 0,
  cfl: 0,
  strouhal: 0,
  steps: 0,
};

/**
 * 2D incompressible Navier–Stokes, collocated Chorin projection.
 * Cell size is 1. The same three stages (advect, diffuse, project) are what
 * the sm_86 kernels implement; this is the reference that runs in the browser.
 */
export class FluidSim {
  nx = 0;
  ny = 0;
  flags = new Uint8Array(0);
  u = new Float32Array(0);
  v = new Float32Array(0);
  dye = new Float32Array(0);
  p = new Float32Array(0);
  omega = new Float32Array(0);
  periodicX = false;
  charLength = 1;
  uIn = 0;
  hasBody = false;
  dyeInlet = false;
  inject: Scene["inject"] = [];

  px = new Float32Array(0);
  py = new Float32Array(0);
  ox = new Float32Array(0);
  oy = new Float32Array(0);

  time = 0;
  steps = 0;
  strouhal = 0;

  private uT = new Float32Array(0);
  private vT = new Float32Array(0);
  private dyeT = new Float32Array(0);
  private hold = new Float32Array(0);
  private div = new Float32Array(0);
  private maxSpeed = 1;
  private prevFy = 0;
  private lastCross = -1;
  private ready = false;

  get particleCount(): number {
    return this.px.length;
  }

  load(scene: Scene): void {
    const n = scene.nx * scene.ny;
    this.nx = scene.nx;
    this.ny = scene.ny;
    this.flags = scene.flags.slice();
    this.u = scene.u.slice();
    this.v = scene.v.slice();
    this.dye = scene.dye.slice();
    this.uT = new Float32Array(n);
    this.vT = new Float32Array(n);
    this.dyeT = new Float32Array(n);
    this.hold = new Float32Array(n);
    this.p = new Float32Array(n);
    this.div = new Float32Array(n);
    this.omega = new Float32Array(n);
    this.periodicX = scene.periodicX;
    this.charLength = Math.max(1, scene.charLength);
    this.uIn = scene.uIn;
    this.hasBody = scene.hasBody;
    this.dyeInlet = scene.dyeInlet;
    this.inject = scene.inject;
    this.time = 0;
    this.steps = 0;
    this.strouhal = 0;
    this.prevFy = 0;
    this.lastCross = -1;
    this.maxSpeed = Math.max(1, Math.abs(scene.uIn));
    this.ready = true;
    this.seedParticles();
    this.enforce(scene.uIn);
    this.ghost();
  }

  cflDt(cfl = 0.8): number {
    return cfl / Math.max(this.maxSpeed, 1e-3);
  }

  snapshotParticles(): void {
    this.ox.set(this.px);
    this.oy.set(this.py);
  }

  step(dtIn: number, params: SimParams, splats: readonly Splat[]): SimStats {
    if (!this.ready || dtIn <= 0) return { ...EMPTY_STATS, time: this.time };
    const dt = Math.min(dtIn, this.cflDt(0.85));
    const nx = this.nx;
    const ny = this.ny;
    const refU = Math.max(Math.abs(params.uIn), 6);
    const nu = (refU * this.charLength) / Math.max(params.re, 1);
    this.uIn = params.uIn;

    this.forces(dt, params, splats);
    this.enforce(params.uIn);
    this.ghost();
    this.advectVelocity(dt);
    this.enforce(params.uIn);
    this.diffuse(dt, nu);
    this.enforce(params.uIn);
    this.project(dt, params.iterations);
    this.enforce(params.uIn);
    this.ghost();
    this.advectDye(dt);
    this.sources(dt, params.dyeDecay);
    this.measure(dt);

    this.time += dt;
    this.steps += 1;
    const q = 0.5 * params.uIn * params.uIn * this.charLength;
    const cd = this.hasBody && q > 1e-6 ? this._fx / q : 0;
    const cl = this.hasBody && q > 1e-6 ? this._fy / q : 0;
    this.trackStrouhal(this._fy, params.uIn);

    return {
      time: this.time,
      dt,
      ke: this._ke,
      maxSpeed: this.maxSpeed,
      maxAbsOmega: this._maxW,
      fx: this._fx,
      fy: this._fy,
      cd,
      cl,
      re: params.re,
      cfl: this.maxSpeed * dt,
      strouhal: this.strouhal,
      steps: this.steps,
    };
  }

  private _ke = 0;
  private _maxW = 0;
  private _fx = 0;
  private _fy = 0;

  paint(i: number, j: number, solid: boolean, radius: number): void {
    const nx = this.nx;
    const ny = this.ny;
    const r2 = radius * radius;
    for (let y = Math.max(1, j - radius); y <= Math.min(ny - 2, j + radius); y++) {
      for (let x = Math.max(1, i - radius); x <= Math.min(nx - 2, i + radius); x++) {
        const dx = x - i;
        const dy = y - j;
        if (dx * dx + dy * dy > r2) continue;
        const id = x + y * nx;
        const f = this.flags[id]!;
        if (f === F_INLET || f === F_LID) continue;
        if (solid) {
          this.flags[id] = F_BODY;
          this.u[id] = 0;
          this.v[id] = 0;
          this.dye[id] = 0;
        } else if (f === F_BODY || f === F_SOLID) {
          this.flags[id] = 0;
        }
      }
    }
    this.hasBody = true;
  }

  private seedParticles(): void {
    const count = Math.min(1800, Math.max(600, Math.floor(this.nx * this.ny * 0.04)));
    this.px = new Float32Array(count);
    this.py = new Float32Array(count);
    this.ox = new Float32Array(count);
    this.oy = new Float32Array(count);
    for (let n = 0; n < count; n++) this.respawn(n, true);
    this.ox.set(this.px);
    this.oy.set(this.py);
  }

  private respawn(n: number, anywhere: boolean): void {
    const nx = this.nx;
    const ny = this.ny;
    if (!anywhere && !this.periodicX && this.uIn !== 0) {
      const y = 1 + Math.random() * (ny - 3);
      this.px[n] = 1.5;
      this.py[n] = y;
      return;
    }
    for (let k = 0; k < 8; k++) {
      const x = Math.random() * (nx - 2) + 1;
      const y = Math.random() * (ny - 2) + 1;
      const id = (x | 0) + (y | 0) * nx;
      if (!isBlocked(this.flags[id]!)) {
        this.px[n] = x;
        this.py[n] = y;
        return;
      }
    }
    this.px[n] = nx * 0.5;
    this.py[n] = ny * 0.5;
  }

  private enforce(uIn: number): void {
    const { u, v, flags, nx, ny } = this;
    const kick = Math.sin(this.time * 2.7) * uIn * 0.035 * Math.exp(-this.time * 0.12);
    for (let j = 0; j < ny; j++) {
      const row = j * nx;
      for (let i = 0; i < nx; i++) {
        const id = row + i;
        const f = flags[id]!;
        if (isBlocked(f)) {
          u[id] = 0;
          v[id] = 0;
        } else if (f === F_INLET) {
          u[id] = uIn;
          const mid = Math.abs(j - ny * 0.5) < ny * 0.18;
          v[id] = mid ? kick : 0;
        } else if (f === F_LID) {
          u[id] = uIn;
          v[id] = 0;
        }
      }
    }
  }

  private ghost(): void {
    if (this.periodicX) return;
    const { u, v, dye, p, nx, ny } = this;
    for (let j = 0; j < ny; j++) {
      const row = j * nx;
      const last = row + nx - 1;
      const prev = row + nx - 2;
      u[last] = u[prev]!;
      v[last] = v[prev]!;
      dye[last] = dye[prev]!;
      p[last] = p[prev]!;
      u[row] = u[row + 1]!;
      v[row] = v[row + 1]!;
      dye[row] = dye[row + 1]!;
      p[row] = p[row + 1]!;
    }
  }

  private forces(dt: number, params: SimParams, splats: readonly Splat[]): void {
    const { u, v, dye, omega, flags, nx, ny } = this;
    const eps = params.vorticity;
    const buoy = params.buoyancy;
    if (eps > 0 || buoy !== 0) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + j * nx;
          if (isBlocked(flags[id]!)) continue;
          if (buoy !== 0) v[id] = v[id]! + dt * buoy * dye[id]!;
          if (eps > 0) {
            const wL = Math.abs(omega[id - 1]!);
            const wR = Math.abs(omega[id + 1]!);
            const wB = Math.abs(omega[id - nx]!);
            const wT = Math.abs(omega[id + nx]!);
            let nxg = 0.5 * (wR - wL);
            let nyg = 0.5 * (wT - wB);
            const mag = Math.hypot(nxg, nyg);
            if (mag > 1e-4) {
              nxg /= mag;
              nyg /= mag;
              const w = omega[id]!;
              const scale = eps * dt;
              u[id] = u[id]! + scale * nyg * w;
              v[id] = v[id]! - scale * nxg * w;
            }
          }
        }
      }
    }
    for (let s = 0; s < splats.length; s++) {
      const sp = splats[s]!;
      const rad = sp.radius;
      const i0 = Math.max(1, Math.floor(sp.x - rad));
      const i1 = Math.min(nx - 2, Math.ceil(sp.x + rad));
      const j0 = Math.max(1, Math.floor(sp.y - rad));
      const j1 = Math.min(ny - 2, Math.ceil(sp.y + rad));
      const inv = 1 / (rad * rad);
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const id = i + j * nx;
          if (isBlocked(flags[id]!) || flags[id] === F_INLET || flags[id] === F_LID) continue;
          const dx = i - sp.x;
          const dy = j - sp.y;
          const wgt = Math.exp(-(dx * dx + dy * dy) * inv);
          u[id] = u[id]! + sp.du * wgt;
          v[id] = v[id]! + sp.dv * wgt;
          if (sp.dye > 0) dye[id] = Math.min(1, dye[id]! + sp.dye * wgt);
        }
      }
    }
  }

  private advectVelocity(dt: number): void {
    const { u, v, uT, vT, flags, nx, ny, periodicX } = this;
    const i0 = periodicX ? 0 : 1;
    const i1 = periodicX ? nx : nx - 1;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = i0; i < i1; i++) {
        const id = i + j * nx;
        const f = flags[id]!;
        if (isBlocked(f) || f === F_INLET || f === F_LID) {
          uT[id] = u[id]!;
          vT[id] = v[id]!;
          continue;
        }
        const uu0 = u[id]!;
        const vv0 = v[id]!;
        let bx = dt * uu0;
        let by = dt * vv0;
        const mag = Math.hypot(bx, by);
        if (mag > 0.85) {
          const s = 0.85 / mag;
          bx *= s;
          by *= s;
        }
        const x = i - bx;
        const y = j - by;
        uT[id] = sample(u, x, y, nx, ny, periodicX);
        vT[id] = sample(v, x, y, nx, ny, periodicX);
      }
    }
    const swapU = this.u;
    this.u = uT;
    this.uT = swapU;
    const swapV = this.v;
    this.v = vT;
    this.vT = swapV;
    this.ghost();
  }

  private advectDye(dt: number): void {
    const { u, v, dye, dyeT, flags, nx, ny, periodicX } = this;
    const i0 = periodicX ? 0 : 1;
    const i1 = periodicX ? nx : nx - 1;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = i0; i < i1; i++) {
        const id = i + j * nx;
        if (isBlocked(flags[id]!)) {
          dyeT[id] = 0;
          continue;
        }
        const uu0 = u[id]!;
        const vv0 = v[id]!;
        let bx = dt * uu0;
        let by = dt * vv0;
        const mag = Math.hypot(bx, by);
        if (mag > 0.85) {
          const s = 0.85 / mag;
          bx *= s;
          by *= s;
        }
        const x = i - bx;
        const y = j - by;
        dyeT[id] = sample(dye, x, y, nx, ny, periodicX);
      }
    }
    const swap = this.dye;
    this.dye = dyeT;
    this.dyeT = swap;
    this.ghost();
  }

  private diffuse(dt: number, nu: number): void {
    const a = dt * nu;
    if (a < 1e-5) return;
    const { hold } = this;
    const iters = a > 0.2 ? 6 : 3;
    hold.set(this.u);
    this.diffuseField(this.u, this.uT, hold, a, iters, true);
    hold.set(this.v);
    this.diffuseField(this.v, this.vT, hold, a, iters, true);
    this.ghost();
  }

  private diffuseField(
    field: Float32Array,
    temp: Float32Array,
    rhs: Float32Array,
    a: number,
    iters: number,
    dirichletWall: boolean,
  ): void {
    const { flags, nx, ny, periodicX } = this;
    const denom = 1 + 4 * a;
    const i0 = periodicX ? 0 : 1;
    const i1 = periodicX ? nx : nx - 1;
    let cur = field;
    let nxt = temp;
    for (let k = 0; k < iters; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = i0; i < i1; i++) {
          const id = i + j * nx;
          const f = flags[id]!;
          if (isBlocked(f) || f === F_INLET || f === F_LID) {
            nxt[id] = cur[id]!;
            continue;
          }
          const im = periodicX ? (i === 0 ? nx - 1 : i - 1) : i - 1;
          const ip = periodicX ? (i === nx - 1 ? 0 : i + 1) : i + 1;
          const idL = im + j * nx;
          const idR = ip + j * nx;
          const idB = i + (j - 1) * nx;
          const idT = i + (j + 1) * nx;
          const uL = dirichletWall && isBlocked(flags[idL]!) ? 0 : cur[idL]!;
          const uR = dirichletWall && isBlocked(flags[idR]!) ? 0 : cur[idR]!;
          const uB = dirichletWall && isBlocked(flags[idB]!) ? 0 : cur[idB]!;
          const uT = dirichletWall && isBlocked(flags[idT]!) ? 0 : cur[idT]!;
          nxt[id] = (rhs[id]! + a * (uL + uR + uB + uT)) / denom;
        }
      }
      const swap = cur;
      cur = nxt;
      nxt = swap;
    }
    if (cur !== field) field.set(cur);
  }

  private project(dt: number, iterations: number): void {
    const { flags, nx, ny, periodicX, div } = this;
    const u = this.u;
    const v = this.v;
    const invDt = 1 / Math.max(dt, 1e-6);
    const i0 = periodicX ? 0 : 1;
    const i1 = periodicX ? nx : nx - 1;
    const omega = 1.62;

    for (let j = 1; j < ny - 1; j++) {
      for (let i = i0; i < i1; i++) {
        const id = i + j * nx;
        if (isBlocked(flags[id]!)) {
          div[id] = 0;
          continue;
        }
        const im = periodicX ? (i === 0 ? nx - 1 : i - 1) : i - 1;
        const ip = periodicX ? (i === nx - 1 ? 0 : i + 1) : i + 1;
        const uL = isBlocked(flags[im + j * nx]!) ? 0 : u[im + j * nx]!;
        const uR = isBlocked(flags[ip + j * nx]!) ? 0 : u[ip + j * nx]!;
        const vB = isBlocked(flags[i + (j - 1) * nx]!) ? 0 : v[i + (j - 1) * nx]!;
        const vT = isBlocked(flags[i + (j + 1) * nx]!) ? 0 : v[i + (j + 1) * nx]!;
        div[id] = 0.5 * (uR - uL + (vT - vB));
      }
    }

    const p = this.p;
    const iters = Math.max(4, iterations | 0);
    for (let k = 0; k < iters; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = i0; i < i1; i++) {
          const id = i + j * nx;
          if (isBlocked(flags[id]!)) {
            p[id] = 0;
            continue;
          }
          const im = periodicX ? (i === 0 ? nx - 1 : i - 1) : i - 1;
          const ip = periodicX ? (i === nx - 1 ? 0 : i + 1) : i + 1;
          const idL = im + j * nx;
          const idR = ip + j * nx;
          const idB = i + (j - 1) * nx;
          const idT = i + (j + 1) * nx;
          let sum = 0;
          let nFluid = 0;
          if (!isBlocked(flags[idL]!)) {
            sum += p[idL]!;
            nFluid++;
          }
          if (!isBlocked(flags[idR]!)) {
            sum += p[idR]!;
            nFluid++;
          }
          if (!isBlocked(flags[idB]!)) {
            sum += p[idB]!;
            nFluid++;
          }
          if (!isBlocked(flags[idT]!)) {
            sum += p[idT]!;
            nFluid++;
          }
          if (nFluid === 0) {
            p[id] = 0;
            continue;
          }
          const gs = (sum - div[id]! * invDt) / nFluid;
          p[id] = (1 - omega) * p[id]! + omega * gs;
        }
      }
    }

    for (let j = 1; j < ny - 1; j++) {
      for (let i = i0; i < i1; i++) {
        const id = i + j * nx;
        const f = flags[id]!;
        if (isBlocked(f)) continue;
        const im = periodicX ? (i === 0 ? nx - 1 : i - 1) : i - 1;
        const ip = periodicX ? (i === nx - 1 ? 0 : i + 1) : i + 1;
        const idL = im + j * nx;
        const idR = ip + j * nx;
        const idB = i + (j - 1) * nx;
        const idT = i + (j + 1) * nx;
        const pc = p[id]!;
        const pL = isBlocked(flags[idL]!) ? pc : p[idL]!;
        const pR = isBlocked(flags[idR]!) ? pc : p[idR]!;
        const pB = isBlocked(flags[idB]!) ? pc : p[idB]!;
        const pT = isBlocked(flags[idT]!) ? pc : p[idT]!;
        if (f !== F_INLET && f !== F_LID) {
          u[id] = u[id]! - dt * 0.5 * (pR - pL);
          v[id] = v[id]! - dt * 0.5 * (pT - pB);
        }
      }
    }
  }

  private sources(dt: number, decay: number): void {
    const { dye, flags, nx, ny, inject } = this;
    if (decay > 0) {
      const keep = Math.max(0, 1 - decay * dt);
      for (let i = 0; i < dye.length; i++) dye[i] = dye[i]! * keep;
    }
    for (let n = 0; n < inject.length; n++) {
      const s = inject[n]!;
      if (s.i < 0 || s.j < 0 || s.i >= nx || s.j >= ny) continue;
      const id = s.i + s.j * nx;
      if (isBlocked(flags[id]!)) continue;
      dye[id] = Math.min(1, dye[id]! + s.rate * dt);
    }
    if (this.dyeInlet) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = i + j * nx;
          if (flags[id] === F_INLET) dye[id] = 1;
        }
      }
    }
    for (let i = 0; i < flags.length; i++) {
      if (isBlocked(flags[i]!)) dye[i] = 0;
    }
  }

  private measure(dt: number): void {
    const { u, v, flags, p, nx, ny, periodicX } = this;
    let ke = 0;
    let maxS = 1e-4;
    let maxW = 0;
    let fx = 0;
    let fy = 0;
    const i0 = periodicX ? 0 : 1;
    const i1 = periodicX ? nx : nx - 1;
    const omega = this.omega;

    for (let j = 1; j < ny - 1; j++) {
      for (let i = i0; i < i1; i++) {
        const id = i + j * nx;
        const f = flags[id]!;
        if (isBlocked(f)) {
          omega[id] = 0;
          if (f === F_BODY) {
            const im = i > 0 ? i - 1 : i;
            const ip = i < nx - 1 ? i + 1 : i;
            const jm = j > 0 ? j - 1 : j;
            const jp = j < ny - 1 ? j + 1 : j;
            const left = flags[im + j * nx]!;
            const right = flags[ip + j * nx]!;
            const below = flags[i + jm * nx]!;
            const above = flags[i + jp * nx]!;
            if (!isBlocked(left)) fx += p[im + j * nx]!;
            if (!isBlocked(right)) fx -= p[ip + j * nx]!;
            if (!isBlocked(below)) fy += p[i + jm * nx]!;
            if (!isBlocked(above)) fy -= p[i + jp * nx]!;
          }
          continue;
        }
        const uu = u[id]!;
        const vv = v[id]!;
        if (!Number.isFinite(uu) || !Number.isFinite(vv)) {
          u[id] = 0;
          v[id] = 0;
          continue;
        }
        ke += uu * uu + vv * vv;
        const sp = Math.hypot(uu, vv);
        if (sp > maxS) maxS = sp;
        const im = periodicX ? (i === 0 ? nx - 1 : i - 1) : i - 1;
        const ip = periodicX ? (i === nx - 1 ? 0 : i + 1) : i + 1;
        const dvdx = 0.5 * ((v[ip + j * nx] ?? 0) - (v[im + j * nx] ?? 0));
        const dudy = 0.5 * ((u[i + (j + 1) * nx] ?? 0) - (u[i + (j - 1) * nx] ?? 0));
        const w = dvdx - dudy;
        omega[id] = w;
        const aw = Math.abs(w);
        if (aw > maxW) maxW = aw;
      }
    }

    const count = this.px.length;
    for (let n = 0; n < count; n++) {
      const x = this.px[n]!;
      const y = this.py[n]!;
      const uu = sample(u, x, y, nx, ny, periodicX);
      const vv = sample(v, x, y, nx, ny, periodicX);
      let xn = x + uu * dt;
      let yn = y + vv * dt;
      if (periodicX) {
        xn = xn % nx;
        if (xn < 0) xn += nx;
      }
      const xi = xn | 0;
      const yi = yn | 0;
      const inside = yi >= 1 && yi <= ny - 2 && xi >= 0 && xi < nx;
      const blocked = !inside || isBlocked(flags[xi + yi * nx]!);
      const leave = !periodicX && (xn < 1 || xn > nx - 2);
      if (leave || blocked || yn < 1 || yn > ny - 2 || !Number.isFinite(xn)) this.respawn(n, false);
      else {
        this.px[n] = xn;
        this.py[n] = yn;
      }
    }

    this._ke = 0.5 * ke;
    this._maxW = maxW;
    this._fx = fx;
    this._fy = fy;
    this.maxSpeed = maxS;
  }

  private trackStrouhal(fy: number, uIn: number): void {
    if (!this.hasBody || Math.abs(uIn) < 1e-3) {
      this.strouhal = 0;
      this.prevFy = fy;
      return;
    }
    if (this.prevFy < 0 && fy >= 0) {
      if (this.lastCross > 0) {
        const period = this.time - this.lastCross;
        if (period > 0.05) this.strouhal = this.charLength / (period * Math.abs(uIn));
      }
      this.lastCross = this.time;
    }
    this.prevFy = fy;
  }
}
