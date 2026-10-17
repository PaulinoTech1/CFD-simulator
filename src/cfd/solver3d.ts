import type { BodyKind, Scene3, Splat3 } from "./presets3d";
import { F_BODY, F_FLUID, F_INLET, F_LID, F_OUTLET, F_SOLID, type SimParams, type SimStats } from "./types";

function blocked(flag: number): boolean {
  return flag === F_SOLID || flag === F_BODY;
}

const EMPTY: SimStats = {
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

export class Fluid3 {
  nx = 0;
  ny = 0;
  nz = 0;
  flags = new Uint8Array(0);
  u = new Float32Array(0);
  v = new Float32Array(0);
  w = new Float32Array(0);
  dye = new Float32Array(0);
  temp = new Float32Array(0);
  tAmb = 20;
  tWall = 20;
  p = new Float32Array(0);
  px = new Float32Array(0);
  py = new Float32Array(0);
  pz = new Float32Array(0);
  bodyKind: Scene3["bodyKind"] = "none";
  hasBody = false;
  cx = 0;
  cy = 0;
  cz = 0;
  radius = 1;
  aoa = 0;
  charLength = 1;
  divResidual = 0;
  dissipation = 0;
  inletFlux = 0;
  circulation = 0;
  reflectionDefect = 0;
  solidComponents = 0;
  time = 0;
  steps = 0;
  strouhal = 0;

  private uT = new Float32Array(0);
  private vT = new Float32Array(0);
  private wT = new Float32Array(0);
  private dyeT = new Float32Array(0);
  private mag = new Float32Array(0);
  private div = new Float32Array(0);
  uIn = 1;
  maxSpeed = 1;
  ke = 0;
  private maxW = 0;
  private fx = 0;
  private fy = 0;
  private prevFy = 0;
  private lastCross = -1;
  private ready = false;
  private seed = 1;

  get particleCount(): number {
    return this.px.length;
  }

  load(scene: Scene3): void {
    const n = scene.nx * scene.ny * scene.nz;
    this.nx = scene.nx;
    this.ny = scene.ny;
    this.nz = scene.nz;
    this.flags = scene.flags.slice();
    this.u = scene.u.slice();
    this.v = scene.v.slice();
    this.w = scene.w.slice();
    this.dye = scene.dye.slice();
    this.temp = new Float32Array(n);
    this.tAmb = 20;
    for (let id = 0; id < n; id++) this.temp[id] = scene.dye[id]! > 0.5 ? 70 : 20;
    this.uT = new Float32Array(n);
    this.vT = new Float32Array(n);
    this.wT = new Float32Array(n);
    this.dyeT = new Float32Array(n);
    this.mag = new Float32Array(n);
    this.p = new Float32Array(n);
    this.div = new Float32Array(n);
    this.uIn = scene.uIn;
    this.bodyKind = scene.bodyKind;
    this.hasBody = scene.hasBody;
    this.cx = scene.cx;
    this.cy = scene.cy;
    this.cz = scene.cz;
    this.radius = scene.radius;
    this.aoa = scene.aoa;
    this.charLength = Math.max(1, scene.charLength);
    this.time = 0;
    this.steps = 0;
    this.strouhal = 0;
    this.prevFy = 0;
    this.lastCross = -1;
    this.maxSpeed = Math.max(0.2, Math.abs(scene.uIn));
    this.ke = 0;
    this.divResidual = 0;
    this.dissipation = 0;
    this.inletFlux = 0;
    this.circulation = 0;
    this.reflectionDefect = 0;
    this.solidComponents = this.countBodies();
    this.ready = true;
    this.seedParticles();
    this.enforce(scene.uIn);
  }

  cflDt(cfl = 0.7): number {
    const adv = cfl / Math.max(this.maxSpeed, 0.15);
    return Math.min(0.45, adv);
  }

  splat(s: Splat3): void {
    const { nx, ny, nz, flags, u, v, w, dye } = this;
    const r = Math.max(1.5, s.radius);
    const r2 = r * r;
    const i0 = Math.max(1, Math.floor(s.x - r));
    const i1 = Math.min(nx - 2, Math.ceil(s.x + r));
    const j0 = Math.max(1, Math.floor(s.y - r));
    const j1 = Math.min(ny - 2, Math.ceil(s.y + r));
    const k0 = Math.max(1, Math.floor(s.z - r));
    const k1 = Math.min(nz - 2, Math.ceil(s.z + r));
    for (let k = k0; k <= k1; k++) {
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const dx = i - s.x;
          const dy = j - s.y;
          const dz = k - s.z;
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 > r2) continue;
          const id = i + nx * (j + ny * k);
          if (blocked(flags[id]!)) continue;
          const g = Math.exp(-d2 / (r2 * 0.45));
          u[id] = u[id]! + s.du * g;
          v[id] = v[id]! + s.dv * g;
          w[id] = w[id]! + s.dw * g;
          dye[id] = Math.min(1, dye[id]! + s.dye * g);
        }
      }
    }
  }

  step(dtIn: number, params: SimParams, splats: readonly Splat3[]): SimStats {
    if (!this.ready || dtIn <= 0) return { ...EMPTY, time: this.time };
    const dt = Math.min(dtIn, this.cflDt(0.7));
    const refU = Math.max(Math.abs(params.uIn), 0.35);
    const nu = (refU * this.charLength) / Math.max(params.re, 1);
    this.uIn = params.uIn;
    for (let i = 0; i < splats.length; i++) this.splat(splats[i]!);
    this.drive(dt, params.gradP ?? 0);
    this.thermal(dt, params);
    this.buoy(dt, params.buoyancy, params.tAmb ?? this.tAmb);
    if (params.vorticity > 0.01) this.confine(dt, params.vorticity);
    this.enforce(params.uIn);
    this.advect(dt);
    this.enforce(params.uIn);
    this.diffuse(dt, nu);
    this.enforce(params.uIn);
    this.project(dt, params.iterations);
    this.enforce(params.uIn);
    this.advectDye(dt);
    this.decayDye(dt, params.dyeDecay);
    this.measure(dt, nu);

    this.time += dt;
    this.steps += 1;
    const area = !this.hasBody ? 1 : this.charLength * Math.max(1, this.nz - 2);
    const q = 0.5 * refU * refU * area;
    const cd = this.hasBody && q > 1e-8 ? this.fx / q : 0;
    const cl = this.hasBody && q > 1e-8 ? this.fy / q : 0;
    this.trackStrouhal(this.fy, refU);
    return {
      time: this.time,
      dt,
      ke: this.ke,
      maxSpeed: this.maxSpeed,
      maxAbsOmega: this.maxW,
      fx: this.fx,
      fy: this.fy,
      cd,
      cl,
      re: params.re,
      cfl: this.maxSpeed * dt,
      strouhal: this.strouhal,
      steps: this.steps,
    };
  }

  sample(field: Float32Array, x: number, y: number, z: number): number {
    const { nx, ny, nz } = this;
    const xx = x < 0 ? 0 : x > nx - 1 ? nx - 1 : x;
    const yy = y < 0 ? 0 : y > ny - 1 ? ny - 1 : y;
    const zz = z < 0 ? 0 : z > nz - 1 ? nz - 1 : z;
    const i0 = xx | 0;
    const j0 = yy | 0;
    const k0 = zz | 0;
    const i1 = i0 + 1 < nx ? i0 + 1 : i0;
    const j1 = j0 + 1 < ny ? j0 + 1 : j0;
    const k1 = k0 + 1 < nz ? k0 + 1 : k0;
    const tx = xx - i0;
    const ty = yy - j0;
    const tz = zz - k0;
    const nxy = nx * ny;
    const s = (i: number, j: number, k: number) => field[i + nx * j + nxy * k] ?? 0;
    const c00 = s(i0, j0, k0) * (1 - tx) + s(i1, j0, k0) * tx;
    const c10 = s(i0, j1, k0) * (1 - tx) + s(i1, j1, k0) * tx;
    const c01 = s(i0, j0, k1) * (1 - tx) + s(i1, j0, k1) * tx;
    const c11 = s(i0, j1, k1) * (1 - tx) + s(i1, j1, k1) * tx;
    return (c00 * (1 - ty) + c10 * ty) * (1 - tz) + (c01 * (1 - ty) + c11 * ty) * tz;
  }

  private rand(): number {
    this.seed = (this.seed * 1664525 + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }

  private seedParticles(): void {
    const count = Math.min(640, Math.max(180, (this.nx * this.ny) >> 1));
    this.px = new Float32Array(count);
    this.py = new Float32Array(count);
    this.pz = new Float32Array(count);
    for (let n = 0; n < count; n++) this.respawn(n, true);
  }

  private respawn(n: number, anywhere: boolean): void {
    const { nx, ny, nz, flags } = this;
    for (let attempt = 0; attempt < 8; attempt++) {
      const i = anywhere ? 1 + this.rand() * (nx - 2) : 1 + this.rand() * Math.min(4, nx * 0.15);
      const j = 1 + this.rand() * (ny - 2);
      const k = 1 + this.rand() * (nz - 2);
      const id = (i | 0) + nx * ((j | 0) + ny * (k | 0));
      if (!blocked(flags[id] ?? F_SOLID)) {
        this.px[n] = i;
        this.py[n] = j;
        this.pz[n] = k;
        return;
      }
    }
    this.px[n] = 1.5;
    this.py[n] = ny * 0.5;
    this.pz[n] = nz * 0.5;
  }

  private enforce(uIn: number): void {
    const { flags, u, v, w, nx, ny, nz } = this;
    const nxy = nx * ny;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = i + nx * j + nxy * k;
          const f = flags[id]!;
          if (blocked(f)) {
            u[id] = 0;
            v[id] = 0;
            w[id] = 0;
          } else if (f === F_INLET || f === F_LID) {
            u[id] = uIn;
            v[id] = 0;
            w[id] = 0;
          } else if (f === F_OUTLET && i > 0) {
            const up = id - 1;
            u[id] = u[up]!;
            v[id] = v[up]!;
            w[id] = w[up]!;
          }
        }
      }
    }
  }

  reshape(kind: BodyKind, size: number, at?: { x: number; y: number; z: number }): void {
    const x = at?.x ?? this.cx;
    const y = at?.y ?? this.cy;
    const z = at?.z ?? this.cz;
    this.clearBodies();
    this.stamp(kind, x, y, z, Math.max(1.2, size));
  }

  stamp(kind: BodyKind, x: number, y: number, z: number, size: number): void {
    if (kind === "none") return;
    const { nx, ny, nz, flags, u, v, w, temp } = this;
    const r = Math.max(1.2, size);
    const i0 = Math.max(1, Math.floor(x - r * 2));
    const i1 = Math.min(nx - 2, Math.ceil(x + r * 2));
    const j0 = Math.max(1, Math.floor(y - r * 2));
    const j1 = Math.min(ny - 2, Math.ceil(y + r * 2));
    const k0 = Math.max(1, Math.floor(z - r * 2));
    const k1 = Math.min(nz - 2, Math.ceil(z + r * 2));
    const aoa = (this.aoa * Math.PI) / 180;
    for (let k = k0; k <= k1; k++) {
      for (let j = j0; j <= j1; j++) {
        for (let i = i0; i <= i1; i++) {
          const id = i + nx * (j + ny * k);
          const f = flags[id]!;
          if (f === F_SOLID || f === F_INLET || f === F_OUTLET || f === F_LID) continue;
          const dx = i - x;
          const dy = j - y;
          const dz = k - z;
          let inside = false;
          if (kind === "sphere") inside = dx * dx + dy * dy + dz * dz <= r * r;
          else if (kind === "cylinder") inside = dx * dx + dy * dy <= r * r && Math.abs(dz) <= r * 1.6;
          else if (kind === "box") inside = Math.abs(dx) <= r && Math.abs(dy) <= r * 0.7 && Math.abs(dz) <= r;
          else if (kind === "plate") inside = Math.abs(dx) <= 0.7 && Math.abs(dy) <= r * 1.4 && Math.abs(dz) <= r * 1.2;
          else if (kind === "cone") {
            const t = (dx + r) / (2 * r);
            if (t >= 0 && t <= 1) {
              const rad = r * 0.85 * (1 - t);
              inside = dy * dy + dz * dz <= rad * rad;
            }
          } else if (kind === "airfoil") {
            const xl = dx * Math.cos(aoa) - dy * Math.sin(aoa);
            const yl = dx * Math.sin(aoa) + dy * Math.cos(aoa);
            const chord = r * 2.4;
            if (xl >= -0.4 && xl <= chord && Math.abs(dz) <= r * 1.4) {
              const xn = Math.min(1, Math.max(0, xl / chord));
              const yt =
                5 * 0.12 * (0.2969 * Math.sqrt(xn) - 0.126 * xn - 0.3516 * xn * xn + 0.2843 * xn * xn * xn - 0.1015 * xn * xn * xn * xn);
              inside = Math.abs(yl) <= Math.max(1.15, yt * chord);
            }
          }
          if (!inside) continue;
          flags[id] = F_BODY;
          u[id] = 0;
          v[id] = 0;
          w[id] = 0;
          temp[id] = this.tWall;
        }
      }
    }
    this.hasBody = true;
    this.bodyKind = kind;
    this.cx = x;
    this.cy = y;
    this.cz = z;
    this.radius = r;
    this.charLength = Math.max(this.charLength, r * 2);
    this.solidComponents = this.countBodies();
  }

  eraseBody(x: number, y: number, z: number, size: number): void {
    const { nx, ny, nz, flags } = this;
    const r2 = Math.max(1.2, size) ** 2;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const dx = i - x;
          const dy = j - y;
          const dz = k - z;
          if (dx * dx + dy * dy + dz * dz > r2) continue;
          const id = i + nx * (j + ny * k);
          if (flags[id] === F_BODY) flags[id] = F_FLUID;
        }
      }
    }
    this.hasBody = this.flags.some((f) => f === F_BODY);
    if (!this.hasBody) this.bodyKind = "none";
    this.solidComponents = this.countBodies();
  }

  clearBodies(): void {
    for (let id = 0; id < this.flags.length; id++) {
      if (this.flags[id] === F_BODY) this.flags[id] = F_FLUID;
    }
    this.hasBody = false;
    this.bodyKind = "none";
    this.solidComponents = 0;
  }

  pressureDrop(): number {
    const { p, flags, nx, ny, nz } = this;
    const k = (nz / 2) | 0;
    let up = 0;
    let down = 0;
    let nu = 0;
    let nd = 0;
    for (let j = 1; j < ny - 1; j++) {
      const a = 2 + nx * (j + ny * k);
      const b = nx - 3 + nx * (j + ny * k);
      if (!blocked(flags[a]!)) {
        up += p[a]!;
        nu++;
      }
      if (!blocked(flags[b]!)) {
        down += p[b]!;
        nd++;
      }
    }
    if (!nu || !nd) return 0;
    return up / nu - down / nd;
  }

  private drive(dt: number, gradP: number): void {
    if (Math.abs(gradP) < 1e-6) return;
    const { u, flags } = this;
    for (let id = 0; id < u.length; id++) {
      if (blocked(flags[id]!)) continue;
      u[id] = u[id]! + gradP * dt;
    }
  }

  private thermal(dt: number, params: SimParams): void {
    const tIn = params.tIn ?? this.tAmb;
    const tWall = params.tWall ?? this.tAmb;
    this.tWall = tWall;
    this.tAmb = params.tAmb ?? this.tAmb;
    const { temp, flags, u, v, w, nx, ny, nz } = this;
    const nxy = nx * ny;
    const out = this.dyeT;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = i + nx * j + nxy * k;
          const f = flags[id]!;
          if (f === F_BODY) {
            out[id] = tWall;
            continue;
          }
          if (f === F_INLET) {
            out[id] = tIn;
            continue;
          }
          if (blocked(f)) {
            out[id] = this.tAmb;
            continue;
          }
          let bx = u[id]! * dt;
          let by = v[id]! * dt;
          let bz = w[id]! * dt;
          const mag = Math.hypot(bx, by, bz);
          if (mag > 0.85) {
            const s = 0.85 / mag;
            bx *= s;
            by *= s;
            bz *= s;
          }
          out[id] = this.sample(temp, i - bx, j - by, k - bz);
        }
      }
    }
    const kappa = Math.min(dt * (params.kappa ?? 0.08), 0.18);
    if (kappa > 0) {
      for (let k = 1; k < nz - 1; k++) {
        for (let j = 1; j < ny - 1; j++) {
          for (let i = 1; i < nx - 1; i++) {
            const id = i + nx * j + nxy * k;
            const f = flags[id]!;
            if (f === F_BODY) {
              temp[id] = tWall;
              continue;
            }
            if (blocked(f) || f === F_INLET) {
              temp[id] = out[id]!;
              continue;
            }
            const nbr = [id - 1, id + 1, id - nx, id + nx, id - nxy, id + nxy];
            let sum = 0;
            for (let n = 0; n < 6; n++) {
              const nb = nbr[n]!;
              const fb = flags[nb]!;
              if (fb === F_BODY) sum += tWall;
              else if (fb === F_INLET) sum += tIn;
              else if (blocked(fb)) sum += this.tAmb;
              else sum += out[nb]!;
            }
            temp[id] = out[id]! + kappa * (sum - 6 * out[id]!);
          }
        }
      }
    } else {
      temp.set(out);
    }
    for (let id = 0; id < temp.length; id++) {
      if (flags[id] === F_BODY) temp[id] = tWall;
      else if (flags[id] === F_INLET) temp[id] = tIn;
    }
  }
  private buoy(dt: number, scale: number, ambient: number): void {
    if (scale === 0) return;
    const { v, dye, temp, flags } = this;
    for (let id = 0; id < v.length; id++) {
      if (blocked(flags[id]!)) continue;
      const lift = dye[id]! + (temp[id]! - ambient) / 40;
      v[id] = v[id]! + dt * scale * lift;
    }
  }

  private advect(dt: number): void {
    const { u, v, w, flags, uT, vT, wT, nx, ny, nz } = this;
    const nxy = nx * ny;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) {
            uT[id] = 0;
            vT[id] = 0;
            wT[id] = 0;
            continue;
          }
          let bx = u[id]! * dt;
          let by = v[id]! * dt;
          let bz = w[id]! * dt;
          const mag = Math.hypot(bx, by, bz);
          if (mag > 0.85) {
            const s = 0.85 / mag;
            bx *= s;
            by *= s;
            bz *= s;
          }
          uT[id] = this.sample(u, i - bx, j - by, k - bz);
          vT[id] = this.sample(v, i - bx, j - by, k - bz);
          wT[id] = this.sample(w, i - bx, j - by, k - bz);
        }
      }
    }
    u.set(uT);
    v.set(vT);
    w.set(wT);
  }

  private advectDye(dt: number): void {
    const { u, v, w, dye, dyeT, flags, nx, ny, nz } = this;
    const nxy = nx * ny;
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const id = i + nx * j + nxy * k;
          const f = flags[id]!;
          if (blocked(f)) {
            dyeT[id] = 0;
            continue;
          }
          if (f === F_INLET) {
            dyeT[id] = 1;
            continue;
          }
          let bx = u[id]! * dt;
          let by = v[id]! * dt;
          let bz = w[id]! * dt;
          const mag = Math.hypot(bx, by, bz);
          if (mag > 0.85) {
            const s = 0.85 / mag;
            bx *= s;
            by *= s;
            bz *= s;
          }
          dyeT[id] = this.sample(dye, i - bx, j - by, k - bz);
        }
      }
    }
    dye.set(dyeT);
  }

  private decayDye(dt: number, decay: number): void {
    if (decay <= 0) return;
    const keep = Math.max(0, 1 - decay * dt);
    const { dye, flags } = this;
    for (let id = 0; id < dye.length; id++) {
      if (blocked(flags[id]!)) dye[id] = 0;
      else dye[id] = dye[id]! * keep;
    }
  }

  private diffuse(dt: number, nu: number): void {
    if (nu < 1e-4) return;
    const alpha = Math.min(dt * nu, 0.12);
    if (alpha <= 0) return;
    const { u, v, w, flags, uT, vT, wT, nx, ny, nz } = this;
    const nxy = nx * ny;
    const relax = (field: Float32Array, out: Float32Array) => {
      for (let k = 1; k < nz - 1; k++) {
        for (let j = 1; j < ny - 1; j++) {
          for (let i = 1; i < nx - 1; i++) {
            const id = i + nx * j + nxy * k;
            if (blocked(flags[id]!)) {
              out[id] = 0;
              continue;
            }
            let sum = 0;
            const nbr = [id - 1, id + 1, id - nx, id + nx, id - nxy, id + nxy];
            for (let n = 0; n < 6; n++) {
              const nb = nbr[n]!;
              sum += blocked(flags[nb]!) ? 0 : field[nb]!;
            }
            out[id] = field[id]! + alpha * (sum - 6 * field[id]!);
          }
        }
      }
    };
    relax(u, uT);
    relax(v, vT);
    relax(w, wT);
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) continue;
          u[id] = uT[id]!;
          v[id] = vT[id]!;
          w[id] = wT[id]!;
        }
      }
    }
  }

  private confine(dt: number, eps: number): void {
    const { u, v, w, flags, mag, nx, ny, nz } = this;
    const nxy = nx * ny;
    mag.fill(0);
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) continue;
          const dwdy = 0.5 * (w[id + nx]! - w[id - nx]!);
          const dvdz = 0.5 * (v[id + nxy]! - v[id - nxy]!);
          const dudz = 0.5 * (u[id + nxy]! - u[id - nxy]!);
          const dwdx = 0.5 * (w[id + 1]! - w[id - 1]!);
          const dvdx = 0.5 * (v[id + 1]! - v[id - 1]!);
          const dudy = 0.5 * (u[id + nx]! - u[id - nx]!);
          const wx = dwdy - dvdz;
          const wy = dudz - dwdx;
          const wz = dvdx - dudy;
          mag[id] = Math.hypot(wx, wy, wz);
        }
      }
    }
    const gain = eps * 0.35;
    for (let k = 2; k < nz - 2; k++) {
      for (let j = 2; j < ny - 2; j++) {
        for (let i = 2; i < nx - 2; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) continue;
          const dwdy = 0.5 * (w[id + nx]! - w[id - nx]!);
          const dvdz = 0.5 * (v[id + nxy]! - v[id - nxy]!);
          const dudz = 0.5 * (u[id + nxy]! - u[id - nxy]!);
          const dwdx = 0.5 * (w[id + 1]! - w[id - 1]!);
          const dvdx = 0.5 * (v[id + 1]! - v[id - 1]!);
          const dudy = 0.5 * (u[id + nx]! - u[id - nx]!);
          const wx = dwdy - dvdz;
          const wy = dudz - dwdx;
          const wz = dvdx - dudy;
          let nxg = 0.5 * (mag[id + 1]! - mag[id - 1]!);
          let nyg = 0.5 * (mag[id + nx]! - mag[id - nx]!);
          let nzg = 0.5 * (mag[id + nxy]! - mag[id - nxy]!);
          const nlen = Math.hypot(nxg, nyg, nzg);
          if (nlen < 1e-6) continue;
          nxg /= nlen;
          nyg /= nlen;
          nzg /= nlen;
          u[id] = u[id]! + dt * gain * (nyg * wz - nzg * wy);
          v[id] = v[id]! + dt * gain * (nzg * wx - nxg * wz);
          w[id] = w[id]! + dt * gain * (nxg * wy - nyg * wx);
        }
      }
    }
  }

  private project(dt: number, iterations: number): void {
    const { flags, u, v, w, p, div, nx, ny, nz } = this;
    const nxy = nx * ny;
    const invDt = 1 / Math.max(dt, 1e-5);
    const omega = 1.45;
    let residual = 0;
    let count = 0;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) {
            div[id] = 0;
            continue;
          }
          const uR = blocked(flags[id + 1]!) ? 0 : u[id + 1]!;
          const uL = blocked(flags[id - 1]!) ? 0 : u[id - 1]!;
          const vT = blocked(flags[id + nx]!) ? 0 : v[id + nx]!;
          const vB = blocked(flags[id - nx]!) ? 0 : v[id - nx]!;
          const wF = blocked(flags[id + nxy]!) ? 0 : w[id + nxy]!;
          const wK = blocked(flags[id - nxy]!) ? 0 : w[id - nxy]!;
          const d = 0.5 * (uR - uL + (vT - vB) + (wF - wK));
          div[id] = d;
          residual += Math.abs(d);
          count++;
        }
      }
    }
    this.divResidual = count > 0 ? residual / count : 0;
    const iters = Math.max(4, iterations | 0);
    for (let sweep = 0; sweep < iters; sweep++) {
      for (let k = 1; k < nz - 1; k++) {
        for (let j = 1; j < ny - 1; j++) {
          for (let i = 1; i < nx - 1; i++) {
            const id = i + nx * j + nxy * k;
            const f = flags[id]!;
            if (blocked(f) || f === F_OUTLET) {
              p[id] = 0;
              continue;
            }
            let sum = 0;
            let nFluid = 0;
            const nbr = [id - 1, id + 1, id - nx, id + nx, id - nxy, id + nxy];
            for (let n = 0; n < 6; n++) {
              const nb = nbr[n]!;
              if (!blocked(flags[nb]!)) {
                sum += p[nb]!;
                nFluid++;
              }
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
    }
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          const f = flags[id]!;
          if (blocked(f) || f === F_INLET || f === F_LID) continue;
          const pL = blocked(flags[id - 1]!) ? p[id]! : p[id - 1]!;
          const pR = blocked(flags[id + 1]!) ? p[id]! : p[id + 1]!;
          const pB = blocked(flags[id - nx]!) ? p[id]! : p[id - nx]!;
          const pT = blocked(flags[id + nx]!) ? p[id]! : p[id + nx]!;
          const pK = blocked(flags[id - nxy]!) ? p[id]! : p[id - nxy]!;
          const pF = blocked(flags[id + nxy]!) ? p[id]! : p[id + nxy]!;
          u[id] = u[id]! - dt * 0.5 * (pR - pL);
          v[id] = v[id]! - dt * 0.5 * (pT - pB);
          w[id] = w[id]! - dt * 0.5 * (pF - pK);
        }
      }
    }
    residual = 0;
    count = 0;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          if (blocked(flags[id]!)) continue;
          const uR = blocked(flags[id + 1]!) ? 0 : u[id + 1]!;
          const uL = blocked(flags[id - 1]!) ? 0 : u[id - 1]!;
          const vT = blocked(flags[id + nx]!) ? 0 : v[id + nx]!;
          const vB = blocked(flags[id - nx]!) ? 0 : v[id - nx]!;
          const wF = blocked(flags[id + nxy]!) ? 0 : w[id + nxy]!;
          const wK = blocked(flags[id - nxy]!) ? 0 : w[id - nxy]!;
          residual += Math.abs(0.5 * (uR - uL + (vT - vB) + (wF - wK)));
          count++;
        }
      }
    }
    this.divResidual = count > 0 ? residual / count : 0;
  }

  private measure(dt: number, nu: number): void {
    const { u, v, w, flags, p, nx, ny, nz } = this;
    const nxy = nx * ny;
    let ke = 0;
    let maxS = 0.05;
    let maxCurl = 0;
    let fx = 0;
    let fy = 0;
    let phi = 0;
    let flux = 0;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const id = i + nx * j + nxy * k;
          const f = flags[id]!;
          if (f === F_BODY) {
            if (!blocked(flags[id - 1]!)) fx += p[id - 1]!;
            if (!blocked(flags[id + 1]!)) fx -= p[id + 1]!;
            if (!blocked(flags[id - nx]!)) fy += p[id - nx]!;
            if (!blocked(flags[id + nx]!)) fy -= p[id + nx]!;
            continue;
          }
          if (blocked(f)) continue;
          const uu = u[id]!;
          const vv = v[id]!;
          const ww = w[id]!;
          if (!Number.isFinite(uu) || !Number.isFinite(vv) || !Number.isFinite(ww)) {
            u[id] = 0;
            v[id] = 0;
            w[id] = 0;
            continue;
          }
          ke += uu * uu + vv * vv + ww * ww;
          const sp = Math.hypot(uu, vv, ww);
          if (sp > maxS) maxS = sp;
          const dudx = 0.5 * ((blocked(flags[id + 1]!) ? 0 : u[id + 1]!) - (blocked(flags[id - 1]!) ? 0 : u[id - 1]!));
          const dudy = 0.5 * ((blocked(flags[id + nx]!) ? 0 : u[id + nx]!) - (blocked(flags[id - nx]!) ? 0 : u[id - nx]!));
          const dudz = 0.5 * ((blocked(flags[id + nxy]!) ? 0 : u[id + nxy]!) - (blocked(flags[id - nxy]!) ? 0 : u[id - nxy]!));
          const dvdx = 0.5 * ((blocked(flags[id + 1]!) ? 0 : v[id + 1]!) - (blocked(flags[id - 1]!) ? 0 : v[id - 1]!));
          const dwdx = 0.5 * ((blocked(flags[id + 1]!) ? 0 : w[id + 1]!) - (blocked(flags[id - 1]!) ? 0 : w[id - 1]!));
          const curl = Math.hypot(dudy - 0, dvdx - dudz, dwdx - dudy);
          if (curl > maxCurl) maxCurl = curl;
          phi += nu * (dudx * dudx + dudy * dudy + dudz * dudz + dvdx * dvdx + dwdx * dwdx);
          if (f === F_INLET) flux += 0.5 * (uu * uu + vv * vv + ww * ww) * Math.max(uu, 0);
        }
      }
    }

    const count = this.px.length;
    for (let n = 0; n < count; n++) {
      const x = this.px[n]!;
      const y = this.py[n]!;
      const z = this.pz[n]!;
      const uu = this.sample(u, x, y, z);
      const vv = this.sample(v, x, y, z);
      const ww = this.sample(w, x, y, z);
      const xn = x + uu * dt;
      const yn = y + vv * dt;
      const zn = z + ww * dt;
      const xi = xn | 0;
      const yi = yn | 0;
      const zi = zn | 0;
      const inside = xi > 0 && yi > 0 && zi > 0 && xi < nx - 1 && yi < ny - 1 && zi < nz - 1;
      const id = xi + nx * (yi + ny * zi);
      if (!inside || blocked(flags[id] ?? F_SOLID) || !Number.isFinite(xn)) this.respawn(n, false);
      else {
        this.px[n] = xn;
        this.py[n] = yn;
        this.pz[n] = zn;
      }
    }

    this.ke = 0.5 * ke;
    this.maxSpeed = maxS;
    this.maxW = maxCurl;
    this.fx = fx;
    this.fy = fy;
    this.dissipation = phi;
    this.inletFlux = flux;
    this.circulation = this.loopCirculation();
    this.reflectionDefect = this.yDefect();
  }

  private loopCirculation(): number {
    const { u, v, nx, ny, nz, cx, cy, radius } = this;
    const k = (nz / 2) | 0;
    const r = Math.min(nx * 0.2, Math.max(radius + 2, 3));
    const i0 = Math.max(1, Math.round(cx - r));
    const i1 = Math.min(nx - 2, Math.round(cx + r));
    const j0 = Math.max(1, Math.round(cy - r));
    const j1 = Math.min(ny - 2, Math.round(cy + r));
    let gamma = 0;
    const nxy = nx * ny;
    for (let i = i0; i < i1; i++) gamma += u[i + nx * j0 + nxy * k] ?? 0;
    for (let j = j0; j < j1; j++) gamma += v[i1 + nx * j + nxy * k] ?? 0;
    for (let i = i1; i > i0; i--) gamma -= u[i + nx * j1 + nxy * k] ?? 0;
    for (let j = j1; j > j0; j--) gamma -= v[i0 + nx * j + nxy * k] ?? 0;
    return gamma;
  }

  private yDefect(): number {
    const { v, flags, nx, ny, nz } = this;
    const nxy = nx * ny;
    let num = 0;
    let den = 0;
    const mid = (ny / 2) | 0;
    for (let k = 1; k < nz - 1; k += 2) {
      for (let j = 1; j < mid; j++) {
        const jm = ny - 1 - j;
        for (let i = 1; i < nx - 1; i += 2) {
          const a = i + nx * j + nxy * k;
          const b = i + nx * jm + nxy * k;
          if (blocked(flags[a]!) || blocked(flags[b]!)) continue;
          const va = v[a]!;
          const vb = v[b]!;
          num += Math.abs(va + vb);
          den += Math.abs(va) + Math.abs(vb);
        }
      }
    }
    return den > 1e-6 ? num / den : 0;
  }

  private countBodies(): number {
    const { flags, nx, ny, nz } = this;
    const n = flags.length;
    const seen = new Uint8Array(n);
    let components = 0;
    const stack = new Int32Array(n);
    const nxy = nx * ny;
    for (let start = 0; start < n; start++) {
      if (flags[start] !== F_BODY || seen[start]) continue;
      components++;
      let top = 0;
      stack[top++] = start;
      seen[start] = 1;
      while (top > 0) {
        const id = stack[--top]!;
        const nbr = [id - 1, id + 1, id - nx, id + nx, id - nxy, id + nxy];
        for (let s = 0; s < 6; s++) {
          const nb = nbr[s]!;
          if (nb < 0 || nb >= n || seen[nb] || flags[nb] !== F_BODY) continue;
          seen[nb] = 1;
          stack[top++] = nb;
        }
      }
    }
    return components;
  }

  private trackStrouhal(fy: number, uIn: number): void {
    if (!this.hasBody || Math.abs(uIn) < 1e-3) {
      this.strouhal = 0;
      this.prevFy = fy;
      return;
    }
    if (this.prevFy < 0 && fy >= 0 && this.lastCross > 0) {
      const period = this.time - this.lastCross;
      if (period > 0.2) this.strouhal = this.charLength / (period * Math.abs(uIn));
    }
    if (this.prevFy < 0 && fy >= 0) this.lastCross = this.time;
    this.prevFy = fy;
  }
}
