import { LUT_DIVERGE, LUT_INK, LUT_SPEED, lutAt } from "./colormap";
import type { Fluid3 } from "./solver3d";
import { F_BODY, F_SOLID, type ViewMode } from "./types";

export type Camera = { yaw: number; pitch: number; dist: number };

type P2 = { x: number; y: number; z: number };

function basis(cam: Camera, sim: Fluid3) {
  const tx = (sim.nx - 1) / 2;
  const ty = (sim.ny - 1) / 2;
  const tz = (sim.nz - 1) / 2;
  const cp = Math.cos(cam.pitch);
  const sp = Math.sin(cam.pitch);
  const cy = Math.cos(cam.yaw);
  const sy = Math.sin(cam.yaw);
  const ex = tx + cam.dist * cp * sy;
  const ey = ty + cam.dist * sp;
  const ez = tz + cam.dist * cp * cy;
  let fx = tx - ex;
  let fy = ty - ey;
  let fz = tz - ez;
  const fl = Math.hypot(fx, fy, fz) || 1;
  fx /= fl;
  fy /= fl;
  fz /= fl;
  let rx = fy * 0 - fz * 1;
  let ry = fz * 0 - fx * 0;
  let rz = fx * 1 - fy * 0;
  const rl = Math.hypot(rx, ry, rz) || 1;
  rx /= rl;
  ry /= rl;
  rz /= rl;
  const ux = ry * fz - rz * fy;
  const uy = rz * fx - rx * fz;
  const uz = rx * fy - ry * fx;
  return { ex, ey, ez, fx, fy, fz, rx, ry, rz, ux, uy, uz };
}

export function project(sim: Fluid3, cam: Camera, x: number, y: number, z: number, w: number, h: number): P2 | null {
  const b = basis(cam, sim);
  const dx = x - b.ex;
  const dy = y - b.ey;
  const dz = z - b.ez;
  const depth = dx * b.fx + dy * b.fy + dz * b.fz;
  if (depth < 0.4) return null;
  const right = dx * b.rx + dy * b.ry + dz * b.rz;
  const up = dx * b.ux + dy * b.uy + dz * b.uz;
  const f = Math.min(w, h) * 0.92;
  return { x: w / 2 + (right / depth) * f, y: h / 2 - (up / depth) * f, z: depth };
}

export function hitVolume(
  sim: Fluid3,
  cam: Camera,
  px: number,
  py: number,
  w: number,
  h: number,
): { x: number; y: number; z: number } | null {
  const b = basis(cam, sim);
  const ndcX = (px / w) * 2 - 1;
  const ndcY = 1 - (py / h) * 2;
  const span = Math.min(w, h) * 0.92;
  const scale = (Math.hypot(b.ex - (sim.nx - 1) / 2, b.ey - (sim.ny - 1) / 2, b.ez - (sim.nz - 1) / 2) || 1) / span;
  let dx = b.fx + ndcX * b.rx * scale * w * 0.5 + ndcY * b.ux * scale * h * 0.5;
  let dy = b.fy + ndcX * b.ry * scale * w * 0.5 + ndcY * b.uy * scale * h * 0.5;
  let dz = b.fz + ndcX * b.rz * scale * w * 0.5 + ndcY * b.uz * scale * h * 0.5;
  const len = Math.hypot(dx, dy, dz) || 1;
  dx /= len;
  dy /= len;
  dz /= len;
  const planeZ = (sim.nz - 1) / 2;
  if (Math.abs(dz) < 1e-4) return null;
  const t = (planeZ - b.ez) / dz;
  if (t < 0) return null;
  return { x: b.ex + t * dx, y: b.ey + t * dy, z: planeZ };
}

function bodyExtents(sim: Fluid3): {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
  z0: number;
  z1: number;
  cx: number;
  cy: number;
  cz: number;
  rx: number;
  ry: number;
  rz: number;
} | null {
  const { flags, nx, ny, nz } = sim;
  let x0 = nx;
  let x1 = 0;
  let y0 = ny;
  let y1 = 0;
  let z0 = nz;
  let z1 = 0;
  let n = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        if (flags[i + nx * (j + ny * k)] !== F_BODY) continue;
        if (i < x0) x0 = i;
        if (i > x1) x1 = i;
        if (j < y0) y0 = j;
        if (j > y1) y1 = j;
        if (k < z0) z0 = k;
        if (k > z1) z1 = k;
        n++;
      }
    }
  }
  if (n < 1) return null;
  return {
    x0,
    x1,
    y0,
    y1,
    z0,
    z1,
    cx: (x0 + x1) / 2,
    cy: (y0 + y1) / 2,
    cz: (z0 + z1) / 2,
    rx: Math.max(0.6, (x1 - x0) / 2),
    ry: Math.max(0.6, (y1 - y0) / 2),
    rz: Math.max(0.6, (z1 - z0) / 2),
  };
}

function line(ctx: CanvasRenderingContext2D, a: P2 | null, b: P2 | null): void {
  if (!a || !b) return;
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}

function scalarAt(sim: Fluid3, id: number, view: ViewMode): number {
  if (view === "dye") return sim.dye[id] ?? 0;
  if (view === "pressure") return sim.p[id] ?? 0;
  if (view === "temperature") return sim.temp[id] ?? sim.tAmb;
  if (view === "speed") return Math.hypot(sim.u[id] ?? 0, sim.v[id] ?? 0, sim.w[id] ?? 0);
  const nx = sim.nx;
  const nxy = sim.nx * sim.ny;
  const dwdy = 0.5 * ((sim.w[id + nx] ?? 0) - (sim.w[id - nx] ?? 0));
  const dvdz = 0.5 * ((sim.v[id + nxy] ?? 0) - (sim.v[id - nxy] ?? 0));
  const dvdx = 0.5 * ((sim.v[id + 1] ?? 0) - (sim.v[id - 1] ?? 0));
  const dudy = 0.5 * ((sim.u[id + nx] ?? 0) - (sim.u[id - nx] ?? 0));
  return dvdx - dudy + (dwdy - dvdz) * 0.15;
}

export function drawVolume(
  canvas: HTMLCanvasElement,
  sim: Fluid3,
  view: ViewMode,
  cam: Camera,
  particles: boolean,
): void {
  const ctx = canvas.getContext("2d", { alpha: false });
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#101412";
  ctx.fillRect(0, 0, w, h);
  const p = (x: number, y: number, z: number) => project(sim, cam, x, y, z, w, h);
  ctx.strokeStyle = "#2c3833";
  ctx.lineWidth = Math.max(1, w / 700);
  const nx = sim.nx - 1;
  const ny = sim.ny - 1;
  const nz = sim.nz - 1;
  const corners = [
    p(0, 0, 0),
    p(nx, 0, 0),
    p(nx, ny, 0),
    p(0, ny, 0),
    p(0, 0, nz),
    p(nx, 0, nz),
    p(nx, ny, nz),
    p(0, ny, nz),
  ];
  const edges: ReadonlyArray<readonly [number, number]> = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  for (let e = 0; e < edges.length; e++) {
    const edge = edges[e]!;
    line(ctx, corners[edge[0]] ?? null, corners[edge[1]] ?? null);
  }

  ctx.strokeStyle = "#e39b2b";
  ctx.lineWidth = Math.max(1, w / 900);
  const box = bodyExtents(sim);
  if (box && sim.bodyKind === "sphere") {
    const rings = 18;
    const rad = Math.min(box.rx, box.ry, box.rz);
    for (let axis = 0; axis < 3; axis++) {
      let prev: P2 | null = null;
      for (let s = 0; s <= rings; s++) {
        const a = (s / rings) * Math.PI * 2;
        const c = Math.cos(a) * rad;
        const sn = Math.sin(a) * rad;
        const x = box.cx + (axis === 0 ? 0 : c);
        const y = box.cy + (axis === 1 ? 0 : axis === 0 ? c : sn);
        const z = box.cz + (axis === 2 ? 0 : axis === 1 ? c : sn);
        const q = p(x, y, z);
        if (s > 0) line(ctx, prev, q);
        prev = q;
      }
    }
  } else if (box && sim.bodyKind === "cylinder") {
    const z0 = box.z0;
    const z1 = box.z1;
    const rad = Math.max(box.rx, box.ry);
    for (const z of [z0, z1]) {
      let prev: P2 | null = null;
      for (let s = 0; s <= 20; s++) {
        const a = (s / 20) * Math.PI * 2;
        const q = p(box.cx + Math.cos(a) * rad, box.cy + Math.sin(a) * rad, z);
        if (s > 0) line(ctx, prev, q);
        prev = q;
      }
    }
    for (let s = 0; s < 4; s++) {
      const a = (s / 4) * Math.PI * 2;
      line(ctx, p(box.cx + Math.cos(a) * rad, box.cy + Math.sin(a) * rad, z0), p(box.cx + Math.cos(a) * rad, box.cy + Math.sin(a) * rad, z1));
    }
  } else if (box && sim.bodyKind === "airfoil") {
    const chord = Math.max(2, box.x1 - box.x0);
    const z0 = box.z0;
    const z1 = box.z1;
    const ang = (sim.aoa * Math.PI) / 180;
    for (const z of [z0, (z0 + z1) / 2, z1]) {
      let prev: P2 | null = null;
      const samples = 28;
      for (let s = 0; s <= samples; s++) {
        const upper = s <= samples / 2;
        const t = upper ? s / (samples / 2) : 1 - (s - samples / 2) / (samples / 2);
        const x = Math.min(1, Math.max(0, t));
        const yt = 5 * 0.12 * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
        const half = Math.max(yt, 1.15 / chord);
        const yl = (upper ? half : -half) * chord;
        const xl = x * chord;
        const X = box.x0 + xl * Math.cos(ang) + yl * Math.sin(ang);
        const Y = box.cy - xl * Math.sin(ang) + yl * Math.cos(ang);
        const q = p(X, Y, z);
        if (s > 0) line(ctx, prev, q);
        prev = q;
      }
    }
  } else if (box && sim.bodyKind === "cone") {
    const tip = p(box.x1, box.cy, box.cz);
    const rad = Math.max(box.ry, box.rz);
    let prev: P2 | null = null;
    for (let s = 0; s <= 16; s++) {
      const a = (s / 16) * Math.PI * 2;
      const q = p(box.x0, box.cy + Math.cos(a) * rad, box.cz + Math.sin(a) * rad);
      if (s > 0) line(ctx, prev, q);
      if (s % 4 === 0) line(ctx, q, tip);
      prev = q;
    }
  } else if (box) {
    const xs = [box.x0, box.x1];
    const ys = [box.y0, box.y1];
    const zs = [box.z0, box.z1];
    for (const x of xs) for (const y of ys) line(ctx, p(x, y, zs[0]!), p(x, y, zs[1]!));
    for (const y of ys) for (const z of zs) line(ctx, p(xs[0]!, y, z), p(xs[1]!, y, z));
    for (const x of xs) for (const z of zs) line(ctx, p(x, ys[0]!, z), p(x, ys[1]!, z));
  }

  if (particles && sim.particleCount > 0) {
    const order = new Array<number>(sim.particleCount);
    const depth = new Float32Array(sim.particleCount);
    for (let n = 0; n < sim.particleCount; n++) {
      order[n] = n;
      const q = p(sim.px[n]!, sim.py[n]!, sim.pz[n]!);
      depth[n] = q ? q.z : -1;
    }
    order.sort((a, b) => depth[b]! - depth[a]!);
    const diverge = view === "vorticity" || view === "pressure";
    const heat = view === "temperature";
    let lo = 0;
    let hi = 1;
    if (diverge || heat) {
      lo = Infinity;
      hi = -Infinity;
      for (let n = 0; n < sim.particleCount; n += 3) {
        const id = cellId(sim, sim.px[n]!, sim.py[n]!, sim.pz[n]!);
        const s = scalarAt(sim, id, view);
        if (s < lo) lo = s;
        if (s > hi) hi = s;
      }
      if (diverge) {
        const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-4);
        lo = -m;
        hi = m;
      } else if (hi - lo < 1e-3) {
        lo -= 1;
        hi += 1;
      }
    }
    const color = { r: 0, g: 0, b: 0 };
    for (let k = 0; k < order.length; k++) {
      const n = order[k]!;
      if (depth[n]! < 0) continue;
      const q = p(sim.px[n]!, sim.py[n]!, sim.pz[n]!);
      if (!q) continue;
      const id = cellId(sim, sim.px[n]!, sim.py[n]!, sim.pz[n]!);
      if (sim.flags[id] === F_BODY || sim.flags[id] === F_SOLID) continue;
      const s0 = scalarAt(sim, id, view);
      const speed = Math.hypot(sim.u[id] ?? 0, sim.v[id] ?? 0, sim.w[id] ?? 0);
      const s = view === "dye" ? Math.max(s0, 0.28 * (speed / (sim.maxSpeed || 1))) : s0;
      const t = diverge || heat ? (s - lo) / (hi - lo || 1) : Math.max(0, Math.min(1, view === "dye" ? s : s / (sim.maxSpeed || 1)));
      const lut = view === "dye" ? LUT_INK : heat ? LUT_SPEED : diverge ? LUT_DIVERGE : LUT_SPEED;
      lutAt(lut, Math.max(0, Math.min(1, t)) ** (view === "dye" ? 0.6 : 1), color);
      ctx.fillStyle = `rgb(${color.r},${color.g},${color.b})`;
      const size = Math.max(2, ((w / 700) * cam.dist) / q.z);
      ctx.fillRect(q.x, q.y, size, size);
    }
  }

  drawSlice(ctx, sim, view, w, h);
}

function cellId(sim: Fluid3, x: number, y: number, z: number): number {
  const i = Math.max(0, Math.min(sim.nx - 1, x | 0));
  const j = Math.max(0, Math.min(sim.ny - 1, y | 0));
  const k = Math.max(0, Math.min(sim.nz - 1, z | 0));
  return i + sim.nx * (j + sim.ny * k);
}

function drawSlice(ctx: CanvasRenderingContext2D, sim: Fluid3, view: ViewMode, w: number, h: number): void {
  const sw = Math.max(48, Math.round(w * 0.22));
  const sh = Math.max(24, Math.round(sw * (sim.ny / sim.nx)));
  const image = ctx.createImageData(sw, sh);
  const k = (sim.nz / 2) | 0;
  const diverge = view === "vorticity" || view === "pressure";
  const heat = view === "temperature";
  let lo = 0;
  let hi = 1;
  if (view === "speed") hi = Math.max(sim.maxSpeed, 0.2);
  if (diverge || heat) {
    lo = Infinity;
    hi = -Infinity;
    for (let j = 0; j < sim.ny; j += 2) {
      for (let i = 0; i < sim.nx; i += 2) {
        const s = scalarAt(sim, i + sim.nx * (j + sim.ny * k), view);
        if (s < lo) lo = s;
        if (s > hi) hi = s;
      }
    }
    if (diverge) {
      const m = Math.max(Math.abs(lo), Math.abs(hi), 1e-4);
      lo = -m;
      hi = m;
    } else if (hi - lo < 1e-3) {
      lo -= 1;
      hi += 1;
    }
  }
  const lut = view === "dye" ? LUT_INK : heat ? LUT_SPEED : diverge ? LUT_DIVERGE : LUT_SPEED;
  for (let y = 0; y < sh; y++) {
    const j = Math.min(sim.ny - 1, ((sh - 1 - y) / sh) * sim.ny) | 0;
    for (let x = 0; x < sw; x++) {
      const i = Math.min(sim.nx - 1, (x / sw) * sim.nx) | 0;
      const id = i + sim.nx * (j + sim.ny * k);
      const o = (x + y * sw) * 4;
      if (sim.flags[id] === F_BODY) {
        image.data[o] = 227;
        image.data[o + 1] = 155;
        image.data[o + 2] = 43;
        image.data[o + 3] = 255;
        continue;
      }
      const color = { r: 0, g: 0, b: 0 };
      const s = scalarAt(sim, id, view);
      const t = Math.max(0, Math.min(1, (s - lo) / (hi - lo || 1)));
      lutAt(lut, view === "dye" ? t ** 0.55 : t, color);
      image.data[o] = color.r;
      image.data[o + 1] = color.g;
      image.data[o + 2] = color.b;
      image.data[o + 3] = 255;
    }
  }
  const pad = Math.round(w * 0.02);
  ctx.putImageData(image, pad, h - sh - pad);
  ctx.strokeStyle = "#8d9a92";
  ctx.strokeRect(pad + 0.5, h - sh - pad + 0.5, sw, sh);
}

export function drawCurve(
  canvas: HTMLCanvasElement,
  a: Float32Array,
  b: Float32Array | null,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1a211e";
  ctx.fillRect(0, 0, w, h);
  let lo = Infinity;
  let hi = -Infinity;
  const scan = (arr: Float32Array | null) => {
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      const v = arr[i]!;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  };
  scan(a);
  scan(b);
  if (!Number.isFinite(lo) || hi - lo < 1e-6) {
    lo = 0;
    hi = 1;
  }
  const plot = (arr: Float32Array, color: string) => {
    ctx.beginPath();
    for (let i = 0; i < arr.length; i++) {
      const x = (i / (arr.length - 1)) * (w - 8) + 4;
      const y = h - 6 - ((arr[i]! - lo) / (hi - lo)) * (h - 12);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  };
  plot(a, "#e39b2b");
  if (b) plot(b, "#8d9a92");
}
