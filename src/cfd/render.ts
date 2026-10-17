import { LUT_DIVERGE, LUT_INK, LUT_SPEED, lutAt } from "./colormap";
import type { FluidSim } from "./solver";
import { F_BODY, F_SOLID, type ViewMode } from "./types";

const SOLID = [22, 28, 25] as const;
const EDGE = [214, 168, 70] as const;
const INK = { r: 0, g: 0, b: 0 };

export type FrameTarget = {
  nx: number;
  ny: number;
  image: ImageData;
  off: HTMLCanvasElement;
};

export function ensureTarget(nx: number, ny: number, prev: FrameTarget | null): FrameTarget {
  if (prev && prev.nx === nx && prev.ny === ny) return prev;
  const off = document.createElement("canvas");
  off.width = nx;
  off.height = ny;
  const ctx = off.getContext("2d");
  if (!ctx) throw new Error("2D canvas is unavailable");
  return { nx, ny, image: ctx.createImageData(nx, ny), off };
}

export type DrawState = {
  dyeScale: number;
  speedScale: number;
  vortScale: number;
  presScale: number;
};

export function createDrawState(): DrawState {
  return { dyeScale: 1, speedScale: 8, vortScale: 1, presScale: 1 };
}

function blocked(flag: number): boolean {
  return flag === F_SOLID || flag === F_BODY;
}

export function drawField(
  view: HTMLCanvasElement,
  sim: FluidSim,
  target: FrameTarget,
  viewMode: ViewMode,
  vectors: boolean,
  particles: boolean,
  scale: DrawState,
): void {
  const ctx = view.getContext("2d");
  const off = target.off.getContext("2d");
  if (!ctx || !off) return;
  const { nx, ny } = sim;
  const { image } = target;
  const pix = image.data;
  const { flags, u, v, dye, p, omega } = sim;

  let maxS = 1e-3;
  let maxW = 1e-3;
  let maxP = 1e-3;
  if (viewMode !== "dye") {
    for (let i = 0; i < flags.length; i++) {
      if (blocked(flags[i]!)) continue;
      if (viewMode === "speed") {
        const s = Math.hypot(u[i]!, v[i]!);
        if (s > maxS) maxS = s;
      } else if (viewMode === "vorticity") {
        const w = Math.abs(omega[i]!);
        if (w > maxW) maxW = w;
      } else {
        const a = Math.abs(p[i]!);
        if (a > maxP) maxP = a;
      }
    }
  }
  scale.speedScale = Math.max(maxS, scale.speedScale * 0.94);
  scale.vortScale = Math.max(maxW, scale.vortScale * 0.94);
  scale.presScale = Math.max(maxP, scale.presScale * 0.94);

  const lut = viewMode === "speed" ? LUT_SPEED : viewMode === "dye" ? LUT_INK : LUT_DIVERGE;

  for (let j = 0; j < ny; j++) {
    const row = (ny - 1 - j) * nx;
    for (let i = 0; i < nx; i++) {
      const id = i + j * nx;
      const f = flags[id]!;
      let r: number;
      let g: number;
      let b: number;
      if (blocked(f)) {
        let edge = false;
        if (f === F_BODY) {
          const im = i > 0 ? flags[id - 1]! : F_SOLID;
          const ip = i + 1 < nx ? flags[id + 1]! : F_SOLID;
          const jb = j > 0 ? flags[id - nx]! : F_SOLID;
          const jt = j + 1 < ny ? flags[id + nx]! : F_SOLID;
          edge = !blocked(im) || !blocked(ip) || !blocked(jb) || !blocked(jt);
        }
        if (edge) {
          r = EDGE[0];
          g = EDGE[1];
          b = EDGE[2];
        } else {
          r = SOLID[0];
          g = SOLID[1];
          b = SOLID[2];
        }
      } else {
        let t = 0;
        if (viewMode === "dye") t = Math.pow(Math.min(1, Math.max(0, dye[id]!)), 0.55);
        else if (viewMode === "speed") t = Math.min(1, Math.hypot(u[id]!, v[id]!) / scale.speedScale);
        else if (viewMode === "vorticity")
          t = Math.min(1, Math.max(0, 0.5 + (0.5 * omega[id]!) / scale.vortScale));
        else t = Math.min(1, Math.max(0, 0.5 + (0.5 * p[id]!) / scale.presScale));
        lutAt(lut, t, INK);
        r = INK.r;
        g = INK.g;
        b = INK.b;
      }
      const o = (row + i) * 4;
      pix[o] = r;
      pix[o + 1] = g;
      pix[o + 2] = b;
      pix[o + 3] = 255;
    }
  }

  off.putImageData(image, 0, 0);
  const w = view.width;
  const h = view.height;
  ctx.imageSmoothingEnabled = true;
  ctx.fillStyle = "#101412";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(target.off, 0, 0, w, h);

  if (vectors) {
    ctx.strokeStyle = "rgba(231, 239, 233, 0.45)";
    ctx.lineWidth = 1;
    const stride = nx > 180 ? 10 : 8;
    const gain = (w / nx) * 0.85;
    const denom = Math.max(scale.speedScale, 1e-3);
    ctx.beginPath();
    for (let j = stride; j < ny - 1; j += stride) {
      for (let i = stride; i < nx - 1; i += stride) {
        const id = i + j * nx;
        if (blocked(flags[id]!)) continue;
        const sx = ((i + 0.5) / nx) * w;
        const sy = (1 - (j + 0.5) / ny) * h;
        const vx = (u[id]! / denom) * gain * stride * 0.45;
        const vy = -(v[id]! / denom) * gain * stride * 0.45;
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx + vx, sy + vy);
      }
    }
    ctx.stroke();
  }

  if (particles) {
    ctx.strokeStyle = "rgba(244, 240, 228, 0.8)";
    ctx.lineWidth = Math.max(1, w / nx / 3);
    ctx.beginPath();
    const n = sim.particleCount;
    for (let k = 0; k < n; k++) {
      const x0 = ((sim.ox[k] ?? sim.px[k]!) / nx) * w;
      const y0 = (1 - (sim.oy[k] ?? sim.py[k]!) / ny) * h;
      const x1 = (sim.px[k]! / nx) * w;
      const y1 = (1 - sim.py[k]! / ny) * h;
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
    }
    ctx.stroke();
  }
}

export function drawSpark(canvas: HTMLCanvasElement, values: Float32Array, count: number, cursor: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || canvas.width === 0) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(227, 155, 43, 0.9)";
  ctx.lineWidth = 1.5;
  let lo = Infinity;
  let hi = -Infinity;
  const n = Math.min(count, values.length);
  if (n < 2) return;
  const at = (i: number) => values[(cursor - n + i + values.length) % values.length]!;
  for (let i = 0; i < n; i++) {
    const v = at(i);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo < 1e-4) {
    lo -= 0.5;
    hi += 0.5;
  }
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const v = at(i);
    const x = (i / (n - 1)) * (w - 2) + 1;
    const y = h - 2 - ((v - lo) / (hi - lo)) * (h - 4);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}
