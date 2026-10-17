import {
  F_BODY,
  F_INLET,
  F_LID,
  F_SOLID,
  type PresetId,
  type Scene,
  type ViewMode,
} from "./types";

export type PresetInfo = {
  id: PresetId;
  name: string;
  blurb: string;
  aspect: number;
  baseNy: number;
  view: ViewMode;
  vectors: boolean;
};

export const PRESETS: readonly PresetInfo[] = [
  {
    id: "karman",
    name: "Kármán street",
    blurb: "Cylinder in a tunnel. Shedding should sit near St ≈ 0.20.",
    aspect: 2,
    baseNy: 96,
    view: "dye",
    vectors: false,
  },
  {
    id: "airfoil",
    name: "NACA 0012",
    blurb: "8° angle of attack. Watch the suction peak and wake.",
    aspect: 2,
    baseNy: 96,
    view: "speed",
    vectors: false,
  },
  {
    id: "cavity",
    name: "Lid cavity",
    blurb: "Ghia benchmark geometry. The lid drives a primary vortex.",
    aspect: 1,
    baseNy: 96,
    view: "vorticity",
    vectors: true,
  },
  {
    id: "jet",
    name: "Plane jet",
    blurb: "Slot inlet into a quiet chamber. Roll-up along the shear layer.",
    aspect: 2,
    baseNy: 96,
    view: "dye",
    vectors: false,
  },
  {
    id: "plume",
    name: "Thermal plume",
    blurb: "Buoyant scalar rising off a floor source.",
    aspect: 1,
    baseNy: 96,
    view: "dye",
    vectors: false,
  },
  {
    id: "kelvin",
    name: "Kelvin–Helmholtz",
    blurb: "Periodic shear layer. The braid rolls up into cat’s eyes.",
    aspect: 2,
    baseNy: 96,
    view: "dye",
    vectors: false,
  },
  {
    id: "ink",
    name: "Stirred ink",
    blurb: "Quiescent box. Drag in the field to shed dye and vorticity.",
    aspect: 1.5,
    baseNy: 96,
    view: "dye",
    vectors: false,
  },
];

export function presetInfo(id: PresetId): PresetInfo {
  return PRESETS.find((p) => p.id === id) ?? PRESETS[0]!;
}

function fillWalls(flags: Uint8Array, nx: number, ny: number): void {
  for (let i = 0; i < nx; i++) {
    flags[i] = F_SOLID;
    flags[i + (ny - 1) * nx] = F_SOLID;
  }
  for (let j = 0; j < ny; j++) {
    flags[j * nx] = F_SOLID;
    flags[nx - 1 + j * nx] = F_SOLID;
  }
}

function clearLeftRightGhosts(flags: Uint8Array, nx: number, ny: number): void {
  for (let j = 1; j < ny - 1; j++) {
    flags[j * nx] = 0;
    flags[nx - 1 + j * nx] = 0;
  }
}

function paintDisc(flags: Uint8Array, nx: number, ny: number, cx: number, cy: number, r: number): void {
  const r2 = r * r;
  const j0 = Math.max(1, Math.floor(cy - r - 1));
  const j1 = Math.min(ny - 2, Math.ceil(cy + r + 1));
  const i0 = Math.max(1, Math.floor(cx - r - 1));
  const i1 = Math.min(nx - 2, Math.ceil(cx + r + 1));
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const dx = i + 0.5 - cx;
      const dy = j + 0.5 - cy;
      if (dx * dx + dy * dy <= r2) flags[i + j * nx] = F_BODY;
    }
  }
}

function paintAirfoil(
  flags: Uint8Array,
  nx: number,
  ny: number,
  leX: number,
  leY: number,
  chord: number,
  aoaDeg: number,
): void {
  const t = 0.12;
  const aoa = (aoaDeg * Math.PI) / 180;
  const c = Math.cos(aoa);
  const s = Math.sin(aoa);
  const pad = chord * 0.6 + 2;
  const i0 = Math.max(1, Math.floor(leX - pad));
  const i1 = Math.min(nx - 2, Math.ceil(leX + pad));
  const j0 = Math.max(1, Math.floor(leY - pad));
  const j1 = Math.min(ny - 2, Math.ceil(leY + pad));
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const dx = i + 0.5 - leX;
      const dy = j + 0.5 - leY;
      const xl = c * dx - s * dy;
      const yl = s * dx + c * dy;
      const x = xl / chord;
      if (x < 0 || x > 1) continue;
      const yt =
        5 *
        t *
        (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
      if (Math.abs(yl / chord) <= yt) flags[i + j * nx] = F_BODY;
    }
  }
}

function addVortex(u: Float32Array, v: Float32Array, flags: Uint8Array, nx: number, ny: number, cx: number, cy: number, gamma: number, radius: number): void {
  const a2 = radius * radius;
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const id = i + j * nx;
      if (flags[id] === F_SOLID || flags[id] === F_BODY) continue;
      const dx = i - cx;
      const dy = j - cy;
      const r2 = dx * dx + dy * dy + 0.25;
      const r = Math.sqrt(r2);
      const vt = (gamma / (2 * Math.PI * r)) * (1 - Math.exp(-r2 / a2));
      u[id] = u[id]! - (vt * dy) / r;
      v[id] = v[id]! + (vt * dx) / r;
    }
  }
}

function seedStream(u: Float32Array, flags: Uint8Array, speed: number): void {
  for (let i = 0; i < u.length; i++) {
    if (flags[i] === F_SOLID || flags[i] === F_BODY) u[i] = 0;
    else u[i] = speed;
  }
}

export function buildScene(id: PresetId, nyIn: number): Scene {
  const info = presetInfo(id);
  const ny = Math.max(48, Math.round(nyIn));
  const nx = Math.max(48, Math.round(ny * info.aspect));
  const n = nx * ny;
  const flags = new Uint8Array(n);
  const u = new Float32Array(n);
  const v = new Float32Array(n);
  const dye = new Float32Array(n);
  fillWalls(flags, nx, ny);

  const scene: Scene = {
    id,
    nx,
    ny,
    flags,
    u,
    v,
    dye,
    inject: [],
    periodicX: false,
    uIn: 40,
    charLength: ny * 0.16,
    hasBody: false,
    dyeInlet: false,
    dyeDecay: 0,
    buoyancy: 0,
    vorticity: 0.15,
    re: 150,
    view: info.view,
    vectors: info.vectors,
    driven: false,
  };

  if (id === "karman") {
    clearLeftRightGhosts(flags, nx, ny);
    for (let j = 1; j < ny - 1; j++) flags[1 + j * nx] = F_INLET;
    const cx = nx * 0.22;
    const cy = ny * 0.5;
    const radius = ny * 0.085;
    paintDisc(flags, nx, ny, cx, cy, radius);
    scene.uIn = 42;
    scene.charLength = radius * 2;
    scene.hasBody = true;
    scene.re = 160;
    scene.vorticity = 0.22;
    seedStream(u, flags, scene.uIn);
    addVortex(u, v, flags, nx, ny, cx + radius * 2.4, cy + radius * 0.85, scene.uIn * radius * 1.4, radius * 0.9);
    addVortex(u, v, flags, nx, ny, cx + radius * 2.4, cy - radius * 0.85, -scene.uIn * radius * 1.4, radius * 0.9);
    const wire = Math.max(2, Math.round(cx - radius - 4));
    const jMid = Math.round(cy);
    for (let o = -2; o <= 2; o++) scene.inject.push({ i: wire, j: jMid + o, rate: 6 });
    for (let k = 0; k < nx; k++) {
      const id = k + Math.round(cy) * nx;
      const span = (k - wire) / (nx * 0.35);
      if (flags[id] === 0 && k > wire) dye[id] = Math.exp(-(span * span)) * 0.85;
    }
  } else if (id === "airfoil") {
    clearLeftRightGhosts(flags, nx, ny);
    for (let j = 1; j < ny - 1; j++) flags[1 + j * nx] = F_INLET;
    const chord = ny * 0.46;
    const leX = nx * 0.28;
    const leY = ny * 0.46;
    paintAirfoil(flags, nx, ny, leX, leY, chord, 8);
    scene.uIn = 36;
    scene.charLength = chord;
    scene.hasBody = true;
    scene.re = 480;
    scene.vorticity = 0.12;
    seedStream(u, flags, scene.uIn);
    const wire = Math.max(2, Math.round(leX - 6));
    const jWire = Math.round(leY);
    for (let o = -1; o <= 1; o++) scene.inject.push({ i: wire, j: jWire + o, rate: 8 });
  } else if (id === "cavity") {
    for (let i = 1; i < nx - 1; i++) flags[i + (ny - 2) * nx] = F_LID;
    scene.uIn = 28;
    scene.charLength = nx - 2;
    scene.re = 200;
    scene.vorticity = 0;
    scene.buoyancy = 0;
    for (let i = 1; i < nx - 1; i++) {
      u[i + (ny - 2) * nx] = scene.uIn;
      dye[i + (ny - 3) * nx] = 0.7;
    }
  } else if (id === "jet") {
    clearLeftRightGhosts(flags, nx, ny);
    for (let j = 1; j < ny - 1; j++) flags[1 + j * nx] = F_SOLID;
    const y0 = Math.round(ny * 0.42);
    const y1 = Math.round(ny * 0.58);
    for (let j = y0; j <= y1; j++) flags[1 + j * nx] = F_INLET;
    scene.uIn = 48;
    scene.charLength = Math.max(4, y1 - y0);
    scene.re = 240;
    scene.vorticity = 0.28;
    scene.dyeInlet = true;
    for (let j = y0; j <= y1; j++) {
      u[1 + j * nx] = scene.uIn;
      dye[1 + j * nx] = 1;
    }
  } else if (id === "plume") {
    scene.uIn = 0;
    scene.charLength = ny * 0.2;
    scene.re = 80;
    scene.vorticity = 0.18;
    scene.buoyancy = 55;
    scene.dyeDecay = 0.12;
    const cx = Math.round(nx * 0.5);
    const j = 2;
    for (let i = cx - 4; i <= cx + 4; i++) {
      scene.inject.push({ i, j, rate: 2.2 });
      dye[i + j * nx] = 1;
    }
  } else if (id === "kelvin") {
    for (let j = 1; j < ny - 1; j++) {
      flags[j * nx] = 0;
      flags[nx - 1 + j * nx] = 0;
    }
    scene.periodicX = true;
    scene.uIn = 22;
    scene.charLength = ny * 0.25;
    scene.re = 800;
    scene.vorticity = 0.05;
    scene.dyeDecay = 0;
    const s = 3.2;
    for (let j = 1; j < ny - 1; j++) {
      const shear = Math.tanh((j - ny / 2) / s);
      for (let i = 0; i < nx; i++) {
        const id = i + j * nx;
        if (flags[id] === F_SOLID) continue;
        u[id] = scene.uIn * shear;
        const band = (j - ny / 2) / (s * 2);
        v[id] =
          scene.uIn *
          0.06 *
          Math.sin((2 * Math.PI * i * 3) / nx) *
          Math.exp(-(band * band));
        dye[id] = 0.5 + 0.5 * shear;
      }
    }
  } else {
    scene.uIn = 0;
    scene.charLength = ny * 0.25;
    scene.re = 400;
    scene.vorticity = 0.2;
    scene.dyeDecay = 0.04;
    const cx = nx * 0.5;
    const cy = ny * 0.55;
    const rad = ny * 0.12;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const dx = (i - cx) / rad;
        const dy = (j - cy) / rad;
        dye[i + j * nx] = Math.exp(-(dx * dx + dy * dy));
      }
    }
  }

  scene.driven = id === "karman" || id === "airfoil" || id === "cavity" || id === "jet";

  for (let i = 0; i < n; i++) {
    if (flags[i] === F_SOLID || flags[i] === F_BODY) {
      u[i] = 0;
      v[i] = 0;
      dye[i] = 0;
    }
  }

  return scene;
}
