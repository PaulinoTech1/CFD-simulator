import { F_BODY, F_FLUID, F_INLET, F_LID, F_OUTLET, F_SOLID, type ViewMode } from "./types.ts";

export type Preset3 = "karman" | "airfoil" | "cavity" | "plume" | "jet" | "kelvin" | "ink";

export type BodyKind = "cylinder" | "airfoil" | "sphere" | "box" | "cone" | "plate" | "none";

export type Splat3 = {
  x: number;
  y: number;
  z: number;
  du: number;
  dv: number;
  dw: number;
  dye: number;
  radius: number;
};

export type Scene3 = {
  id: Preset3;
  nx: number;
  ny: number;
  nz: number;
  flags: Uint8Array;
  u: Float32Array;
  v: Float32Array;
  w: Float32Array;
  dye: Float32Array;
  uIn: number;
  re: number;
  charLength: number;
  hasBody: boolean;
  bodyKind: BodyKind;
  dyeDecay: number;
  buoyancy: number;
  vorticity: number;
  view: ViewMode;
  cx: number;
  cy: number;
  cz: number;
  radius: number;
  aoa: number;
};

export type Preset3Info = {
  id: Preset3;
  name: string;
  blurb: string;
  view: ViewMode;
};

export const PRESETS3: readonly Preset3Info[] = [
  {
    id: "karman",
    name: "Kármán street",
    blurb: "Spanwise cylinder. The mid-span slice is the street. Shedding wants St near 0.20 once the grid can hold it.",
    view: "dye",
  },
  {
    id: "airfoil",
    name: "NACA 0012",
    blurb: "The section is extruded through the span at 8°. On a coarse grid the thickness is held to one cell.",
    view: "speed",
  },
  {
    id: "cavity",
    name: "Lid cavity",
    blurb: "The lid slides in x. One primary vortex, then corner eddies if the span can hold them.",
    view: "vorticity",
  },
  {
    id: "plume",
    name: "Thermal plume",
    blurb: "A hot disk on the floor. Buoyancy is the only force, so the scalar is the momentum source.",
    view: "dye",
  },
  {
    id: "jet",
    name: "Round jet",
    blurb: "A disk inlet into a quiet box. The shear layer rolls up around the jet.",
    view: "dye",
  },
  {
    id: "kelvin",
    name: "Kelvin–Helmholtz",
    blurb: "A tanh shear across the height, seeded with one wave. The box is closed, so the roll-up is not periodic.",
    view: "dye",
  },
  {
    id: "ink",
    name: "Stirred ink",
    blurb: "A closed volume at rest. Switch to stir and drag to write dye into the mid-span plane.",
    view: "dye",
  },
];

export function preset3Info(id: Preset3): Preset3Info {
  return PRESETS3.find((p) => p.id === id) ?? PRESETS3[0]!;
}

function idAt(i: number, j: number, k: number, nx: number, ny: number): number {
  return i + nx * (j + ny * k);
}

function nacaHalf(x: number, chord: number): number {
  const t = 0.12;
  const yt = 5 * t * (0.2969 * Math.sqrt(Math.max(x, 0)) - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1015 * x * x * x * x);
  return Math.max(1.25, yt * chord);
}

function inAirfoil(i: number, j: number, cx: number, cy: number, chord: number, aoaDeg: number): boolean {
  const a = (aoaDeg * Math.PI) / 180;
  const dx = i - cx;
  const dy = j - cy;
  const xl = dx * Math.cos(a) - dy * Math.sin(a);
  const yl = dx * Math.sin(a) + dy * Math.cos(a);
  if (xl < -0.5 || xl > chord + 0.5) return false;
  const x = Math.min(1, Math.max(0, xl / chord));
  return Math.abs(yl) <= nacaHalf(x, chord);
}

export function buildScene3(id: Preset3, spanIn: number): Scene3 {
  const span = Math.max(12, Math.round(spanIn));
  const info = preset3Info(id);
  const nx = span * 2;
  const ny = span;
  const nz = span;
  const n = nx * ny * nz;
  const flags = new Uint8Array(n);
  const u = new Float32Array(n);
  const v = new Float32Array(n);
  const w = new Float32Array(n);
  const dye = new Float32Array(n);
  const cx = nx * 0.28;
  const cy = (ny - 1) / 2;
  const cz = (nz - 1) / 2;
  const radius = Math.max(2.4, span * 0.18);

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const wall = j === 0 || j === ny - 1 || k === 0 || k === nz - 1;
        flags[idAt(i, j, k, nx, ny)] = wall ? F_SOLID : F_FLUID;
      }
    }
  }

  let uIn = 1;
  let re = 180;
  let buoyancy = 0;
  let vorticity = 0.16;
  let dyeDecay = 0.12;
  let hasBody = false;
  let bodyKind: BodyKind = "none";
  let charLength = radius * 2;
  let aoa = 0;
  const chord = Math.max(8, span * 0.72);

  const openTunnel = () => {
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        flags[idAt(0, j, k, nx, ny)] = F_INLET;
        flags[idAt(nx - 1, j, k, nx, ny)] = F_OUTLET;
      }
    }
  };

  if (id === "karman") {
    hasBody = true;
    bodyKind = "cylinder";
    re = 180;
    openTunnel();
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const dx = i - cx;
          const dy = j - cy;
          if (dx * dx + dy * dy <= radius * radius) flags[idAt(i, j, k, nx, ny)] = F_BODY;
        }
      }
    }
  } else if (id === "airfoil") {
    hasBody = true;
    bodyKind = "airfoil";
    re = 400;
    aoa = 8;
    charLength = chord;
    vorticity = 0.08;
    openTunnel();
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 2; i++) {
          if (inAirfoil(i, j, cx, cy, chord, aoa)) flags[idAt(i, j, k, nx, ny)] = F_BODY;
        }
      }
    }
  } else if (id === "cavity") {
    uIn = 1;
    re = 100;
    vorticity = 0;
    dyeDecay = 0.04;
    charLength = ny - 2;
    for (let k = 1; k < nz - 1; k++) {
      for (let i = 1; i < nx - 1; i++) flags[idAt(i, ny - 1, k, nx, ny)] = F_LID;
    }
  } else if (id === "plume") {
    uIn = 0;
    re = 40;
    buoyancy = 2.4;
    vorticity = 0.04;
    dyeDecay = 0.06;
    charLength = ny * 0.45;
    for (let k = 1; k < nz - 1; k++) {
      for (let i = 1; i < nx - 1; i++) flags[idAt(i, ny - 1, k, nx, ny)] = F_OUTLET;
    }
    const r2 = radius * radius;
    for (let k = 1; k < nz - 1; k++) {
      for (let i = 1; i < nx - 1; i++) {
        if ((i - nx * 0.5) * (i - nx * 0.5) + (k - cz) * (k - cz) > r2) continue;
        for (let j = 1; j < 3; j++) dye[idAt(i, j, k, nx, ny)] = 1;
      }
    }
  } else if (id === "jet") {
    uIn = 1.4;
    re = 300;
    vorticity = 0.1;
    const jr = Math.max(2, span * 0.14);
    charLength = jr * 2;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        const inside = (j - cy) * (j - cy) + (k - cz) * (k - cz) <= jr * jr;
        flags[idAt(0, j, k, nx, ny)] = inside ? F_INLET : F_SOLID;
        flags[idAt(nx - 1, j, k, nx, ny)] = F_OUTLET;
      }
    }
  } else if (id === "kelvin") {
    uIn = 0;
    re = 500;
    vorticity = 0.02;
    dyeDecay = 0.02;
    charLength = ny * 0.5;
    const delta = 1.5;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const s = Math.tanh((j - cy) / delta);
          const id = idAt(i, j, k, nx, ny);
          u[id] = 0.85 * s;
          v[id] = 0.12 * Math.sin((2 * Math.PI * i) / nx) * Math.exp(-(((j - cy) / (2 * delta)) ** 2));
          dye[id] = s > 0 ? 1 : 0.02;
        }
      }
    }
  } else {
    uIn = 0;
    re = 80;
    vorticity = 0.08;
    dyeDecay = 0.03;
    charLength = ny * 0.5;
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = 1; i < nx - 1; i++) {
          const dx = i - nx * 0.45;
          const dy = j - cy;
          const dz = k - cz;
          if (dx * dx + dy * dy + dz * dz < radius * radius) dye[idAt(i, j, k, nx, ny)] = 1;
        }
      }
    }
  }

  if (uIn !== 0) {
    for (let id = 0; id < n; id++) {
      if (flags[id] === F_FLUID || flags[id] === F_OUTLET) u[id] = uIn;
    }
  }
  if (id === "karman") {
    for (let k = 1; k < nz - 1; k++) {
      for (let j = 1; j < ny - 1; j++) {
        for (let i = Math.ceil(cx + radius); i < nx - 1; i++) {
          const id = idAt(i, j, k, nx, ny);
          if (flags[id] !== F_FLUID) continue;
          v[id] = uIn * 0.07 * Math.sin((Math.PI * (i - cx)) / nx);
        }
      }
    }
  }

  return {
    id,
    nx,
    ny,
    nz,
    flags,
    u,
    v,
    w,
    dye,
    uIn,
    re,
    charLength,
    hasBody,
    bodyKind,
    dyeDecay,
    buoyancy,
    vorticity,
    view: info.view,
    cx,
    cy,
    cz,
    radius: id === "airfoil" ? chord * 0.5 : radius,
    aoa,
  };
}
