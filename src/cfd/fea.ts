import type { Fluid3 } from "./solver3d.ts";
import { F_BODY, F_SOLID } from "./types.ts";

export type FeaReport = {
  nodes: number;
  triangles: number;
  maxVonMises: number;
  maxDisp: number;
  meanPressure: number;
  note: string;
  outline: { x: number; y: number }[];
};

const EMPTY: FeaReport = {
  nodes: 0,
  triangles: 0,
  maxVonMises: 0,
  maxDisp: 0,
  meanPressure: 0,
  note: "Stamp a solid. The section solve needs a mid-span cross-section.",
  outline: [],
};

/** Plane-stress triangles on the mid-span body. Pressure on the wetted edge is the load. */
export function sectionFea(sim: Fluid3, young: number, poisson: number): FeaReport {
  const { flags, p, nx, ny, nz } = sim;
  const k = (nz / 2) | 0;
  const stride = nx * ny > 48 * 24 ? 2 : 1;
  const nodes: { i: number; j: number }[] = [];
  const index = new Map<number, number>();
  for (let j = 1; j < ny - 1; j += stride) {
    for (let i = 1; i < nx - 1; i += stride) {
      if (flags[i + nx * (j + ny * k)] !== F_BODY) continue;
      index.set(i + j * nx, nodes.length);
      nodes.push({ i, j });
      if (nodes.length >= 90) break;
    }
    if (nodes.length >= 90) break;
  }
  if (nodes.length < 4) return EMPTY;

  const tris: [number, number, number][] = [];
  const at = (i: number, j: number) => index.get(i + j * nx);
  for (let j = 1; j < ny - 1; j += stride) {
    for (let i = 1; i < nx - 1; i += stride) {
      const a = at(i, j);
      const b = at(i + stride, j);
      const c = at(i, j + stride);
      const d = at(i + stride, j + stride);
      if (a !== undefined && b !== undefined && c !== undefined) tris.push([a, b, c]);
      if (b !== undefined && d !== undefined && c !== undefined) tris.push([b, d, c]);
    }
  }
  if (tris.length < 1) return { ...EMPTY, nodes: nodes.length, note: "The section is too thin to mesh. Use a thicker solid." };

  const nu = Math.min(0.45, Math.max(0.05, poisson));
  const E = Math.max(0.05, young);
  const C = E / (1 - nu * nu);
  const D = [
    [C, C * nu, 0],
    [C * nu, C, 0],
    [0, 0, C * (1 - nu) / 2],
  ];
  const n = nodes.length;
  const nd = n * 2;
  const A = Array.from({ length: nd }, () => new Float64Array(nd));
  const b = new Float64Array(nd);
  let load = 0;
  let loadN = 0;

  for (const tri of tris) {
    const xy = tri.map((id) => ({ x: nodes[id]!.i, y: nodes[id]!.j }));
    const x = xy.map((q) => q.x);
    const y = xy.map((q) => q.y);
    const b2 = [y[1]! - y[2]!, y[2]! - y[0]!, y[0]! - y[1]!];
    const c2 = [x[2]! - x[1]!, x[0]! - x[2]!, x[1]! - x[0]!];
    const area2 = x[0]! * (y[1]! - y[2]!) + x[1]! * (y[2]! - y[0]!) + x[2]! * (y[0]! - y[1]!);
    const area = Math.abs(area2) * 0.5;
    if (area < 1e-6) continue;
    const s = 1 / (2 * area);
    const B = [
      [b2[0]! * s, 0, b2[1]! * s, 0, b2[2]! * s, 0],
      [0, c2[0]! * s, 0, c2[1]! * s, 0, c2[2]! * s],
      [c2[0]! * s, b2[0]! * s, c2[1]! * s, b2[1]! * s, c2[2]! * s, b2[2]! * s],
    ];
    const dof = [tri[0]! * 2, tri[0]! * 2 + 1, tri[1]! * 2, tri[1]! * 2 + 1, tri[2]! * 2, tri[2]! * 2 + 1];
    for (let r = 0; r < 6; r++) {
      for (let c = 0; c < 6; c++) {
        let ke = 0;
        for (let i = 0; i < 3; i++) {
          for (let j = 0; j < 3; j++) ke += B[i]![r]! * D[i]![j]! * B[j]![c]!;
        }
        A[dof[r]!]![dof[c]!]! += ke * area;
      }
    }
  }

  for (let id = 0; id < n; id++) {
    const i = nodes[id]!.i;
    const j = nodes[id]!.j;
    const nbr = [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ];
    for (const [di, dj] of nbr) {
      const ni = i + di * stride;
      const nj = j + dj * stride;
      if (ni < 0 || nj < 0 || ni >= nx || nj >= ny) continue;
      const fid = ni + nx * (nj + ny * k);
      const f = flags[fid]!;
      if (f === F_BODY || f === F_SOLID) continue;
      const press = p[fid] ?? 0;
      b[id * 2] += -press * di;
      b[id * 2 + 1] += -press * dj;
      load += press;
      loadN++;
    }
  }

  const adj: number[][] = Array.from({ length: n }, () => []);
  for (const tri of tris) {
    for (let e = 0; e < 3; e++) {
      const a = tri[e]!;
      const b = tri[(e + 1) % 3]!;
      adj[a]!.push(b);
      adj[b]!.push(a);
    }
  }
  const seen = new Uint8Array(n);
  const fixed: number[] = [];
  for (let start = 0; start < n; start++) {
    if (seen[start]) continue;
    const stack = [start];
    const members: number[] = [];
    seen[start] = 1;
    while (stack.length) {
      const id = stack.pop()!;
      members.push(id);
      for (const nb of adj[id]!) {
        if (seen[nb]) continue;
        seen[nb] = 1;
        stack.push(nb);
      }
    }
    let ax = 0;
    let ay = 0;
    for (const id of members) {
      ax += nodes[id]!.i;
      ay += nodes[id]!.j;
    }
    ax /= members.length;
    ay /= members.length;
    let pin = members[0]!;
    let best = Infinity;
    for (const id of members) {
      const d = (nodes[id]!.i - ax) ** 2 + (nodes[id]!.j - ay) ** 2;
      if (d < best) {
        best = d;
        pin = id;
      }
    }
    fixed.push(pin * 2, pin * 2 + 1);
    if (members.length > 1) {
      let pin2 = members[0] === pin ? members[1]! : members[0]!;
      best = -1;
      for (const id of members) {
        if (id === pin) continue;
        const d = Math.abs(nodes[id]!.j - ay);
        if (d > best) {
          best = d;
          pin2 = id;
        }
      }
      fixed.push(pin2 * 2);
    }
  }
  for (const dof of fixed) {
    for (let c = 0; c < nd; c++) A[dof]![c] = 0;
    for (let r = 0; r < nd; r++) A[r]![dof] = 0;
    A[dof]![dof] = 1;
    b[dof] = 0;
  }
  for (let r = 0; r < nd; r++) {
    let mag = 0;
    for (let c = 0; c < nd; c++) mag += Math.abs(A[r]![c]!);
    if (mag < 1e-10) {
      A[r]![r] = 1;
      b[r] = 0;
    }
  }

  const x = solve(A, b);
  if (!x) return { ...EMPTY, nodes: n, triangles: tris.length, note: "The stiffness matrix was singular. The section may be only one cell thick." };

  let maxDisp = 0;
  for (let id = 0; id < n; id++) maxDisp = Math.max(maxDisp, Math.hypot(x[id * 2]!, x[id * 2 + 1]!));

  let maxVm = 0;
  for (const tri of tris) {
    const xy = tri.map((id) => nodes[id]!);
    const xs = xy.map((q) => q.i);
    const ys = xy.map((q) => q.j);
    const b2 = [ys[1]! - ys[2]!, ys[2]! - ys[0]!, ys[0]! - ys[1]!];
    const c2 = [xs[2]! - xs[1]!, xs[0]! - xs[2]!, xs[1]! - xs[0]!];
    const area2 = xs[0]! * (ys[1]! - ys[2]!) + xs[1]! * (ys[2]! - ys[0]!) + xs[2]! * (ys[0]! - ys[1]!);
    const area = Math.abs(area2) * 0.5;
    if (area < 1e-6) continue;
    const s = 1 / (2 * area);
    const ue = [
      x[tri[0]! * 2]!,
      x[tri[0]! * 2 + 1]!,
      x[tri[1]! * 2]!,
      x[tri[1]! * 2 + 1]!,
      x[tri[2]! * 2]!,
      x[tri[2]! * 2 + 1]!,
    ];
    const ex = s * (b2[0]! * ue[0]! + b2[1]! * ue[2]! + b2[2]! * ue[4]!);
    const ey = s * (c2[0]! * ue[1]! + c2[1]! * ue[3]! + c2[2]! * ue[5]!);
    const gxy = s * (c2[0]! * ue[0]! + b2[0]! * ue[1]! + c2[1]! * ue[2]! + b2[1]! * ue[3]! + c2[2]! * ue[4]! + b2[2]! * ue[5]!);
    const sx = D[0]![0]! * ex + D[0]![1]! * ey;
    const sy = D[1]![0]! * ex + D[1]![1]! * ey;
    const txy = D[2]![2]! * gxy;
    const vm = Math.sqrt(Math.max(0, sx * sx - sx * sy + sy * sy + 3 * txy * txy));
    if (vm > maxVm) maxVm = vm;
  }

  const gain = maxDisp > 1e-8 ? 2.2 / maxDisp : 1;
  const outline = nodes
    .filter((node) => {
      const nbr = [
        [stride, 0],
        [-stride, 0],
        [0, stride],
        [0, -stride],
      ];
      return nbr.some(([di, dj]) => {
        const f = flags[node.i + di + nx * (node.j + dj + ny * k)];
        return f !== F_BODY && f !== F_SOLID;
      });
    })
    .map((node, listIndex) => {
      const id = index.get(node.i + node.j * nx)!;
      return { x: node.i + x[id * 2]! * gain, y: node.j + x[id * 2 + 1]! * gain, listIndex };
    });

  return {
    nodes: n,
    triangles: tris.length,
    maxVonMises: maxVm,
    maxDisp,
    meanPressure: loadN ? load / loadN : 0,
    note:
      stride === 1
        ? "Plane stress, one support per solid. Displacement is in cell widths at this modulus, not pascals. The sketch is exaggerated."
        : "Section coarsened to stay interactive. Displacement is in cell widths. The sketch is exaggerated.",
    outline,
  };
}

/** Constant-strain triangle. Imposed uniaxial strain must return σx = E εx and σy = 0. */
export function patchPlaneStress(young: number, poisson: number, eps: number): { sigmaX: number; sigmaY: number; tau: number } {
  const nu = poisson;
  const E = young;
  const C = E / (1 - nu * nu);
  const ex = eps;
  const ey = -nu * eps;
  const sx = C * ex + C * nu * ey;
  const sy = C * nu * ex + C * ey;
  return { sigmaX: sx, sigmaY: sy, tau: 0 };
}

function solve(A: Float64Array[], bIn: Float64Array): Float64Array | null {
  const n = bIn.length;
  const b = Float64Array.from(bIn);
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(A[col]![col]!);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(A[row]![col]!);
      if (v > best) {
        best = v;
        piv = row;
      }
    }
    if (best < 1e-14) return null;
    if (piv !== col) {
      const row = A[col]!;
      A[col] = A[piv]!;
      A[piv] = row;
      const tmp = b[col]!;
      b[col] = b[piv]!;
      b[piv] = tmp;
    }
    const div = A[col]![col]!;
    for (let row = col + 1; row < n; row++) {
      const f = A[row]![col]! / div;
      if (f === 0) continue;
      for (let k = col; k < n; k++) A[row]![k]! -= f * A[col]![k]!;
      b[row] = b[row]! - f * b[col]!;
    }
  }
  const x = new Float64Array(n);
  for (let col = n - 1; col >= 0; col--) {
    let sum = b[col]!;
    for (let k = col + 1; k < n; k++) sum -= A[col]![k]! * x[k]!;
    const diag = A[col]![col]!;
    if (Math.abs(diag) < 1e-14) return null;
    x[col] = sum / diag;
  }
  return x;
}
