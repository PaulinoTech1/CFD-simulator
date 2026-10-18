export type ChartSeries = {
  name: string;
  values: ArrayLike<number>;
  color: string;
};

/** Draws one or more series against their index. NaN breaks the line. */
export function paintChart(canvas: HTMLCanvasElement, series: readonly ChartSeries[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.max(2, Math.round(canvas.clientWidth * dpr));
  const height = Math.max(2, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101412";
  ctx.fillRect(0, 0, width, height);
  const pad = 8 * dpr;
  let lo = Infinity;
  let hi = -Infinity;
  let count = 0;
  for (const s of series) {
    for (let i = 0; i < s.values.length; i++) {
      const v = s.values[i]!;
      if (!Number.isFinite(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
      count += 1;
    }
  }
  if (count === 0) {
    ctx.fillStyle = "#8d9a92";
    ctx.font = `${11 * dpr}px ui-monospace, monospace`;
    ctx.fillText("no samples yet", pad, height / 2);
    return;
  }
  if (hi - lo < 1e-8) {
    lo -= 1;
    hi += 1;
  }
  const span = hi - lo;
  lo -= span * 0.08;
  hi += span * 0.08;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const yOf = (v: number) => pad + ((hi - v) / (hi - lo)) * plotH;
  const zero = yOf(0);
  if (zero >= pad && zero <= height - pad) {
    ctx.strokeStyle = "#2a3330";
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(pad, zero);
    ctx.lineTo(width - pad, zero);
    ctx.stroke();
  }
  series.forEach((s, ordinal) => {
    const n = s.values.length;
    if (n < 2) return;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    let drawing = false;
    for (let i = 0; i < n; i++) {
      const v = s.values[i]!;
      if (!Number.isFinite(v)) {
        drawing = false;
        continue;
      }
      const x = pad + (n === 1 ? 0 : (i / (n - 1)) * plotW);
      const y = yOf(v);
      if (!drawing) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      drawing = true;
    }
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.font = `${10 * dpr}px ui-monospace, monospace`;
    ctx.fillText(s.name, pad + ordinal * 72 * dpr, pad + 10 * dpr);
  });
}
