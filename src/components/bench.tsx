import { useEffect, useRef, useState } from "react";
import { Copy, Download, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { CPU, GPU, formatBytes, planLaunch } from "@/cfd/device";
import { KERNELS } from "@/cfd/kernels";
import { PRESETS, buildScene, presetInfo } from "@/cfd/presets";
import { LUT_DIVERGE, LUT_INK, LUT_SPEED, lutAt } from "@/cfd/colormap";
import { createDrawState, drawField, drawSpark, ensureTarget, type FrameTarget } from "@/cfd/render";
import { FluidSim } from "@/cfd/solver";
import type { PresetId, SimParams, SimStats, Splat, ToolMode, ViewMode } from "@/cfd/types";

const HEIGHTS = [72, 96, 120, 160] as const;

const HEIGHT_NAME: Record<(typeof HEIGHTS)[number], string> = {
  72: "Draft",
  96: "Bench",
  120: "Fine",
  160: "5900X",
};

type Panel = "field" | "rig" | "kernels";

type Hud = SimStats & { stepMs: number; cells: number; hover: string };

const ZERO: Hud = {
  time: 0,
  dt: 0,
  ke: 0,
  maxSpeed: 0,
  maxAbsOmega: 0,
  fx: 0,
  fy: 0,
  cd: 0,
  cl: 0,
  re: 160,
  cfl: 0,
  strouhal: 0,
  steps: 0,
  stepMs: 0,
  cells: 0,
  hover: "move over the field",
};

function num(n: number, d = 2): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export function Bench() {
  const [presetId, setPresetId] = useState<PresetId>("karman");
  const [ny, setNy] = useState(96);
  const [running, setRunning] = useState(true);
  const [view, setView] = useState<ViewMode>("dye");
  const [tool, setTool] = useState<ToolMode>("stir");
  const [vectors, setVectors] = useState(false);
  const [particles, setParticles] = useState(true);
  const [panel, setPanel] = useState<Panel>("field");
  const [kernelId, setKernelId] = useState(KERNELS[0]!.id);
  const [copied, setCopied] = useState(false);
  const [re, setRe] = useState(160);
  const [uIn, setUIn] = useState(42);
  const [vorticity, setVorticity] = useState(0.22);
  const [buoyancy, setBuoyancy] = useState(0);
  const [iterations, setIterations] = useState(16);
  const [driven, setDriven] = useState(true);
  const [hud, setHud] = useState<Hud>(ZERO);
  const [traceName, setTraceName] = useState("Cl");

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sparkRef = useRef<HTMLCanvasElement>(null);
  const legendRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<FluidSim | null>(null);
  const targetRef = useRef<FrameTarget | null>(null);
  const drawState = useRef(createDrawState());
  const splats = useRef<Splat[]>([]);
  const down = useRef(false);
  const lastCell = useRef({ x: 0, y: 0 });
  const hoverCell = useRef<{ x: number; y: number } | null>(null);
  const stepOnce = useRef(false);
  const decayRef = useRef(0);
  const spark = useRef(new Float32Array(160));
  const sparkN = useRef(0);
  const sparkAt = useRef(0);
  const resetRef = useRef<() => void>(() => {});

  const live = useRef({
    view,
    vectors,
    particles,
    running,
    tool,
    re,
    uIn,
    vorticity,
    buoyancy,
    iterations,
  });
  live.current = { view, vectors, particles, running, tool, re, uIn, vorticity, buoyancy, iterations };

  const info = presetInfo(presetId);
  const kernel = KERNELS.find((k) => k.id === kernelId) ?? KERNELS[0]!;

  function loadPreset(id: PresetId, height: number) {
    const scene = buildScene(id, height);
    const next = presetInfo(id);
    setPresetId(id);
    setNy(scene.ny);
    setView(scene.view);
    setVectors(scene.vectors);
    setRe(scene.re);
    setUIn(scene.uIn);
    setVorticity(scene.vorticity);
    setBuoyancy(scene.buoyancy);
    setDriven(scene.driven);
    setTraceName(scene.hasBody ? "Cl" : "ω");
    decayRef.current = scene.dyeDecay;
    return { scene, blurb: next.blurb };
  }

  resetRef.current = () => {
    const scene = buildScene(presetId, ny);
    decayRef.current = scene.dyeDecay;
    simRef.current?.load(scene);
    drawState.current = createDrawState();
    sparkN.current = 0;
    sparkAt.current = 0;
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = buildScene(presetId, ny);
    decayRef.current = scene.dyeDecay;
    const sim = new FluidSim();
    const params = (): SimParams => ({
      re: live.current.re,
      uIn: live.current.uIn,
      vorticity: live.current.vorticity,
      buoyancy: live.current.buoyancy,
      iterations: live.current.iterations,
      dyeDecay: decayRef.current,
    });
    sim.load(scene);
    let lastSt: SimStats | null = null;
    const warmStart = performance.now();
    let warmed = 0;
    while (warmed < 1.1 && performance.now() - warmStart < 450) {
      const h = sim.cflDt(0.8);
      lastSt = sim.step(h, params(), []);
      warmed += lastSt.dt;
    }
    sim.snapshotParticles();
    simRef.current = sim;
    targetRef.current = null;
    sparkN.current = 0;
    sparkAt.current = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.max(2, Math.round(rect.width * dpr));
      const h = Math.max(2, Math.round(rect.height * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      const spark = sparkRef.current;
      if (spark) {
        const sr = spark.getBoundingClientRect();
        const sw = Math.max(2, Math.round(sr.width * dpr));
        const sh = Math.max(2, Math.round(sr.height * dpr));
        if (spark.width !== sw || spark.height !== sh) {
          spark.width = sw;
          spark.height = sh;
        }
      }
    };
    resize();
    const ro = new ResizeObserver(resize);

    let raf = 0;
    let last = performance.now();
    let acc = 0;
    let hudAt = 0;
    let stepMs = 0;

    const frame = (now: number) => {
      const wall = Math.min(0.05, (now - last) / 1000);
      last = now;
      const want = live.current.running ? wall : 0;
      acc += want;
      if (stepOnce.current) {
        acc = Math.max(acc, sim.cflDt(0.8));
        stepOnce.current = false;
      }
      let guard = 0;
      const tStep = performance.now();
      sim.snapshotParticles();
      while (acc > 1e-4 && guard < 3) {
        const h = Math.min(sim.cflDt(0.8), acc);
        const st = sim.step(h, params(), splats.current);
        lastSt = st;
        splats.current = [];
        acc -= st.dt;
        guard += 1;
        const trace = sim.hasBody ? st.cl : st.maxAbsOmega;
        spark.current[sparkAt.current] = trace;
        sparkAt.current = (sparkAt.current + 1) % spark.current.length;
        if (sparkN.current < spark.current.length) sparkN.current += 1;
      }
      if (guard === 3) acc = 0;
      if (guard > 0) stepMs = (performance.now() - tStep) / guard;

      targetRef.current = ensureTarget(sim.nx, sim.ny, targetRef.current);
      drawField(
        canvas,
        sim,
        targetRef.current,
        live.current.view,
        live.current.vectors,
        live.current.particles,
        drawState.current,
      );
      const sparkCanvas = sparkRef.current;
      if (sparkCanvas && sparkCanvas.width > 0) {
        drawSpark(sparkCanvas, spark.current, sparkN.current, sparkAt.current);
      }

      if (now - hudAt > 180 && lastSt) {
        hudAt = now;
        const hov = hoverCell.current;
        let hover = "move over the field";
        if (hov) {
          const i = Math.max(0, Math.min(sim.nx - 1, hov.x | 0));
          const j = Math.max(0, Math.min(sim.ny - 1, hov.y | 0));
          const id = i + j * sim.nx;
          hover = `i ${i}  j ${j}   u ${num(sim.u[id] ?? 0, 1)}   v ${num(sim.v[id] ?? 0, 1)}   ω ${num(sim.omega[id] ?? 0, 2)}`;
        }
        setHud({
          ...lastSt,
          stepMs,
          cells: sim.nx * sim.ny,
          hover,
          re: live.current.re,
        });
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    ro.observe(canvas);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      if (simRef.current === sim) simRef.current = null;
    };
  }, [presetId, ny]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        setRunning((v) => !v);
      } else if (e.key === "r" || e.key === "R") {
        resetRef.current();
      } else if (e.key === ".") {
        stepOnce.current = true;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const canvas = legendRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const lut = view === "speed" ? LUT_SPEED : view === "dye" ? LUT_INK : LUT_DIVERGE;
    const ink = { r: 0, g: 0, b: 0 };
    canvas.width = 128;
    canvas.height = 12;
    for (let i = 0; i < 128; i++) {
      lutAt(lut, i / 127, ink);
      ctx.fillStyle = `rgb(${ink.r}, ${ink.g}, ${ink.b})`;
      ctx.fillRect(i, 0, 1, 12);
    }
  }, [view]);

  function cellOf(e: React.PointerEvent<HTMLCanvasElement>) {
    const sim = simRef.current;
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = sim?.nx ?? 2;
    const nyG = sim?.ny ?? 2;
    const x = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * nx;
    const y = (1 - (e.clientY - rect.top) / Math.max(rect.height, 1)) * nyG;
    return { x, y };
  }

  function interact(e: React.PointerEvent<HTMLCanvasElement>) {
    const sim = simRef.current;
    if (!sim || !down.current) return;
    const c = cellOf(e);
    const mode = live.current.tool;
    if (mode === "stir") {
      const prev = lastCell.current;
      splats.current.push({
        x: c.x,
        y: c.y,
        du: (c.x - prev.x) * 2.4,
        dv: (c.y - prev.y) * 2.4,
        dye: 0.65,
        radius: Math.max(3, sim.ny * 0.05),
      });
      lastCell.current = c;
    } else if (mode === "wall" || mode === "erase") {
      sim.paint(c.x | 0, c.y | 0, mode === "wall", Math.max(2, (sim.ny * 0.028) | 0));
    }
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    down.current = true;
    lastCell.current = cellOf(e);
    interact(e);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    hoverCell.current = cellOf(e);
    if (down.current) interact(e);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    down.current = false;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
  }

  async function copyKernel() {
    try {
      await navigator.clipboard.writeText(kernel.source);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  function downloadKernel() {
    const blob = new Blob([kernel.source], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = kernel.filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  const plan = planLaunch(Math.round(ny * info.aspect), ny);

  return (
    <div className="min-h-screen bg-bg text-fg">
      <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-border bg-bg px-4 py-3">
        <div className="min-w-40">
          <p className="font-mono text-xs tracking-widest text-primary">GA102 · SM 8.6</p>
          <h1 className="text-balance text-xl font-semibold tracking-tight">Karman</h1>
        </div>
        <p className="hidden max-w-md text-pretty text-sm text-muted md:block">
          2D incompressible Navier–Stokes. This view is the reference solve. The CUDA and PTX are the same method for the 3090 Ti.
        </p>
        <div className="ml-auto flex flex-wrap gap-2">
          <button type="button" className={btn} onClick={() => setRunning((v) => !v)}>
            {running ? <Pause className="size-4" /> : <Play className="size-4" />}
            {running ? "Pause" : "Run"}
          </button>
          <button
            type="button"
            className={btn}
            onClick={() => {
              stepOnce.current = true;
            }}
          >
            <SkipForward className="size-4" />
            Step
          </button>
          <button type="button" className={btn} onClick={() => resetRef.current()}>
            <RotateCcw className="size-4" />
            Reset
          </button>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4">
        <section className="relative min-w-0 p-3 lg:col-span-3 lg:p-4">
          <div className="relative overflow-hidden rounded-lg border border-border bg-surface">
            <canvas
              ref={canvasRef}
              className="block w-full touch-none bg-bg"
              style={{ aspectRatio: `${info.aspect} / 1` }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            />
            <div className="pointer-events-none absolute top-3 left-3 font-mono text-xs leading-5 text-fg">
              <p className="rounded-sm bg-bg/80 px-2 py-1">
                Re {num(hud.re, 0)} · CFL {num(hud.cfl, 2)} · t {num(hud.time, 2)}s
              </p>
            </div>
            <div className="pointer-events-none absolute right-3 bottom-3 left-3 flex items-end justify-between gap-2">
              <p className="min-w-0 truncate rounded-sm bg-bg/80 px-2 py-1 font-mono text-xs text-fg">{hud.hover}</p>
              <div className="shrink-0 rounded-sm bg-bg/80 px-2 py-1">
                <canvas ref={legendRef} className="h-3 w-32" />
                <p className="mt-1 flex justify-between font-mono text-xs text-muted">
                  <span>{view === "vorticity" || view === "pressure" ? "−" : "0"}</span>
                  <span className="text-fg">{view}</span>
                  <span>+</span>
                </p>
              </div>
            </div>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
            <Stat label="St" value={hud.strouhal > 0 ? num(hud.strouhal, 3) : "—"} />
            <Stat label="Cd" value={num(hud.cd, 2)} />
            <Stat label="Cl" value={num(hud.cl, 2)} />
            <Stat label="ms / step" value={num(hud.stepMs, 1)} />
            <Stat label="cells" value={hud.cells.toLocaleString("en-US")} />
            <Stat label="max |u|" value={num(hud.maxSpeed, 1)} />
            <Stat label="KE" value={hud.ke > 0 ? hud.ke.toExponential(2) : "—"} />
          </dl>
        </section>

        <aside className="min-w-0 border-border lg:col-span-1 lg:max-h-screen lg:overflow-y-auto lg:border-l">
          <div className="flex border-b border-border" role="tablist">
            {(
              [
                ["field", "Field"],
                ["rig", "Rig"],
                ["kernels", "Kernels"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={panel === id}
                className={`h-11 flex-1 text-sm ${panel === id ? "bg-surface text-primary" : "text-muted"}`}
                onClick={() => setPanel(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {panel === "field" ? (
            <div className="flex flex-col gap-4 p-4">
              <p className="text-pretty text-sm text-muted">{info.blurb}</p>
              <div className="flex flex-wrap gap-2">
                {PRESETS.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    aria-pressed={p.id === presetId}
                    className={`${chip} ${p.id === presetId ? chipOn : ""}`}
                    onClick={() => loadPreset(p.id, ny)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>

              <label className="flex flex-col gap-1 text-xs text-muted">
                Grid height
                <select
                  className={field}
                  value={ny}
                  onChange={(e) => {
                    const height = Number(e.target.value);
                    setNy(height);
                  }}
                >
                  {HEIGHTS.map((h) => (
                    <option key={h} value={h}>
                      {HEIGHT_NAME[h]} · {h} cells
                    </option>
                  ))}
                </select>
              </label>

              <div>
                <p className="mb-1 text-xs text-muted">Scalar</p>
                <div className="grid grid-cols-2 gap-2">
                  {(
                    [
                      ["dye", "Dye"],
                      ["vorticity", "Vorticity"],
                      ["speed", "Speed"],
                      ["pressure", "Pressure"],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      aria-pressed={view === id}
                      className={`${chip} ${view === id ? chipOn : ""}`}
                      onClick={() => setView(id)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <p className="mb-1 text-xs text-muted">Pointer</p>
                <div className="grid grid-cols-2 gap-2">
                  {(
                    [
                      ["stir", "Stir"],
                      ["wall", "Wall"],
                      ["erase", "Erase"],
                      ["probe", "Probe"],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      aria-pressed={tool === id}
                      className={`${chip} ${tool === id ? chipOn : ""}`}
                      onClick={() => setTool(id)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              <Slider label="Reynolds" min={30} max={1200} step={10} value={re} onChange={setRe} digits={0} />
              {driven ? (
                <Slider label="Inlet speed" min={0} max={80} step={1} value={uIn} onChange={setUIn} digits={0} />
              ) : null}
              <Slider
                label="Vorticity confinement"
                min={0}
                max={0.8}
                step={0.02}
                value={vorticity}
                onChange={setVorticity}
                digits={2}
              />
              <Slider label="Buoyancy" min={0} max={80} step={1} value={buoyancy} onChange={setBuoyancy} digits={0} />
              <Slider
                label="Pressure iterations"
                min={6}
                max={40}
                step={1}
                value={iterations}
                onChange={setIterations}
                digits={0}
              />

              <div className="flex gap-2">
                <button
                  type="button"
                  aria-pressed={particles}
                  className={`${chip} flex-1 ${particles ? chipOn : ""}`}
                  onClick={() => setParticles((v) => !v)}
                >
                  Tracers
                </button>
                <button
                  type="button"
                  aria-pressed={vectors}
                  className={`${chip} flex-1 ${vectors ? chipOn : ""}`}
                  onClick={() => setVectors((v) => !v)}
                >
                  Vectors
                </button>
              </div>

              <div>
                <div className="mb-1 flex justify-between text-xs text-muted">
                  <span>{traceName} trace</span>
                  <span className="font-mono text-fg">{traceName === "Cl" ? num(hud.cl, 2) : num(hud.maxAbsOmega, 2)}</span>
                </div>
                <canvas ref={sparkRef} className="h-12 w-full rounded-md border border-border bg-surface" />
              </div>
              <p className="font-mono text-xs text-muted">Space pause · R reset · . step</p>
            </div>
          ) : null}

          {panel === "rig" ? (
            <div className="flex flex-col gap-4 p-4 text-sm">
              <RigRow title={GPU.name} meta={`${GPU.chip} · ${GPU.arch} · cc ${GPU.cc}`} />
              <dl className="grid grid-cols-2 gap-2">
                <Stat label="SMs" value={String(GPU.sms)} />
                <Stat label="FP32" value={GPU.fp32.toLocaleString("en-US")} />
                <Stat label="VRAM" value={`${GPU.vramGb} GB`} />
                <Stat label="Bus" value={`${GPU.bandwidthGBps} GB/s`} />
                <Stat label="L2" value={`${GPU.l2Mb} MB`} />
                <Stat label="Boost" value={`${GPU.boostGhz} GHz`} />
              </dl>
              <RigRow title={CPU.name} meta={`${CPU.arch} · ${CPU.cores} cores / ${CPU.threads} threads · ${CPU.l3Mb} MB L3`} />
              <Stat label="Host memory" value={`${CPU.ramGb} GB`} />
              <div className="rounded-md border border-border bg-surface p-3 font-mono text-xs leading-5 text-fg">
                <p>
                  block {plan.block}×{plan.block}
                </p>
                <p>
                  grid {plan.gridX}×{plan.gridY}
                </p>
                <p>{plan.threads.toLocaleString("en-US")} threads launched</p>
                <p>{formatBytes(plan.bytes)} field storage</p>
                <p>Jacobi sweep ≈ {plan.dramMs.toFixed(4)} ms at DRAM peak</p>
                <p className="mt-2 text-muted">{plan.fitsL2 ? "Fits in L2" : "Exceeds L2"}</p>
              </div>
              <p className="text-pretty text-sm text-muted">{plan.note}</p>
              <p className="text-pretty text-sm text-muted">
                The live field runs in the browser on the CPU, which is the 5900X’s job: reductions, traces, and this reference. The card kernels are in the next tab. A browser cannot put this process on the 3090 Ti.
              </p>
            </div>
          ) : null}

          {panel === "kernels" ? (
            <div className="flex flex-col gap-3 p-4">
              <div className="flex flex-wrap gap-2">
                {KERNELS.map((k) => (
                  <button
                    key={k.id}
                    type="button"
                    aria-pressed={k.id === kernel.id}
                    className={`${chip} ${k.id === kernel.id ? chipOn : ""}`}
                    onClick={() => setKernelId(k.id)}
                  >
                    {k.label}
                  </button>
                ))}
              </div>
              <p className="text-pretty text-sm text-muted">{kernel.blurb}</p>
              <div className="flex gap-2">
                <button type="button" className={btn} onClick={() => void copyKernel()}>
                  <Copy className="size-4" />
                  {copied ? "Copied" : "Copy"}
                </button>
                <button type="button" className={btn} onClick={downloadKernel}>
                  <Download className="size-4" />
                  {kernel.filename}
                </button>
              </div>
              <pre className="max-h-96 overflow-auto rounded-md border border-border bg-surface p-3 font-mono text-xs leading-5 text-fg">
                {kernel.source}
              </pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-surface px-2 py-2">
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="font-mono text-sm text-fg tabular-nums">{value}</dd>
    </div>
  );
}

function RigRow({ title, meta }: { title: string; meta: string }) {
  return (
    <div>
      <h2 className="text-balance text-base font-semibold">{title}</h2>
      <p className="text-pretty text-xs text-muted">{meta}</p>
    </div>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  digits,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  digits: number;
  onChange: (n: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="flex justify-between text-xs text-muted">
        <span>{label}</span>
        <span className="font-mono text-fg tabular-nums">{value.toFixed(digits)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-11 w-full accent-primary"
      />
    </label>
  );
}

const btn =
  "inline-flex h-11 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-sm text-fg hover:border-primary";
const chip = "inline-flex h-11 items-center justify-center rounded-md border border-border bg-surface px-3 text-sm text-fg";
const chipOn = "border-primary text-primary";
const field = "h-11 rounded-md border border-border bg-surface px-2 text-sm text-fg";
