import { useEffect, useRef, useState } from "react";
import { Copy, Download, Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { CPU, GPU, cellBudget, formatBytes, planVolume } from "@/cfd/device";
import { KERNELS } from "@/cfd/kernels";
import { sectionFea, type FeaReport } from "@/cfd/fea";
import { PRESETS3, buildScene3, preset3Info, type BodyKind, type Preset3, type Splat3 } from "@/cfd/presets3d";
import { drawCurve, drawVolume, hitVolume, type Camera } from "@/cfd/render3d";
import { SKILLS, readVolume, solveBlackScholes, solveHeston, type SkillId } from "@/cfd/skills";
import { runExperimental } from "@/cfd/experimental";
import { Fluid3 } from "@/cfd/solver3d";
import type { SimParams, SimStats, ViewMode } from "@/cfd/types";
import { GUIDE, Guide, guideWasFinished, markGuideFinished } from "@/components/guide";

const SPANS = [16, 24, 32, 48, 64] as const;
const SPAN_NAME: Record<(typeof SPANS)[number], string> = {
  16: "Bench",
  24: "Fine",
  32: "Dense",
  48: "Large",
  64: "Max preview",
};

type Panel = "analysis" | "field" | "suite" | "skills" | "rig" | "kernels";
type AnalysisId =
  | "incompressible" | "heat" | "static"
  | "lbm" | "multipurpose" | "comfort" | "compressible" | "conjugate" | "multiphase"
  | "dynamic" | "heatSolid" | "thermomech" | "harmonic" | "frequency" | "nonlinear";
const ANALYSIS_LABEL: Record<AnalysisId, string> = {
  incompressible: "Incompressible",
  heat: "Convective heat transfer (preview)",
  static: "Static section (preview)",
  lbm: "Incompressible LBM (experimental)",
  multipurpose: "Multi-purpose (experimental)",
  comfort: "Pedestrian wind comfort (experimental)",
  compressible: "Compressible (experimental)",
  conjugate: "Conjugate heat transfer (experimental)",
  multiphase: "Multiphase (experimental)",
  dynamic: "Dynamic (experimental)",
  heatSolid: "Solid heat transfer (experimental)",
  thermomech: "Thermomechanical (experimental)",
  harmonic: "Harmonic (experimental)",
  frequency: "Frequency analysis (experimental)",
  nonlinear: "Nonlinear mechanical (experimental)",
};
type Tool = "orbit" | "stir" | "place" | "erase";
type Hud = SimStats & { stepMs: number; cells: string; hover: string };

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
  re: 180,
  cfl: 0,
  strouhal: 0,
  steps: 0,
  stepMs: 0,
  cells: "—",
  hover: "drag to orbit the volume",
};

function num(n: number, d = 2): string {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

export function Bench() {
  const [presetId, setPresetId] = useState<Preset3>("karman");
  const [span, setSpan] = useState<(typeof SPANS)[number]>(16);
  const [running, setRunning] = useState(true);
  const [view, setView] = useState<ViewMode>("dye");
  const [tool, setTool] = useState<Tool>("orbit");
  const [particles, setParticles] = useState(true);
  const [panel, setPanel] = useState<Panel>("analysis");
  const [analysis, setAnalysis] = useState<AnalysisId>("incompressible");
  const [kernelId, setKernelId] = useState(KERNELS[0]!.id);
  const [copied, setCopied] = useState(false);
  const [re, setRe] = useState(180);
  const [uIn, setUIn] = useState(1);
  const [vorticity, setVorticity] = useState(0.16);
  const [buoyancy, setBuoyancy] = useState(0);
  const [iterations, setIterations] = useState(8);
  const [solid, setSolid] = useState<BodyKind>("sphere");
  const [solidSize, setSolidSize] = useState(2.4);
  const [gradP, setGradP] = useState(0);
  const [tIn, setTIn] = useState(20);
  const [tWall, setTWall] = useState(20);
  const [tAmb, setTAmb] = useState(20);
  const [kappa, setKappa] = useState(0.08);
  const [young, setYoung] = useState(2);
  const [fea, setFea] = useState<FeaReport | null>(null);
  const [drop, setDrop] = useState(0);
  const [skillId, setSkillId] = useState<SkillId>("topology");
  const [sigma, setSigma] = useState(0.2);
  const [xi, setXi] = useState(0.5);
  const [hud, setHud] = useState<Hud>(ZERO);
  const [reading, setReading] = useState("");
  const [dist, setDist] = useState(43);
  const [guideOpen, setGuideOpen] = useState(false);
  const [guideStep, setGuideStep] = useState(0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sparkRef = useRef<HTMLCanvasElement>(null);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const feaRef = useRef<HTMLCanvasElement>(null);
  const simRef = useRef<Fluid3 | null>(null);
  const splats = useRef<Splat3[]>([]);
  const down = useRef(false);
  const lastPtr = useRef({ x: 0, y: 0, w: 1, h: 1 });
  const stepOnce = useRef(false);
  const decayRef = useRef(0.12);
  const spark = useRef(new Float32Array(160));
  const sparkN = useRef(0);
  const sparkAt = useRef(0);
  const resetRef = useRef<() => void>(() => {});
  const cam = useRef<Camera>({ yaw: 0.7, pitch: 0.38, dist: 48 });
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef(0);

  const live = useRef({
    view,
    particles,
    running,
    tool,
    re,
    uIn,
    vorticity,
    buoyancy,
    iterations,
    gradP,
    tIn,
    tWall,
    tAmb,
    kappa,
    young,
    panel,
    solid,
    solidSize,
    skillId,
    presetId,
  });
  live.current = {
    view,
    particles,
    running,
    tool,
    re,
    uIn,
    vorticity,
    buoyancy,
    iterations,
    gradP,
    tIn,
    tWall,
    tAmb,
    kappa,
    young,
    panel,
    solid,
    solidSize,
    skillId,
    presetId,
  };

  const info = preset3Info(presetId);
  const kernel = KERNELS.find((k) => k.id === kernelId) ?? KERNELS[0]!;
  const skill = SKILLS.find((s) => s.id === skillId) ?? SKILLS[0]!;

  function applyScene(id: Preset3, nextSpan: number) {
    const scene = buildScene3(id, nextSpan);
    setPresetId(id);
    setView(scene.view);
    setRe(scene.re);
    setUIn(scene.uIn);
    setVorticity(scene.vorticity);
    setBuoyancy(scene.buoyancy);
    decayRef.current = scene.dyeDecay;
    cam.current.dist = scene.nx * 1.35;
    setDist(scene.nx * 1.35);
    return scene;
  }

  resetRef.current = () => {
    const scene = buildScene3(presetId, span);
    decayRef.current = scene.dyeDecay;
    simRef.current?.load(scene);
    sparkN.current = 0;
    sparkAt.current = 0;
  };

  useEffect(() => {
    if (!guideWasFinished()) setGuideOpen(true);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = buildScene3(presetId, span);
    decayRef.current = scene.dyeDecay;
    cam.current.dist = scene.nx * 1.35;
    setDist(scene.nx * 1.35);
    const sim = new Fluid3();
    const params = (): SimParams => ({
      re: live.current.re,
      uIn: live.current.uIn,
      vorticity: live.current.vorticity,
      buoyancy: live.current.buoyancy,
      iterations: live.current.iterations,
      dyeDecay: decayRef.current,
      gradP: live.current.gradP,
      tIn: live.current.tIn,
      tWall: live.current.tWall,
      tAmb: live.current.tAmb,
      kappa: live.current.kappa,
    });
    sim.load(scene);
    let lastSt: SimStats | null = null;
    const warmStart = performance.now();
    let warmed = 0;
    while (warmed < 1.2 && performance.now() - warmStart < 450) {
      const h = sim.cflDt(0.7);
      lastSt = sim.step(h, params(), []);
      warmed += lastSt.dt;
    }
    simRef.current = sim;
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
      const sparkCanvas = sparkRef.current;
      if (sparkCanvas) {
        const sr = sparkCanvas.getBoundingClientRect();
        sparkCanvas.width = Math.max(2, Math.round(sr.width * dpr));
        sparkCanvas.height = Math.max(2, Math.round(sr.height * dpr));
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
      acc += live.current.running ? wall : 0;
      if (stepOnce.current) {
        acc = Math.max(acc, sim.cflDt(0.7));
        stepOnce.current = false;
      }
      let guard = 0;
      const tStep = performance.now();
      while (acc > 1e-4 && guard < 2) {
        const h = Math.min(sim.cflDt(0.7), acc);
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
      if (guard === 2) acc = 0;
      if (guard > 0) stepMs = (performance.now() - tStep) / guard;
      drawVolume(canvas, sim, live.current.view, cam.current, live.current.particles);
      const sparkCanvas = sparkRef.current;
      if (sparkCanvas) paintSpark(sparkCanvas, spark.current, sparkN.current, sparkAt.current);
      if (now - hudAt > 200 && lastSt) {
        hudAt = now;
        if (live.current.panel === "suite") {
          setDrop(sim.pressureDrop());
          setFea(sectionFea(sim, live.current.young, 0.3));
        }
        if (live.current.skillId !== "blackscholes" && live.current.skillId !== "heston") {
          setReading(readVolume(sim, { re: live.current.re, preset: live.current.presetId }, live.current.skillId).note);
        }
        const toolName =
          live.current.tool === "orbit"
            ? "drag to orbit"
            : live.current.tool === "place"
              ? "click to place the solid"
              : live.current.tool === "erase"
                ? "drag to erase solid"
                : "drag to stir the mid-span plane";
        setHud({
          ...lastSt,
          stepMs,
          cells: `${sim.nx}×${sim.ny}×${sim.nz}`,
          hover: toolName,
          re: live.current.re,
        });
      }
      raf = requestAnimationFrame(frame);
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.1 : 0.9;
      const next = Math.min(sim.nx * 3.2, Math.max(sim.nx * 0.22, cam.current.dist * factor));
      cam.current.dist = next;
      setDist(next);
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    raf = requestAnimationFrame(frame);
    ro.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      canvas.removeEventListener("wheel", onWheel);
      ro.disconnect();
      if (simRef.current === sim) simRef.current = null;
    };
  }, [presetId, span]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space") {
        e.preventDefault();
        setRunning((v) => !v);
      } else if (e.key === "r" || e.key === "R") resetRef.current();
      else if (e.key === ".") stepOnce.current = true;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const canvas = chartRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(2, Math.round(rect.width * dpr));
    canvas.height = Math.max(2, Math.round(rect.height * dpr));
    if (skillId === "blackscholes") {
      const solved = solveBlackScholes(sigma);
      drawCurve(canvas, solved.curveN, solved.curveA);
      setReading(
        `Grid call ${solved.numerical.toFixed(2)} against closed form ${solved.analytic.toFixed(2)}. Error ${solved.error.toFixed(2)}. Same parabolic step as the dye.`,
      );
    } else if (skillId === "heston") {
      const solved = solveHeston(xi);
      drawCurve(canvas, solved.variance, null);
      setReading(
        `Heston call ${solved.heston.toFixed(2)}, Black–Scholes at √θ ${solved.blackScholes.toFixed(2)}, gap ${solved.gap.toFixed(2)}. ξ is volatility of volatility.`,
      );
    }
  }, [skillId, sigma, xi, panel]);

  useEffect(() => {
    const canvas = feaRef.current;
    const sim = simRef.current;
    if (!canvas || panel !== "suite") return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(2, Math.round(rect.width * dpr));
    canvas.height = Math.max(2, Math.round(rect.height * dpr));
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#101412";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!sim || !fea || fea.outline.length === 0) return;
    ctx.fillStyle = "#e39b2b";
    for (const q of fea.outline) {
      const x = (q.x / sim.nx) * canvas.width;
      const y = canvas.height - (q.y / sim.ny) * canvas.height;
      ctx.fillRect(x, y, 3, 3);
    }
  }, [fea, panel]);

  function pointer(e: React.PointerEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top, w: rect.width, h: rect.height };
  }

  function setZoom(next: number) {
    const nx = simRef.current?.nx ?? span * 2;
    const clamped = Math.min(nx * 3.2, Math.max(nx * 0.22, next));
    cam.current.dist = clamped;
    setDist(clamped);
  }

  function pinchDistance(): number {
    const pts = [...pointers.current.values()];
    if (pts.length < 2) return 0;
    return Math.hypot(pts[0]!.x - pts[1]!.x, pts[0]!.y - pts[1]!.y);
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size >= 2) {
      pinch.current = pinchDistance();
      down.current = false;
      return;
    }
    down.current = true;
    const p = pointer(e);
    lastPtr.current = p;
    if (live.current.tool === "place" || live.current.tool === "erase") sculpt(p);
    else if (live.current.tool === "stir") stir(p, p);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.current.size >= 2) {
      const d = pinchDistance();
      if (pinch.current > 8 && d > 8) {
        setZoom(cam.current.dist * (pinch.current / d));
        pinch.current = d;
      }
      return;
    }
    if (!down.current) return;
    const p = pointer(e);
    if (live.current.tool === "orbit") {
      cam.current.yaw += (p.x - lastPtr.current.x) * 0.01;
      cam.current.pitch = Math.max(-1.1, Math.min(1.1, cam.current.pitch + (p.y - lastPtr.current.y) * 0.008));
    } else if (live.current.tool === "erase") sculpt(p);
    else if (live.current.tool === "stir") stir(lastPtr.current, p);
    lastPtr.current = p;
  }

  function sculpt(p: { x: number; y: number; w: number; h: number }) {
    const sim = simRef.current;
    if (!sim) return;
    const hit = hitVolume(sim, cam.current, p.x, p.y, p.w, p.h);
    if (!hit) return;
    if (live.current.tool === "erase") sim.eraseBody(hit.x, hit.y, hit.z, live.current.solidSize);
    else sim.reshape(live.current.solid, live.current.solidSize, hit);
  }

  function stir(prev: { x: number; y: number; w: number; h: number }, next: { x: number; y: number; w: number; h: number }) {
    const sim = simRef.current;
    if (!sim) return;
    const hit = hitVolume(sim, cam.current, next.x, next.y, next.w, next.h);
    if (!hit) return;
    const scale = sim.nx / next.w;
    splats.current.push({
      x: hit.x,
      y: hit.y,
      z: hit.z,
      du: (next.x - prev.x) * scale * 0.35,
      dv: -(next.y - prev.y) * scale * 0.35,
      dw: 0,
      dye: 0.7,
      radius: Math.max(2, sim.ny * 0.12),
    });
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    pointers.current.delete(e.pointerId);
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

  const plan = planVolume(span * 2, span, span);
  const budget = cellBudget(span * 2, span, span);
  const exp = runExperimental(analysis, {
    re,
    uIn,
    maxSpeed: hud.maxSpeed,
    tIn,
    tWall,
    young,
    vonMises: fea?.maxVonMises ?? 0,
    meanPressure: fea?.meanPressure ?? 0,
  });
  const priced = skillId === "blackscholes" || skillId === "heston";
  const fluidSkill = !priced && simRef.current ? readVolume(simRef.current, { re, preset: presetId }, skillId) : null;

  return (
    <div className="min-h-screen bg-bg text-fg">
      <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-border bg-bg px-4 py-3">
        <div className="min-w-40">
          <p className="font-mono text-xs tracking-widest text-primary">MAC reference headless · canvas is the preview · GPU unverified</p>
          <h1 className="text-balance text-xl font-semibold tracking-tight">Karman</h1>
        </div>
        <p className="hidden max-w-xl text-pretty text-sm text-muted md:block">
          {ANALYSIS_LABEL[analysis]} · CPU preview · checked MAC runs headless · GPU unverified
        </p>
        <div className="ml-auto flex flex-wrap gap-2">
          <button
            type="button"
            className={btn}
            aria-expanded={guideOpen}
            aria-controls="guide-title"
            onClick={() => {
              setGuideStep(0);
              setGuideOpen(true);
              setPanel("analysis");
            }}
          >
            Guide
          </button>
          <button type="button" className={btn} onClick={() => setRunning((v) => !v)}>
            {running ? <Pause className="size-4" /> : <Play className="size-4" />}
            {running ? "Pause" : "Run"}
          </button>
          <button type="button" className={btn} onClick={() => { stepOnce.current = true; }}>
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
              className="block aspect-[4/3] w-full touch-none bg-bg"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            />
            <div className="absolute top-3 right-3 flex flex-col gap-2">
              <button type="button" className={zoomBtn} aria-label="Zoom in" onClick={() => setZoom(dist * 0.8)}>
                +
              </button>
              <button type="button" className={zoomBtn} aria-label="Zoom out" onClick={() => setZoom(dist * 1.25)}>
                −
              </button>
            </div>
            <div className="pointer-events-none absolute top-3 left-3 font-mono text-xs leading-5 text-fg">
              <p className="rounded-sm bg-bg/80 px-2 py-1">
                {info.name} · Re {num(hud.re, 0)} · CFL {num(hud.cfl, 2)} · t {num(hud.time, 2)}s
              </p>
            </div>
            <p className="pointer-events-none absolute right-3 bottom-3 max-w-[70%] truncate rounded-sm bg-bg/80 px-2 py-1 font-mono text-xs text-fg">
              {hud.hover}
            </p>
          </div>
          <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
            <Stat label="St (preview)" value={hud.strouhal > 0 ? num(hud.strouhal, 3) : "—"} />
            <Stat label="Cd (preview)" value={num(hud.cd, 2)} />
            <Stat label="Cl (preview)" value={num(hud.cl, 2)} />
            <Stat label="ms / step" value={num(hud.stepMs, 1)} />
            <Stat label="grid" value={hud.cells} />
            <Stat label="max |u|" value={num(hud.maxSpeed, 2)} />
            <Stat label="KE" value={hud.ke > 0 ? hud.ke.toExponential(2) : "—"} />
          </dl>
          {guideOpen ? (
            <Guide
              step={guideStep}
              onStep={(next) => {
                setGuideStep(next);
                const panelId = GUIDE[next]?.panel;
                if (panelId) setPanel(panelId);
              }}
              onClose={() => {
                setGuideOpen(false);
                markGuideFinished();
              }}
            />
          ) : null}
        </section>

        <aside className="min-w-0 border-border lg:col-span-1 lg:max-h-screen lg:overflow-y-auto lg:border-l">
          <div className="flex border-b border-border" role="tablist">
            {(
              [
                ["analysis", "Analysis"],
                ["field", "Field"],
                ["suite", "Suite"],
                ["skills", "Skills"],
                ["rig", "Rig"],
                ["kernels", "Kernels"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                role="tab"
                id={`tab-${id}`}
                aria-controls={`panel-${id}`}
                aria-selected={panel === id}
                tabIndex={panel === id ? 0 : -1}
                className={`h-11 min-w-0 flex-1 px-1 text-xs focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${panel === id ? "bg-surface text-primary" : "text-muted"}`}
                onClick={() => setPanel(id)}
              >
                {label}
              </button>
            ))}
          </div>


          {panel === "analysis" ? (
            <div id="panel-analysis" role="tabpanel" aria-labelledby="tab-analysis" className="flex flex-col gap-4 p-4">
              <p id="analysis-help" className="text-pretty text-sm text-muted">
                Envelope: constant-density Newtonian laminar flow on a Cartesian grid. The checked scheme is the staggered MAC projection, run headless. Passive scalar and Boussinesq heat are checked only on the cases named in verify/references.test.mjs. The canvas is still the collocated preview. Screens are not solvers. Electromagnetics and a CAD kernel are outside this program.
              </p>
              <p className="font-mono text-xs text-primary" aria-live="polite">Selected: {ANALYSIS_LABEL[analysis]}</p>
              <div role="status" aria-live="polite" aria-atomic="true" className="rounded-md border border-border bg-surface p-3">
                <p className="text-xs text-muted">{exp.title}</p>
                <p className="mt-1 font-mono text-sm text-primary tabular-nums">{exp.value}</p>
                <p className="mt-1 text-pretty text-sm text-muted">{exp.detail}</p>
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-xs font-semibold tracking-wide text-muted">FLOW</p>
                <AnalysisRow name="Incompressible" status="available" selected={analysis === "incompressible"} onSelect={() => { setAnalysis("incompressible"); setBuoyancy(0); setPanel("field"); }} note="Checked path. Constant-density Newtonian laminar flow on a structured grid." />
                <AnalysisRow name="Incompressible (LBM)" status="experimental" selected={analysis === "lbm"} onSelect={() => setAnalysis("lbm")} note="Runs a periodic D2Q9 step and checks density and momentum. Not coupled to this volume." />
                <AnalysisRow name="Multi-purpose" status="experimental" selected={analysis === "multipurpose"} onSelect={() => setAnalysis("multipurpose")} note="Exposes the same incompressible fields with heat and section controls available. Not a multiphysics coupling." />
                <AnalysisRow name="Pedestrian wind comfort" status="experimental" selected={analysis === "comfort"} onSelect={() => setAnalysis("comfort")} note="Classifies the preview max speed at 4, 6, and 8 m/s. No exceedance probability." />
                <AnalysisRow name="Compressible" status="experimental" selected={analysis === "compressible"} onSelect={() => setAnalysis("compressible")} note="Isentropic density ratio and a Mach 0.3 screen on the constant-density assumption. Not a compressible solver." />
                <AnalysisRow name="Convective heat transfer" status="preview" selected={analysis === "heat"} onSelect={() => { setAnalysis("heat"); setBuoyancy(1.2); setPanel("suite"); }} note="Turns on Boussinesq buoyancy and opens the Suite temperature controls. Not a verified heat-transfer study." />
                <AnalysisRow name="Conjugate heat transfer" status="experimental" selected={analysis === "conjugate"} onSelect={() => setAnalysis("conjugate")} note="One-dimensional series resistance. Fluid and solid fluxes match by construction. Not a field solve." />
                <AnalysisRow name="Multiphase" status="experimental" selected={analysis === "multiphase"} onSelect={() => setAnalysis("multiphase")} note="No phase field is stored. Dye stays a passive scalar." />
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-xs font-semibold tracking-wide text-muted">STRUCTURAL</p>
                <AnalysisRow name="Static" status="preview" selected={analysis === "static"} onSelect={() => { setAnalysis("static"); setPanel("suite"); }} note="One-way plane-stress section. Experimental, not FSI." />
                <AnalysisRow name="Dynamic" status="experimental" selected={analysis === "dynamic"} onSelect={() => setAnalysis("dynamic")} note="Estimates a displacement period from the section stiffness and a unit mass. Not a dynamic solver." />
                <AnalysisRow name="Heat transfer" status="experimental" selected={analysis === "heatSolid"} onSelect={() => setAnalysis("heatSolid")} note="Estimates solid conduction across the section thickness. Not a conduction solver." />
                <AnalysisRow name="Thermomechanical" status="experimental" selected={analysis === "thermomech"} onSelect={() => setAnalysis("thermomech")} note="Adds a thermal strain estimate to the section stress. Not thermomechanical coupling." />
                <AnalysisRow name="Harmonic" status="experimental" selected={analysis === "harmonic"} onSelect={() => setAnalysis("harmonic")} note="Reports a forced-response estimate at 1 Hz. Not a harmonic solver." />
                <AnalysisRow name="Frequency analysis" status="experimental" selected={analysis === "frequency"} onSelect={() => setAnalysis("frequency")} note="Estimates a first natural frequency from the section. Not a modal analysis." />
                <AnalysisRow name="Nonlinear mechanical" status="experimental" selected={analysis === "nonlinear"} onSelect={() => setAnalysis("nonlinear")} note="Compares section strain σ/E with 1%. Not a nonlinear solver." />
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-xs font-semibold tracking-wide text-muted">ELECTROMAGNETIC</p>
                <AnalysisRow name="Electromagnetics" status="planned" note="Not in this project." />
              </div>
            </div>
          ) : null}

          {panel === "field" ? (
            <div className="flex flex-col gap-4 p-4">
              <p className="text-pretty text-sm text-muted">{info.blurb}</p>
              <p className="text-pretty text-xs text-muted">
                This picture is the CPU preview: collocated semi-Lagrangian, fixed pressure iterations. It is not a verified projection. The checked scheme is the staggered MAC solver in double precision, run headless. GPU execution of kernels/karman3d.cu is unverified in this environment.
              </p>
              <div className="flex flex-wrap gap-2">
                {PRESETS3.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    aria-pressed={p.id === presetId}
                    className={`${chip} ${p.id === presetId ? chipOn : ""}`}
                    onClick={() => applyScene(p.id, span)}
                  >
                    {p.name}
                  </button>
                ))}
              </div>
              <label className="flex flex-col gap-1 text-xs text-muted">
                Grid size
                <select className={field} value={span} onChange={(e) => setSpan(Number(e.target.value) as (typeof SPANS)[number])} aria-describedby="cell-budget">
                  {SPANS.map((h) => {
                    const cells = h * 2 * h * h;
                    return (
                      <option key={h} value={h}>
                        {SPAN_NAME[h]} · {h * 2}×{h}×{h} · {cells.toLocaleString("en-US")} cells
                      </option>
                    );
                  })}
                </select>
              </label>
              <div id="cell-budget" className="rounded-md border border-border bg-surface p-3" role="status">
                <p className="text-xs text-muted">Hardware boundary · 3090 Ti 24 GB · 5900X 64 GB</p>
                <p className="mt-1 font-mono text-sm text-primary tabular-nums">{budget.cells.toLocaleString("en-US")} cells · preview {formatBytes(budget.previewBytes)} · reference ~{formatBytes(budget.referenceBytes)}</p>
                <ul className="mt-2 space-y-1 text-xs text-muted">
                  <li>{budget.fitsL2 ? "Fits in 6 MB L2" : "Larger than 6 MB L2"}</li>
                  <li>{budget.previewOk ? "Inside browser preview comfort (~200k cells)" : "Above browser preview comfort — expect slower steps"}</li>
                  <li>{budget.withinReference ? "Within the 2,000,000-cell reference cap" : "Above the reference cap — the checked solver will refuse this grid"}</li>
                  <li>{budget.fitsVram ? "Working-set estimate fits in 70% of 24 GB VRAM" : "Exceeds 70% of 24 GB VRAM"}</li>
                </ul>
                <p className="mt-2 text-pretty text-sm text-muted">{budget.note}</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {(
                  [
                    ["dye", "Dye"],
                    ["temperature", "Temp"],
                    ["vorticity", "Vorticity"],
                    ["speed", "Speed"],
                    ["pressure", "Pressure"],
                  ] as const
                ).map(([id, label]) => (
                  <button key={id} type="button" aria-pressed={view === id} className={`${chip} ${view === id ? chipOn : ""}`} onClick={() => setView(id)}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <button type="button" aria-pressed={tool === "orbit"} className={`${chip} ${tool === "orbit" ? chipOn : ""}`} onClick={() => setTool("orbit")}>
                  Orbit
                </button>
                <button type="button" aria-pressed={tool === "stir"} className={`${chip} ${tool === "stir" ? chipOn : ""}`} onClick={() => setTool("stir")}>
                  Stir
                </button>
              </div>
              <button type="button" aria-pressed={particles} className={`${chip} ${particles ? chipOn : ""}`} onClick={() => setParticles((v) => !v)}>
                Tracers
              </button>
              <Slider
                label="Zoom"
                min={Math.round(span * 2 * 0.22)}
                max={Math.round(span * 2 * 3.2)}
                step={1}
                value={Math.round(dist)}
                digits={0}
                onChange={setZoom}
              />
              <Slider label="Reynolds" min={20} max={800} step={10} value={re} digits={0} onChange={setRe} />
              <Slider label="Inlet" min={0} max={2} step={0.05} value={uIn} digits={2} onChange={setUIn} />
              <Slider label="Confinement" min={0} max={0.6} step={0.02} value={vorticity} digits={2} onChange={setVorticity} />
              <Slider label="Buoyancy" min={0} max={4} step={0.1} value={buoyancy} digits={1} onChange={setBuoyancy} />
              <Slider label="Pressure iterations" min={4} max={20} step={1} value={iterations} digits={0} onChange={setIterations} />
              <div>
                <div className="mb-1 flex justify-between text-xs text-muted">
                  <span>Cl (preview) trace</span>
                  <span className="font-mono text-fg">{num(hud.cl, 2)}</span>
                </div>
                <canvas ref={sparkRef} className="h-12 w-full rounded-md border border-border bg-surface" />
              </div>
              <p className="font-mono text-xs text-muted">Space pause · R reset · . step</p>
            </div>
          ) : null}

          {panel === "suite" ? (
            <div className="flex flex-col gap-4 p-4">
              <p className="text-pretty text-sm text-muted">
                Pick a shape and it replaces the solid. Drag the picture to orbit it. Move, then click, to reposition it.
              </p>
              <div className="flex flex-wrap gap-2">
                {(
                  [
                    ["sphere", "Sphere"],
                    ["cylinder", "Cylinder"],
                    ["box", "Box"],
                    ["cone", "Cone"],
                    ["plate", "Plate"],
                    ["airfoil", "Airfoil"],
                  ] as const
                ).map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    aria-pressed={solid === id}
                    className={`${chip} ${solid === id ? chipOn : ""}`}
                    onClick={() => {
                      setSolid(id);
                      setTool("orbit");
                      simRef.current?.reshape(id, solidSize);
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="grid grid-cols-3 gap-2">
                <button type="button" aria-pressed={tool === "orbit"} className={`${chip} ${tool === "orbit" ? chipOn : ""}`} onClick={() => setTool("orbit")}>
                  Orbit
                </button>
                <button type="button" aria-pressed={tool === "place"} className={`${chip} ${tool === "place" ? chipOn : ""}`} onClick={() => setTool("place")}>
                  Move
                </button>
                <button type="button" aria-pressed={tool === "erase"} className={`${chip} ${tool === "erase" ? chipOn : ""}`} onClick={() => setTool("erase")}>
                  Erase
                </button>
              </div>
              <button
                type="button"
                className={chip}
                onClick={() => {
                  simRef.current?.clearBodies();
                  setFea(null);
                }}
              >
                Clear solids
              </button>
              <Slider
                label="Solid size"
                min={1.2}
                max={6}
                step={0.1}
                value={solidSize}
                digits={1}
                onChange={(n) => {
                  setSolidSize(n);
                  if (simRef.current?.hasBody) simRef.current.reshape(solid, n);
                }}
              />
              <div className="rounded-md border border-border bg-surface p-3">
                <p className="text-xs text-muted">Geometry</p>
                <p className="mt-1 text-pretty text-sm text-muted">
                  Solids are voxel stamps on the Cartesian grid. A separate CAD kernel can be integrated later for edge repair, redundant faces, and small-entity removal. Use Clear solids to remove every stamp.
                </p>
              </div>

              <Slider label="Inlet temperature" min={0} max={120} step={1} value={tIn} digits={0} onChange={setTIn} />
              <Slider label="Solid temperature" min={0} max={120} step={1} value={tWall} digits={0} onChange={setTWall} />
              <Slider label="Ambient temperature" min={0} max={80} step={1} value={tAmb} digits={0} onChange={setTAmb} />
              <Slider label="Thermal diffusion" min={0} max={0.35} step={0.01} value={kappa} digits={2} onChange={setKappa} />
              <Slider label="Streamwise pressure gradient" min={-1.5} max={1.5} step={0.05} value={gradP} digits={2} onChange={setGradP} />
              <div className="rounded-md border border-border bg-surface p-3">
                <p className="text-xs text-muted">Measured inlet-to-outlet pressure drop</p>
                <p className="mt-1 font-mono text-sm text-primary tabular-nums">{num(drop, 3)}</p>
                <p className="mt-2 text-pretty text-sm text-muted">Positive gradient pushes the fluid downstream, on top of whatever the inlet is doing.</p>
              </div>
              <Slider label="Young’s modulus" min={0.2} max={20} step={0.1} value={young} digits={1} onChange={setYoung} />
              <div className="rounded-md border border-border bg-surface p-3">
                <p className="text-xs text-muted">Patch-tested plane stress, one way, not FSI. ν = 0.30. The patch covers the constitutive law, not this mesh.</p>
                <p className="mt-1 font-mono text-sm text-primary tabular-nums">
                  σ {fea ? fea.maxVonMises.toExponential(2) : "—"} · |u| {fea ? fea.maxDisp.toExponential(2) : "—"}
                </p>
                <p className="mt-1 font-mono text-xs text-muted">
                  {fea ? `${fea.nodes} nodes · ${fea.triangles} triangles · p̄ ${num(fea.meanPressure, 2)}` : "no section yet"}
                </p>
                <canvas ref={feaRef} className="mt-2 h-28 w-full rounded-md border border-border bg-bg" />
                <p className="mt-2 text-pretty text-sm text-muted">{fea?.note}</p>
              </div>
            </div>
          ) : null}

          {panel === "skills" ? (
            <div className="flex flex-col gap-4 p-4">
              <div className="flex flex-wrap gap-2">
                {SKILLS.map((s) => (
                  <button key={s.id} type="button" aria-pressed={s.id === skillId} className={`${chip} ${s.id === skillId ? chipOn : ""}`} onClick={() => setSkillId(s.id)}>
                    {s.name}
                  </button>
                ))}
              </div>
              <div>
                <p className="font-mono text-xs tracking-wide text-primary">{skill.subject}</p>
                <h2 className="mt-1 text-balance text-base font-semibold">{skill.name}</h2>
                <p className="mt-2 text-pretty text-sm text-muted">{skill.claim}</p>
                <p className="mt-2 text-pretty text-sm text-fg">{skill.machine}</p>
              </div>
              {priced ? (
                <div className="flex flex-col gap-3">
                  {skillId === "blackscholes" ? (
                    <Slider label="Volatility σ" min={0.05} max={0.8} step={0.01} value={sigma} digits={2} onChange={setSigma} />
                  ) : (
                    <Slider label="Vol of vol ξ" min={0} max={1.4} step={0.02} value={xi} digits={2} onChange={setXi} />
                  )}
                  <canvas ref={chartRef} className="h-28 w-full rounded-md border border-border bg-surface" />
                  <p className="text-pretty text-sm text-muted">{reading}</p>
                </div>
              ) : (
                <div className="rounded-md border border-border bg-surface p-3">
                  <p className="text-xs text-muted">{fluidSkill?.meter ?? "live meter"}</p>
                  <p className="mt-1 font-mono text-sm text-primary tabular-nums">{fluidSkill?.value ?? "—"}</p>
                  <p className="mt-2 text-pretty text-sm text-muted">{reading || fluidSkill?.note}</p>
                </div>
              )}
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
              </dl>
              <RigRow title={CPU.name} meta={`${CPU.arch} · ${CPU.cores} cores / ${CPU.threads} threads`} />
              <div className="rounded-md border border-border bg-surface p-3 font-mono text-xs leading-5 text-fg">
                <p>block 8×4×8</p>
                <p>
                  grid {plan.gridX}×{plan.gridY}×{plan.gridZ}
                </p>
                <p>{formatBytes(plan.bytes)} if this volume is resident</p>
                <p className="mt-2 text-muted">{plan.fitsL2 ? "Fits in the 6 MB L2" : "Larger than L2"}</p>
              </div>
              <p className="text-pretty text-sm text-muted">{plan.note}</p>
              <p className="text-pretty text-sm text-muted">
                This page runs the volume on the CPU so you can see it. The 3090 Ti is the target card, not the device running this page. Device execution of karman3d.cu is unverified. On that machine: one file, nvcc, sm_86, no library between you and the SMs.
              </p>
            </div>
          ) : null}

          {panel === "kernels" ? (
            <div className="flex flex-col gap-3 p-4">
              <div className="flex flex-wrap gap-2">
                {KERNELS.map((k) => (
                  <button key={k.id} type="button" aria-pressed={k.id === kernel.id} className={`${chip} ${k.id === kernel.id ? chipOn : ""}`} onClick={() => setKernelId(k.id)}>
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
              <pre className="max-h-96 overflow-auto rounded-md border border-border bg-surface p-3 font-mono text-xs leading-5 text-fg">{kernel.source}</pre>
            </div>
          ) : null}
        </aside>
      </div>
    </div>
  );
}

function paintSpark(canvas: HTMLCanvasElement, data: Float32Array, n: number, cursor: number): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || n < 2) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1a211e";
  ctx.fillRect(0, 0, w, h);
  let lo = Infinity;
  let hi = -Infinity;
  const values: number[] = [];
  for (let i = 0; i < n; i++) {
    const v = data[(cursor - n + i + data.length) % data.length]!;
    values.push(v);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (hi - lo < 1e-6) {
    lo -= 1;
    hi += 1;
  }
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const x = (i / (values.length - 1)) * (w - 4) + 2;
    const y = h - 3 - ((values[i]! - lo) / (hi - lo)) * (h - 6);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#e39b2b";
  ctx.lineWidth = 1.25;
  ctx.stroke();
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
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="h-11 w-full accent-primary" />
    </label>
  );
}

const btn = "inline-flex h-11 items-center justify-center gap-2 rounded-md border border-border bg-surface px-3 text-sm text-fg hover:border-primary";
const zoomBtn = "inline-flex h-11 w-11 items-center justify-center rounded-md border border-border bg-bg/80 text-lg text-fg hover:border-primary";
const chip = "inline-flex h-11 items-center justify-center rounded-md border border-border bg-surface px-3 text-sm text-fg";
const chipOn = "border-primary text-primary";


function AnalysisRow({ name, status, note, selected, onSelect }: { name: string; status: "available" | "preview" | "experimental" | "planned"; note: string; selected?: boolean; onSelect?: () => void }) {
  const tone =
    status === "available" ? "bg-primary/15 text-primary" :
    status === "preview" ? "bg-primary/10 text-primary" :
    status === "experimental" ? "bg-primary/10 text-primary" :
    "bg-muted/20 text-muted";
  const label = status === "available" ? "Available" : status === "preview" ? "Preview" : status === "experimental" ? "Experimental" : "Planned";
  const canSelect = status !== "planned" && onSelect;
  return (
    <button
      type="button"
      disabled={!canSelect}
      aria-pressed={selected}
      aria-label={`${name}, ${label}${selected ? ", selected" : ""}`}
      onClick={onSelect}
      className={`rounded-md border p-3 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary ${selected ? "border-primary" : "border-border"} ${canSelect ? "bg-surface hover:border-primary" : "bg-surface/60 cursor-not-allowed"}`}
    >
      <div className="flex items-center justify-between gap-2">
        <p className="font-medium text-sm">{name}</p>
        <span className={`rounded-full px-2 py-0.5 text-xs ${tone}`}>{label}</span>
      </div>
      <p className="mt-1 text-pretty text-sm text-muted">{note}</p>
    </button>
  );
}

const field = "h-11 rounded-md border border-border bg-surface px-2 text-sm text-fg";
