import { useEffect, useRef } from "react";

export type GuidePanel = "analysis" | "field" | "suite" | "skills" | "rig" | "kernels";

export type GuideStep = {
  title: string;
  body: string;
  points: readonly string[];
  panel?: GuidePanel;
};

export const GUIDE: readonly GuideStep[] = [
  {
    title: "Two calculations, one screen",
    body: "The checked envelope is constant-density Newtonian laminar flow on a Cartesian grid. The picture and the checked solver are not the same method. Treat the picture as a fast look at the case. Treat the headless MAC run as the number you can defend.",
    points: [
      "The canvas is a collocated CPU preview: semi-Lagrangian advection, a clipped backtrace, and a fixed number of pressure iterations.",
      "The checked scheme is a staggered MAC projection in double precision, in src/cfd/mac.ts. It does not draw this picture.",
      "kernels/karman3d.cu is the native source. Its host path can be compiled with g++. Device execution on the 3090 Ti has not been run here.",
    ],
  },
  {
    title: "What the picture is allowed to get wrong",
    body: "The preview is kept so you can orbit a volume while the reference stays strict. Several shortcuts are still in it on purpose.",
    points: [
      "Viscosity in the preview is clipped. The MAC reference never replaces ν with a clipped product of dt and ν.",
      "Divergence, gradient, and the Laplacian in the preview are not the same operator. The MAC projection uses a compatible pair.",
      "Drag to orbit. + and − or the wheel zoom. Pause, Step, and Reset act on the preview clock only.",
    ],
  },
  {
    title: "The projection you can check",
    body: "Incompressible, constant-density, Newtonian, laminar flow on a Cartesian grid. Stored pressure is kinematic: φ = p/ρ, so the correction is u := u − Δt ∇φ.",
    points: [
      "Advection is flux-form: first-order upwind, or limited linear reconstruction with Heun. A Courant-1 shift matches the translated scalar to 1e-8 on x, y, and z.",
      "A failed substep rolls the fields, the clock, and the step counter back. A closed inlet, or an outlet clamp that breaks mass balance, refuses the step.",
      "Taylor–Green decay is near first order, not second. That tolerance was not loosened to make the plot look better.",
    ],
  },
  {
    title: "Read the HUD as labeled",
    body: "Re, CFL, and time on the picture are from the preview step. St, Cd, and Cl are labeled preview because the force and the shedding period are not a verified measurement.",
    points: [
      "A voxel body and a coarse grid will move those coefficients. Do not quote them as cylinder or airfoil data.",
      "Kinetic energy and max |u| are diagnostics of the picture, not of the MAC residual.",
      "The reference reports a residual, a divergence norm, and a volume-flux defect. Those are the quantities the tests assert.",
    ],
  },
  {
    title: "Analysis status is a claim about evidence",
    body: "Available means the incompressible MAC path. Preview means a model that runs but has not been checked against a reference solution. Experimental means a small formula or a side solver. Planned means it does not run.",
    points: [
      "Convective heat transfer turns on a Boussinesq term in the preview. It is not a verified heat-transfer study.",
      "The other flow and structural rows call runExperimental. They are finite-output checks, not Navier–Stokes, LBM, or solid mechanics.",
      "Electromagnetics stays disabled. A separate CAD kernel is a later integration, not a hidden mode.",
    ],
    panel: "analysis",
  },
  {
    title: "Cases, then the grid you can afford",
    body: "Kármán street, NACA 0012, and the thermal plume are the cases worth looking at first. Lid cavity is the cleanest viscous check. The others are illustrations.",
    points: [
      "Grid size is 2N × N × N. Bench is 8,192 cells. Max preview is 524,288.",
      "About 200,000 cells is the browser comfort line. The reference solver refuses more than 2,000,000 cells.",
      "The L2, VRAM, and RAM lines are estimates for a 3090 Ti and a 5900X with 64 GB. They are not timings.",
    ],
    panel: "field",
  },
  {
    title: "Solids, heat, and the section",
    body: "Suite edits the preview. It does not redefine the MAC case file unless you export one.",
    points: [
      "Shapes are voxel stamps. Orbit stays available. Move, then click, repositions the stamp. Clear solids removes every stamp.",
      "Inlet, wall, and ambient temperature plus a streamwise pressure gradient drive the preview. Buoyancy is −β (T − Tref) g when that path is on.",
      "The mid-span section is plane stress, one way, ν = 0.30. It is not fluid–structure interaction. A singular or empty section is not a stress result.",
    ],
    panel: "suite",
  },
  {
    title: "What is not on the hot path",
    body: "Kernels lists four files. Only one is authoritative, and even that file’s GPU launch is unverified.",
    points: [
      "karman3d.cu is the MAC source. Build the host path first. nvcc -arch=sm_86 is the 3090 Ti line. Do not add --use_fast_math until the same checks have been compared.",
      "The older 2D collocated file, the PTX divergence kernel, and the AVX2 energy reduction are retained. They are not called by the solver.",
      "Geometric multigrid and the D2Q9 lattice step are separate modules with their own tests. They do not replace the pressure CG.",
    ],
    panel: "kernels",
  },
  {
    title: "How to leave the picture",
    body: "If you need a number, leave the browser and run the reference. The picture will not tell you that it failed a conservation check.",
    points: [
      "node --experimental-strip-types --test verify/*.test.mjs runs the regression suite.",
      "node --experimental-strip-types scripts/mac-run.mjs is the headless Couette check.",
      "g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu && ./karman3d is the native host check. A GPU binary that merely links is not verification.",
    ],
  },
];

const DONE_KEY = "karman-guide-v2";

export function guideWasFinished(): boolean {
  try {
    return localStorage.getItem(DONE_KEY) === "1";
  } catch {
    return false;
  }
}

export function markGuideFinished(): void {
  try {
    localStorage.setItem(DONE_KEY, "1");
  } catch {
    /* private mode */
  }
}

export function Guide({
  step,
  onStep,
  onClose,
}: {
  step: number;
  onStep: (next: number) => void;
  onClose: () => void;
}) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  const current = GUIDE[step] ?? GUIDE[0]!;
  const last = step >= GUIDE.length - 1;

  useEffect(() => {
    titleRef.current?.focus();
  }, [step]);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" && !last) onStep(step + 1);
      if (e.key === "ArrowLeft" && step > 0) onStep(step - 1);
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, last, onClose, onStep]);

  return (
    <div
      role="dialog"
      aria-labelledby="guide-title"
      aria-describedby="guide-body"
      className="fixed inset-x-3 bottom-[max(0.75rem,env(safe-area-inset-bottom))] z-30 flex max-h-[calc(100dvh-5.5rem)] max-w-2xl flex-col overflow-hidden rounded-lg border border-primary bg-bg/95 p-4 shadow-lg backdrop-blur-sm"
    >
      <div className="min-h-0 flex-1 overflow-y-auto">
      <p className="font-mono text-xs tracking-widest text-primary">
        Guide · {step + 1} of {GUIDE.length}
      </p>
      <h2 id="guide-title" ref={titleRef} tabIndex={-1} className="mt-1 text-base font-semibold tracking-tight outline-none">
        {current.title}
      </h2>
      <p id="guide-body" className="mt-2 text-pretty text-sm text-fg">
        {current.body}
      </p>
      <ul className="mt-3 list-disc space-y-1 pl-5 text-pretty text-sm text-muted">
        {current.points.map((point) => (
          <li key={point}>{point}</li>
        ))}
      </ul>
      </div>
      <p className="mt-3 font-mono text-xs text-muted">Next advances. Close leaves the guide.</p>
      <div className="mt-3 flex shrink-0 gap-2 border-t border-border pt-3">
        <button type="button" className={`${guideBtn} flex-1`} onClick={() => onStep(Math.max(0, step - 1))} disabled={step === 0}>
          Back
        </button>
        {last ? (
          <button type="button" className={`${guideBtnPrimary} flex-1`} onClick={() => onClose()}>
            Finish
          </button>
        ) : (
          <button type="button" className={`${guideBtnPrimary} flex-1`} onClick={() => onStep(step + 1)}>
            Next
          </button>
        )}
        <button type="button" className={`${guideBtn} flex-1`} onClick={() => onClose()}>
          Close
        </button>
      </div>
    </div>
  );
}

const guideBtn =
  "inline-flex h-11 items-center justify-center rounded-md border border-border bg-surface px-3 text-sm text-fg hover:border-primary disabled:cursor-not-allowed disabled:opacity-40";
const guideBtnPrimary =
  "inline-flex h-11 items-center justify-center rounded-md bg-primary px-3 text-sm font-medium text-primary-fg";
