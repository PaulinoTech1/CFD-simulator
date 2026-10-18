import { useEffect, useRef } from "react";

export type GuidePanel = "analysis" | "field" | "suite" | "skills" | "rig" | "kernels";

export type GuideStep = {
  title: string;
  body: string;
  panel?: GuidePanel;
};

export const GUIDE: readonly GuideStep[] = [
  {
    title: "What you are looking at",
    body: "Karman is a fluid-flow bench. The picture on the left is a live CPU preview, not a GPU run and not an engineering-validated result. The checked solver is a separate headless calculation.",
  },
  {
    title: "Move the picture",
    body: "Drag the picture to orbit. Use + and −, or the scroll wheel, to zoom. Pause stops the clock, Step advances one frame, and Reset returns the case to its start.",
  },
  {
    title: "Read the numbers",
    body: "St, Cd, and Cl are labeled preview. They are estimates from the picture, not measured force coefficients. Reynolds number, CFL, and time sit on the picture itself.",
  },
  {
    title: "Choose a model",
    body: "Open Analysis. Available is the checked incompressible path. Preview and Experimental run a small estimate and say so. Planned stays disabled. Electromagnetics is not in this project.",
    panel: "analysis",
  },
  {
    title: "Pick a case and a grid",
    body: "Open Field. Kármán street, the NACA 0012 airfoil, and the thermal plume are the main cases. Grid size lists the cell count and whether it fits the browser, the reference cap, and the 3090 Ti memory estimate.",
    panel: "field",
  },
  {
    title: "Place a solid",
    body: "Open Suite. Pick a shape, then orbit to look at it. Move, then click, to reposition. Temperature and the streamwise pressure gradient live here. The section stress is a one-way estimate, not fluid-structure interaction.",
    panel: "suite",
  },
  {
    title: "What is actually checked",
    body: "The browser picture uses a simpler method so you can see the flow. The staggered MAC solver is what the numerical tests check. Device execution on the 3090 Ti is still unverified until you compile and run it on that machine.",
    panel: "kernels",
  },
];

const DONE_KEY = "karman-guide-done";

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
      className="absolute inset-x-3 bottom-3 z-20 max-w-xl rounded-lg border border-primary bg-bg/95 p-4 shadow-lg backdrop-blur-sm"
    >
      <p className="font-mono text-xs tracking-widest text-primary">
        Guide · {step + 1} of {GUIDE.length}
      </p>
      <h2 id="guide-title" ref={titleRef} tabIndex={-1} className="mt-1 text-base font-semibold tracking-tight outline-none">
        {current.title}
      </h2>
      <p id="guide-body" className="mt-2 text-pretty text-sm text-muted">
        {current.body}
      </p>
      <p className="mt-3 font-mono text-xs text-muted">Arrow keys move. Escape skips.</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" className={guideBtn} onClick={() => onStep(Math.max(0, step - 1))} disabled={step === 0}>
          Back
        </button>
        {last ? (
          <button type="button" className={guideBtnPrimary} onClick={() => onClose()}>
            Finish
          </button>
        ) : (
          <button type="button" className={guideBtnPrimary} onClick={() => onStep(step + 1)}>
            Next
          </button>
        )}
        <button type="button" className={guideBtn} onClick={() => onClose()}>
          Skip
        </button>
      </div>
    </div>
  );
}

const guideBtn =
  "inline-flex h-11 items-center justify-center rounded-md border border-border bg-surface px-3 text-sm text-fg hover:border-primary disabled:cursor-not-allowed disabled:opacity-40";
const guideBtnPrimary =
  "inline-flex h-11 items-center justify-center rounded-md bg-primary px-3 text-sm font-medium text-primary-fg";
