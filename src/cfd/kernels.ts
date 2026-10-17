import cuda3d from "../../kernels/karman3d.cu?raw";
import cuda2d from "../../kernels/karman_ga102.cu?raw";
import ptx from "../../kernels/divergence_sm86.ptx?raw";
import avx from "../../kernels/znver3_ke.s?raw";

export type KernelFile = {
  id: string;
  label: string;
  filename: string;
  blurb: string;
  source: string;
};

export const KERNELS: readonly KernelFile[] = [
  {
    id: "cu3",
    label: "3D CUDA",
    filename: "karman3d.cu",
    blurb: "Authoritative MAC source. Validation: g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d karman3d.cu && ./karman3d. Device: nvcc -O3 -arch=sm_86 -o karman3d karman3d.cu. sm_86 is the 3090 Ti; pass another -arch to retarget. No --use_fast_math on a validation build.",
    source: cuda3d,
  },
  {
    id: "cu",
    label: "CUDA C",
    filename: "karman_ga102.cu",
    blurb: "Earlier 2D collocated kernel. Not the verified MAC scheme. 16×16 blocks, sm_86.",
    source: cuda2d,
  },
  {
    id: "ptx",
    label: "PTX",
    filename: "divergence_sm86.ptx",
    blurb: "Hand-written divergence stencil. Not wired into the MAC residual until a profile says it should be.",
    source: ptx,
  },
  {
    id: "avx",
    label: "AVX2",
    filename: "znver3_ke.s",
    blurb: "Zen 3 kinetic-energy reduction. Held back until the MAC energy definition is what it sums.",
    source: avx,
  },
];
