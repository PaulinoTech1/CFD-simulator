import { CUDA3D } from "./cuda3d";

export type KernelFile = {
  id: string;
  label: string;
  filename: string;
  blurb: string;
  source: string;
};

const CUDA = `// karman_ga102.cu
// Incompressible 2D Navier–Stokes, same stages as the Karman bench:
//   advect (semi-Lagrangian) -> diffuse -> Chorin projection.
//
// Target: GeForce RTX 3090 Ti, GA102, compute capability 8.6
//   84 SMs, 10752 FP32 cores, 24 GB GDDR6X, 1008 GB/s, 6 MB L2
// Build: nvcc -O3 -arch=sm_86 --use_fast_math -o karman karman_ga102.cu
//
// 16x16 threads = 256 = 8 warps. That fills an Ampere warp scheduler
// evenly and keeps the shared-memory halo square. Do not send this
// stencil to tensor cores — they are MMA units, the wrong tool here.
// Stay in FP32. A grid that fits in 6 MB L2 is latency-bound; the
// memory bus matters once you leave L2.

#include <cuda_runtime.h>
#include <math.h>

enum { F_FLUID = 0, F_SOLID = 1, F_INLET = 2, F_LID = 3, F_BODY = 4 };

__device__ __forceinline__ int blocked(unsigned char f) {
    return f == F_SOLID || f == F_BODY;
}

__device__ float bilerp(const float* __restrict__ f, float x, float y, int nx, int ny) {
    x = fminf(fmaxf(x, 0.f), (float)(nx - 1));
    y = fminf(fmaxf(y, 0.f), (float)(ny - 1));
    int i0 = (int)floorf(x);
    int j0 = (int)floorf(y);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny - 1);
    float tx = x - (float)i0;
    float ty = y - (float)j0;
    float a = f[i0 + j0 * nx];
    float b = f[i1 + j0 * nx];
    float c = f[i0 + j1 * nx];
    float d = f[i1 + j1 * nx];
    return (a * (1.f - tx) + b * tx) * (1.f - ty) + (c * (1.f - tx) + d * tx) * ty;
}

// Gather-bound. The 3090 Ti wins here by feeding the texture units if you
// later bind the field as a cudaTextureObject; global bilerp is the honest
// first port of the bench.
__global__ void __launch_bounds__(256, 4)
advect(const float* __restrict__ src, float* __restrict__ dst,
       const float* __restrict__ u, const float* __restrict__ v,
       const unsigned char* __restrict__ flags,
       int nx, int ny, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int id = i + j * nx;
    unsigned char fl = flags[id];
    if (blocked(fl)) { dst[id] = 0.f; return; }
    if (fl == F_INLET || fl == F_LID) { dst[id] = src[id]; return; }
    float bx = dt * u[id];
    float by = dt * v[id];
    float mag = hypotf(bx, by);
    if (mag > 0.85f) {
        float s = 0.85f / mag;
        bx *= s;
        by *= s;
    }
    dst[id] = bilerp(src, (float)i - bx, (float)j - by, nx, ny);
}

// Parallel Jacobi. The browser reference uses SOR Gauss–Seidel because it
// is single-threaded and warm-starts well. On GA102 the equivalent is
// Jacobi with the previous pressure as the initial guess, ping-ponged.
// Shared tile is 18x18 so the 5-point stencil does not re-fetch the halo
// from L1 on every neighbor.
__global__ void __launch_bounds__(256, 4)
pressure_jacobi(const float* __restrict__ pIn, float* __restrict__ pOut,
                const float* __restrict__ div,
                const unsigned char* __restrict__ flags,
                int nx, int ny, float invDt) {
    constexpr int B = 16;
    __shared__ float tile[B + 2][B + 2];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * B + tx;
    int j = blockIdx.y * B + ty;

    int gi = min(max(i, 0), nx - 1);
    int gj = min(max(j, 0), ny - 1);
    tile[ty + 1][tx + 1] = pIn[gi + gj * nx];
    if (tx == 0) {
        int hi = max(i - 1, 0);
        tile[ty + 1][0] = pIn[hi + gj * nx];
    }
    if (tx == B - 1) {
        int hi = min(i + 1, nx - 1);
        tile[ty + 1][B + 1] = pIn[hi + gj * nx];
    }
    if (ty == 0) {
        int hj = max(j - 1, 0);
        tile[0][tx + 1] = pIn[gi + hj * nx];
    }
    if (ty == B - 1) {
        int hj = min(j + 1, ny - 1);
        tile[B + 1][tx + 1] = pIn[gi + hj * nx];
    }
    __syncthreads();

    if (i <= 0 || j <= 0 || i >= nx - 1 || j >= ny - 1) return;
    int id = i + j * nx;
    if (blocked(flags[id])) { pOut[id] = 0.f; return; }

    float pc = tile[ty + 1][tx + 1];
    float pL = tile[ty + 1][tx];
    float pR = tile[ty + 1][tx + 2];
    float pB = tile[ty][tx + 1];
    float pT = tile[ty + 2][tx + 1];
    int nFluid = 0;
    float sum = 0.f;
    if (!blocked(flags[id - 1])) { sum += pL; nFluid++; }
    if (!blocked(flags[id + 1])) { sum += pR; nFluid++; }
    if (!blocked(flags[id - nx])) { sum += pB; nFluid++; }
    if (!blocked(flags[id + nx])) { sum += pT; nFluid++; }
    pOut[id] = nFluid ? (sum - div[id] * invDt) / (float)nFluid : 0.f;
    (void)pc;
}

__global__ void __launch_bounds__(256, 4)
subtract_gradient(float* __restrict__ u, float* __restrict__ v,
                  const float* __restrict__ p,
                  const unsigned char* __restrict__ flags,
                  int nx, int ny, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || j <= 0 || i >= nx - 1 || j >= ny - 1) return;
    int id = i + j * nx;
    unsigned char fl = flags[id];
    if (blocked(fl) || fl == F_INLET || fl == F_LID) return;
    float pc = p[id];
    float pL = blocked(flags[id - 1]) ? pc : p[id - 1];
    float pR = blocked(flags[id + 1]) ? pc : p[id + 1];
    float pB = blocked(flags[id - nx]) ? pc : p[id - nx];
    float pT = blocked(flags[id + nx]) ? pc : p[id + nx];
    u[id] -= dt * 0.5f * (pR - pL);
    v[id] -= dt * 0.5f * (pT - pB);
}

// Host side (5900X): reductions, Cd/Cl, particle compact, step schedule.
// One Jacobi iteration is a few microseconds of DRAM once the grid is
// large; the CPU should own the running means and the file IO.
static dim3 block16() { return dim3(16, 16, 1); }
static dim3 gridOf(int nx, int ny) {
    return dim3((nx + 15) / 16, (ny + 15) / 16, 1);
}

extern "C" void karman_advect(float* dst, const float* src,
    const float* u, const float* v, const unsigned char* flags,
    int nx, int ny, float dt) {
    advect<<<gridOf(nx, ny), block16()>>>(src, dst, u, v, flags, nx, ny, dt);
}

extern "C" void karman_project_iter(float* pOut, const float* pIn,
    const float* div, const unsigned char* flags,
    int nx, int ny, float invDt) {
    pressure_jacobi<<<gridOf(nx, ny), block16()>>>(pIn, pOut, div, flags, nx, ny, invDt);
}
`;

const PTX = `// divergence_sm86.ptx
// Hand-written PTX for the divergence of a collocated velocity field.
//   div = 0.5 * ((u[i+1]-u[i-1]) + (v[j+1]-v[j-1]))
// Target: sm_86 (RTX 3090 Ti). Assemble with:
//   ptxas -arch=sm_86 -O3 divergence_sm86.ptx -o divergence_sm86.cubin
// Launch with a 16x16 block, matching the CUDA side.

.version 8.4
.target sm_86
.address_size 64

.visible .entry divergence_sm86(
    .param .u64 pU,
    .param .u64 pV,
    .param .u64 pDiv,
    .param .u32 nx,
    .param .u32 ny
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<12>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;

    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %tid.y;
    mad.lo.u32 %r7, %r4, %r5, %r6;

    ld.param.u32 %r8, [nx];
    ld.param.u32 %r9, [ny];

    setp.ge.u32 %p0, %r3, %r8;
    setp.ge.or.u32 %p0, %r7, %r9, %p0;
    @%p0 bra DONE;

    mad.lo.u32 %r10, %r7, %r8, %r3;

    setp.eq.u32 %p1, %r3, 0;
    add.u32 %r11, %r8, -1;
    setp.ge.or.u32 %p1, %r3, %r11, %p1;
    setp.eq.or.u32 %p1, %r7, 0, %p1;
    add.u32 %r12, %r9, -1;
    setp.ge.or.u32 %p1, %r7, %r12, %p1;
    @%p1 bra STORE0;

    ld.param.u64 %rd0, [pU];
    ld.param.u64 %rd1, [pV];
    ld.param.u64 %rd2, [pDiv];

    add.u32 %r13, %r10, 1;
    sub.u32 %r14, %r10, 1;
    add.u32 %r15, %r10, %r8;
    sub.u32 %r16, %r10, %r8;

    mul.wide.u32 %rd3, %r13, 4;
    add.s64 %rd4, %rd0, %rd3;
    ld.global.cs.f32 %f0, [%rd4];

    mul.wide.u32 %rd3, %r14, 4;
    add.s64 %rd4, %rd0, %rd3;
    ld.global.cs.f32 %f1, [%rd4];

    mul.wide.u32 %rd3, %r15, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.cs.f32 %f2, [%rd4];

    mul.wide.u32 %rd3, %r16, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.cs.f32 %f3, [%rd4];

    sub.f32 %f4, %f0, %f1;
    sub.f32 %f5, %f2, %f3;
    add.f32 %f6, %f4, %f5;
    mul.f32 %f7, %f6, 0f3F000000;

    mul.wide.u32 %rd3, %r10, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.cs.f32 [%rd4], %f7;
    bra DONE;

STORE0:
    ld.param.u64 %rd2, [pDiv];
    mul.wide.u32 %rd3, %r10, 4;
    add.s64 %rd4, %rd2, %rd3;
    mov.f32 %f7, 0f00000000;
    st.global.cs.f32 [%rd4], %f7;

DONE:
    ret;
}
`;

const ASM = `; znver3_ke.s
; Kinetic energy on the host. The 3090 Ti should not be asked to do this
; as a scalar tail — solids are already zeroed on the device, so a flat
; AVX2 dot product on the 5900X is the whole reduction.
;
;   float ke_fluid(const float *u, const float *v, int n);
;   System V AMD64: rdi = u, rsi = v, rdx = n, result in xmm0.
;
; Zen 3 (znver3): vfmadd231ps is the right instruction. vzeroupper before
; ret keeps the upper YMM state from taxing a later SSE caller.
; Assemble: clang -c -o znver3_ke.o znver3_ke.s

.intel_syntax noprefix
.section .text
.globl ke_fluid
.type ke_fluid, @function
.align 16
ke_fluid:
    vxorps ymm0, ymm0, ymm0
    xor rax, rax
    cmp rdx, 8
    jl .horiz

.align 16
.loop:
    vmovups ymm1, ymmword ptr [rdi + rax*4]
    vfmadd231ps ymm0, ymm1, ymm1
    vmovups ymm2, ymmword ptr [rsi + rax*4]
    vfmadd231ps ymm0, ymm2, ymm2
    add rax, 8
    cmp rax, rdx
    jb .loop

.horiz:
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0
    vzeroupper

.tail:
    cmp rax, rdx
    jae .half
    movss xmm1, dword ptr [rdi + rax*4]
    mulss xmm1, xmm1
    addss xmm0, xmm1
    movss xmm1, dword ptr [rsi + rax*4]
    mulss xmm1, xmm1
    addss xmm0, xmm1
    inc rax
    jmp .tail

.half:
    mov eax, 0x3f000000
    movd xmm1, eax
    mulss xmm0, xmm1
    ret
.size ke_fluid, .-ke_fluid

; C side, so the 12 cores can split the y-range while the GPU is in the
; field kernel. Compile with: gcc -O3 -march=znver3 -c host_reduce.c
;
;   #include <stddef.h>
;   extern float ke_fluid(const float *u, const float *v, int n);
;
;   float ke_rows(const float *u, const float *v, int nx, int y0, int y1) {
;       return ke_fluid(u + y0 * nx, v + y0 * nx, (y1 - y0) * nx);
;   }
`;

export const KERNELS: readonly KernelFile[] = [
  {
    id: "cu3",
    label: "3D CUDA",
    filename: "karman3d.cu",
    blurb: "The whole 3D suite in one file. nvcc -O3 -arch=sm_86 --use_fast_math -o karman3d karman3d.cu && ./karman3d karman 64 32 32 400",
    source: CUDA3D,
  },
  {
    id: "cu",
    label: "CUDA C",
    filename: "karman_ga102.cu",
    blurb: "Advect, shared-memory Jacobi, and the gradient step. 16×16 blocks, sm_86.",
    source: CUDA,
  },
  {
    id: "ptx",
    label: "PTX",
    filename: "divergence_sm86.ptx",
    blurb: "Hand-written sm_86 divergence. This is the assembly the card actually schedules.",
    source: PTX,
  },
  {
    id: "s",
    label: "AVX2",
    filename: "znver3_ke.s",
    blurb: "Host kinetic-energy reduction for the 5900X. Solids are already zero, so there is no mask.",
    source: ASM,
  },
];
