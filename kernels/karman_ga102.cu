// karman_ga102.cu
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
