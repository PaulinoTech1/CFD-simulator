/** Bare 3D incompressible solver. No CUDA libraries, no frameworks. */
export const CUDA3D = `// karman3d.cu
// 3D incompressible Navier–Stokes for a GeForce RTX 3090 Ti.
// GA102, compute capability 8.6, 84 SMs, 24 GB GDDR6X.
//
// This file is the suite. One translation unit, the C++ the driver
// already speaks, no Kokkos, no AMReX, no vendor CFD library.
//
//   nvcc -O3 -arch=sm_86 --use_fast_math -o karman3d karman3d.cu
//   ./karman3d karman  64 32 32 400 0.2
// The last number is an imposed streamwise pressure gradient. Omit it for 0.
//   ./karman3d airfoil 64 32 32 400
//   ./karman3d cavity  48 32 32 300
//   ./karman3d plume   48 32 32 300
//   ./karman3d jet     64 32 32 300
//   ./karman3d kelvin  64 32 32 200
//   ./karman3d ink     48 32 32 100
//
// Stages match the preview: semi-Lagrangian advection, Jacobi
// pressure, gradient projection, then dye. Jacobi, not SOR —
// SOR is serial. The preview's SOR is the single-thread form of
// the same Poisson problem.

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { F_FLUID = 0, F_SOLID = 1, F_INLET = 2, F_LID = 3, F_BODY = 4, F_OUTLET = 5 };

#define CHECK(call) do { \\
    cudaError_t _e = (call); \\
    if (_e != cudaSuccess) { \\
        fprintf(stderr, "%s:%d %s\\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \\
        exit(1); \\
    } \\
} while (0)

static int idx(int i, int j, int k, int nx, int ny) { return i + nx * (j + ny * k); }

__device__ __forceinline__ int dblocked(unsigned char f) {
    return f == F_SOLID || f == F_BODY;
}

__device__ float trilerp(const float* __restrict__ f,
                          float x, float y, float z, int nx, int ny, int nz) {
    x = fminf(fmaxf(x, 0.f), (float)(nx - 1));
    y = fminf(fmaxf(y, 0.f), (float)(ny - 1));
    z = fminf(fmaxf(z, 0.f), (float)(nz - 1));
    int i0 = (int)floorf(x), j0 = (int)floorf(y), k0 = (int)floorf(z);
    int i1 = min(i0 + 1, nx - 1);
    int j1 = min(j0 + 1, ny - 1);
    int k1 = min(k0 + 1, nz - 1);
    float tx = x - (float)i0, ty = y - (float)j0, tz = z - (float)k0;
    int nxy = nx * ny;
    float c00 = f[i0 + nx * j0 + nxy * k0] * (1.f - tx) + f[i1 + nx * j0 + nxy * k0] * tx;
    float c10 = f[i0 + nx * j1 + nxy * k0] * (1.f - tx) + f[i1 + nx * j1 + nxy * k0] * tx;
    float c01 = f[i0 + nx * j0 + nxy * k1] * (1.f - tx) + f[i1 + nx * j0 + nxy * k1] * tx;
    float c11 = f[i0 + nx * j1 + nxy * k1] * (1.f - tx) + f[i1 + nx * j1 + nxy * k1] * tx;
    return (c00 * (1.f - ty) + c10 * ty) * (1.f - tz) + (c01 * (1.f - ty) + c11 * ty) * tz;
}

__global__ void __launch_bounds__(256, 4)
k_advect(const float* __restrict__ src, float* __restrict__ dst,
         const float* __restrict__ u, const float* __restrict__ v, const float* __restrict__ w,
         const unsigned char* __restrict__ flags,
         int nx, int ny, int nz, float dt, int dye_mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int id = i + nx * (j + ny * k);
    unsigned char fl = flags[id];
    if (dblocked(fl)) { dst[id] = 0.f; return; }
    if (dye_mode && fl == F_INLET) { dst[id] = 1.f; return; }
    float bx = dt * u[id], by = dt * v[id], bz = dt * w[id];
    float mag = sqrtf(bx * bx + by * by + bz * bz);
    if (mag > 0.85f) { float s = 0.85f / mag; bx *= s; by *= s; bz *= s; }
    dst[id] = trilerp(src, (float)i - bx, (float)j - by, (float)k - bz, nx, ny, nz);
}

__global__ void __launch_bounds__(256, 4)
k_enforce(float* __restrict__ u, float* __restrict__ v, float* __restrict__ w,
          const unsigned char* __restrict__ flags,
          int nx, int ny, int nz, float u_in) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int id = i + nx * (j + ny * k);
    unsigned char fl = flags[id];
    if (dblocked(fl)) { u[id] = v[id] = w[id] = 0.f; return; }
    if (fl == F_INLET || fl == F_LID) { u[id] = u_in; v[id] = w[id] = 0.f; return; }
    if (fl == F_OUTLET && i > 0) {
        int up = id - 1;
        u[id] = u[up]; v[id] = v[up]; w[id] = w[up];
    }
}

__global__ void __launch_bounds__(256, 4)
k_grad(float* __restrict__ u, const unsigned char* __restrict__ flags, int n, float impulse) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n || dblocked(flags[id]) || impulse == 0.f) return;
    u[id] += impulse;
}

__global__ void __launch_bounds__(256, 4)
k_buoy(float* __restrict__ v, const float* __restrict__ dye,
       const unsigned char* __restrict__ flags, int n, float scale) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n || dblocked(flags[id]) || scale == 0.f) return;
    v[id] += scale * dye[id];
}

__global__ void __launch_bounds__(256, 4)
k_div(const float* __restrict__ u, const float* __restrict__ v, const float* __restrict__ w,
      const unsigned char* __restrict__ flags, float* __restrict__ divg,
      int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int id = i + nx * (j + ny * k);
    int nxy = nx * ny;
    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 || dblocked(flags[id])) {
        divg[id] = 0.f;
        return;
    }
    float uR = dblocked(flags[id + 1]) ? 0.f : u[id + 1];
    float uL = dblocked(flags[id - 1]) ? 0.f : u[id - 1];
    float vT = dblocked(flags[id + nx]) ? 0.f : v[id + nx];
    float vB = dblocked(flags[id - nx]) ? 0.f : v[id - nx];
    float wF = dblocked(flags[id + nxy]) ? 0.f : w[id + nxy];
    float wK = dblocked(flags[id - nxy]) ? 0.f : w[id - nxy];
    divg[id] = 0.5f * (uR - uL + (vT - vB) + (wF - wK));
}

__global__ void __launch_bounds__(256, 4)
k_jacobi(const float* __restrict__ pIn, float* __restrict__ pOut,
         const float* __restrict__ divg, const unsigned char* __restrict__ flags,
         int nx, int ny, int nz, float inv_dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    int id = i + nx * (j + ny * k);
    int nxy = nx * ny;
    unsigned char fl = flags[id];
    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1 || dblocked(fl) || fl == F_OUTLET) {
        pOut[id] = 0.f;
        return;
    }
    int nbr[6] = { id - 1, id + 1, id - nx, id + nx, id - nxy, id + nxy };
    float sum = 0.f;
    int nfluid = 0;
    for (int s = 0; s < 6; ++s) {
        if (!dblocked(flags[nbr[s]])) { sum += pIn[nbr[s]]; ++nfluid; }
    }
    pOut[id] = nfluid ? (sum - divg[id] * inv_dt) / (float)nfluid : 0.f;
}

__global__ void __launch_bounds__(256, 4)
k_project(float* __restrict__ u, float* __restrict__ v, float* __restrict__ w,
          const float* __restrict__ p, const unsigned char* __restrict__ flags,
          int nx, int ny, int nz, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i <= 0 || j <= 0 || k <= 0 || i >= nx - 1 || j >= ny - 1 || k >= nz - 1) return;
    int id = i + nx * (j + ny * k);
    int nxy = nx * ny;
    unsigned char fl = flags[id];
    if (dblocked(fl) || fl == F_INLET || fl == F_LID) return;
    float pc = p[id];
    float pL = dblocked(flags[id - 1]) ? pc : p[id - 1];
    float pR = dblocked(flags[id + 1]) ? pc : p[id + 1];
    float pB = dblocked(flags[id - nx]) ? pc : p[id - nx];
    float pT = dblocked(flags[id + nx]) ? pc : p[id + nx];
    float pK = dblocked(flags[id - nxy]) ? pc : p[id - nxy];
    float pF = dblocked(flags[id + nxy]) ? pc : p[id + nxy];
    u[id] -= dt * 0.5f * (pR - pL);
    v[id] -= dt * 0.5f * (pT - pB);
    w[id] -= dt * 0.5f * (pF - pK);
}

__global__ void __launch_bounds__(256, 2)
k_ke(const float* __restrict__ u, const float* __restrict__ v, const float* __restrict__ w,
     const unsigned char* __restrict__ flags, float* __restrict__ partial, int n) {
    __shared__ float buf[256];
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    float e = 0.f;
    if (id < n && !dblocked(flags[id])) {
        float uu = u[id], vv = v[id], ww = w[id];
        e = 0.5f * (uu * uu + vv * vv + ww * ww);
    }
    buf[threadIdx.x] = e;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) buf[threadIdx.x] += buf[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = buf[0];
}

static int in_airfoil(int i, int j, float cx, float cy, float chord, float aoa_deg) {
    float a = aoa_deg * 0.0174532925f;
    float dx = (float)i - cx, dy = (float)j - cy;
    float xl = dx * cosf(a) - dy * sinf(a);
    float yl = dx * sinf(a) + dy * cosf(a);
    if (xl < -0.5f || xl > chord + 0.5f) return 0;
    float x = xl / chord;
    if (x < 0.f) x = 0.f;
    if (x > 1.f) x = 1.f;
    float yt = 5.f * 0.12f * (0.2969f * sqrtf(x) - 0.126f * x - 0.3516f * x * x
                               + 0.2843f * x * x * x - 0.1015f * x * x * x * x);
    float half = yt * chord;
    if (half < 1.25f) half = 1.25f;
    return fabsf(yl) <= half;
}

static void build(const char* preset, int nx, int ny, int nz,
                  unsigned char* flags, float* u, float* v, float* w, float* dye,
                  float* u_in, float* re, float* length) {
    int n = nx * ny * nz;
    memset(flags, 0, (size_t)n);
    memset(u, 0, sizeof(float) * (size_t)n);
    memset(v, 0, sizeof(float) * (size_t)n);
    memset(w, 0, sizeof(float) * (size_t)n);
    memset(dye, 0, sizeof(float) * (size_t)n);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                if (j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
                    flags[idx(i, j, k, nx, ny)] = F_SOLID;
    float cx = 0.28f * (float)nx;
    float cy = 0.5f * (float)(ny - 1);
    float cz = 0.5f * (float)(nz - 1);
    float radius = fmaxf(2.4f, 0.18f * (float)ny);
    float chord = fmaxf(8.f, 0.72f * (float)ny);
    *u_in = 1.f;
    *re = 180.f;
    *length = 2.f * radius;
    int tunnel = !strcmp(preset, "karman") || !strcmp(preset, "airfoil") || !strcmp(preset, "jet");
    if (tunnel) {
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j) {
                flags[idx(0, j, k, nx, ny)] = F_INLET;
                flags[idx(nx - 1, j, k, nx, ny)] = F_OUTLET;
            }
    }
    if (!strcmp(preset, "karman")) {
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i) {
                    float dx = (float)i - cx, dy = (float)j - cy;
                    if (dx * dx + dy * dy <= radius * radius)
                        flags[idx(i, j, k, nx, ny)] = F_BODY;
                }
    } else if (!strcmp(preset, "airfoil")) {
        *re = 400.f;
        *length = chord;
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 2; ++i)
                    if (in_airfoil(i, j, cx, cy, chord, 8.f))
                        flags[idx(i, j, k, nx, ny)] = F_BODY;
    } else if (!strcmp(preset, "cavity")) {
        *re = 100.f;
        *length = (float)(ny - 2);
        for (int k = 1; k < nz - 1; ++k)
            for (int i = 1; i < nx - 1; ++i)
                flags[idx(i, ny - 1, k, nx, ny)] = F_LID;
    } else if (!strcmp(preset, "plume")) {
        *u_in = 0.f;
        *re = 40.f;
        *length = 0.45f * (float)ny;
        for (int k = 1; k < nz - 1; ++k)
            for (int i = 1; i < nx - 1; ++i)
                flags[idx(i, ny - 1, k, nx, ny)] = F_OUTLET;
        for (int k = 1; k < nz - 1; ++k)
            for (int i = 1; i < nx - 1; ++i) {
                float dx = (float)i - 0.5f * (float)nx, dz = (float)k - cz;
                if (dx * dx + dz * dz > radius * radius) continue;
                dye[idx(i, 1, k, nx, ny)] = 1.f;
                dye[idx(i, 2, k, nx, ny)] = 1.f;
            }
    } else if (!strcmp(preset, "jet")) {
        *u_in = 1.4f;
        *re = 300.f;
        float jr = fmaxf(2.f, 0.14f * (float)ny);
        *length = 2.f * jr;
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j) {
                float dy = (float)j - cy, dz = (float)k - cz;
                flags[idx(0, j, k, nx, ny)] = (dy * dy + dz * dz <= jr * jr) ? F_INLET : F_SOLID;
            }
    } else if (!strcmp(preset, "kelvin")) {
        *u_in = 0.f;
        *re = 500.f;
        *length = 0.5f * (float)ny;
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i) {
                    float s = tanhf(((float)j - cy) / 1.5f);
                    int id = idx(i, j, k, nx, ny);
                    u[id] = 0.85f * s;
                    v[id] = 0.12f * sinf(6.2831853f * (float)i / (float)nx);
                    dye[id] = s > 0.f ? 1.f : 0.02f;
                }
    } else {
        *u_in = 0.f;
        *re = 80.f;
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = 1; i < nx - 1; ++i) {
                    float dx = (float)i - 0.45f * (float)nx;
                    float dy = (float)j - cy, dz = (float)k - cz;
                    if (dx * dx + dy * dy + dz * dz < radius * radius)
                        dye[idx(i, j, k, nx, ny)] = 1.f;
                }
    }
    if (*u_in != 0.f) {
        for (int id = 0; id < n; ++id)
            if (flags[id] == F_FLUID || flags[id] == F_OUTLET) u[id] = *u_in;
    }
    if (!strcmp(preset, "karman")) {
        for (int k = 1; k < nz - 1; ++k)
            for (int j = 1; j < ny - 1; ++j)
                for (int i = (int)(cx + radius); i < nx - 1; ++i) {
                    int id = idx(i, j, k, nx, ny);
                    if (flags[id] == F_FLUID)
                        v[id] = (*u_in) * 0.07f * sinf(3.14159265f * ((float)i - cx) / (float)nx);
                }
    }
}

int main(int argc, char** argv) {
    const char* preset = argc > 1 ? argv[1] : "karman";
    int nx = argc > 2 ? atoi(argv[2]) : 64;
    int ny = argc > 3 ? atoi(argv[3]) : 32;
    int nz = argc > 4 ? atoi(argv[4]) : 32;
    int steps = argc > 5 ? atoi(argv[5]) : 200;
    if (nx < 8 || ny < 8 || nz < 8) { fprintf(stderr, "grid too small\\n"); return 1; }
    int n = nx * ny * nz;
    unsigned char* flags = (unsigned char*)malloc((size_t)n);
    float *u = (float*)calloc((size_t)n, 4);
    float *v = (float*)calloc((size_t)n, 4);
    float *w = (float*)calloc((size_t)n, 4);
    float *dye = (float*)calloc((size_t)n, 4);
    float u_in = 1.f, re = 180.f, length = 4.f;
    build(preset, nx, ny, nz, flags, u, v, w, dye, &u_in, &re, &length);

    float *d_u, *d_v, *d_w, *d_ut, *d_vt, *d_wt, *d_dye, *d_dyet, *d_p, *d_p2, *d_div, *d_part;
    unsigned char* d_flags;
    CHECK(cudaMalloc(&d_u, sizeof(float) * n));
    CHECK(cudaMalloc(&d_v, sizeof(float) * n));
    CHECK(cudaMalloc(&d_w, sizeof(float) * n));
    CHECK(cudaMalloc(&d_ut, sizeof(float) * n));
    CHECK(cudaMalloc(&d_vt, sizeof(float) * n));
    CHECK(cudaMalloc(&d_wt, sizeof(float) * n));
    CHECK(cudaMalloc(&d_dye, sizeof(float) * n));
    CHECK(cudaMalloc(&d_dyet, sizeof(float) * n));
    CHECK(cudaMalloc(&d_p, sizeof(float) * n));
    CHECK(cudaMalloc(&d_p2, sizeof(float) * n));
    CHECK(cudaMalloc(&d_div, sizeof(float) * n));
    CHECK(cudaMalloc(&d_flags, n));
    int blocks1 = (n + 255) / 256;
    CHECK(cudaMalloc(&d_part, sizeof(float) * blocks1));
    CHECK(cudaMemcpy(d_u, u, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_v, v, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, w, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dye, dye, sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_flags, flags, n, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_p, 0, sizeof(float) * n));
    CHECK(cudaMemset(d_p2, 0, sizeof(float) * n));

    dim3 block(8, 8, 4);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 3) / 4);
    float nu = (fmaxf(fabsf(u_in), 0.35f) * length) / fmaxf(re, 1.f);
    float buoy = !strcmp(preset, "plume") ? 2.4f : 0.f;
    float grad = argc > 6 ? (float)atof(argv[6]) : 0.f;
    printf("karman3d %s  %d x %d x %d  Re %.0f  nu %.4f  sm_86\\n", preset, nx, ny, nz, re, nu);
    printf("block 8x8x4   grid %u x %u x %u\\n", grid.x, grid.y, grid.z);

    float* partial = (float*)malloc(sizeof(float) * (size_t)blocks1);
    for (int step = 1; step <= steps; ++step) {
        float maxs = fmaxf(fabsf(u_in), 0.4f);
        float dt = fminf(0.45f, 0.7f / maxs);
        k_buoy<<<blocks1, 256>>>(d_v, d_dye, d_flags, n, dt * buoy);
        k_grad<<<blocks1, 256>>>(d_u, d_flags, n, dt * grad);
        k_enforce<<<grid, block>>>(d_u, d_v, d_w, d_flags, nx, ny, nz, u_in);
        k_advect<<<grid, block>>>(d_u, d_ut, d_u, d_v, d_w, d_flags, nx, ny, nz, dt, 0);
        k_advect<<<grid, block>>>(d_v, d_vt, d_u, d_v, d_w, d_flags, nx, ny, nz, dt, 0);
        k_advect<<<grid, block>>>(d_w, d_wt, d_u, d_v, d_w, d_flags, nx, ny, nz, dt, 0);
        float* su = d_u; d_u = d_ut; d_ut = su;
        float* sv = d_v; d_v = d_vt; d_vt = sv;
        float* sw = d_w; d_w = d_wt; d_wt = sw;
        k_enforce<<<grid, block>>>(d_u, d_v, d_w, d_flags, nx, ny, nz, u_in);
        k_div<<<grid, block>>>(d_u, d_v, d_w, d_flags, d_div, nx, ny, nz);
        float* pin = d_p;
        float* pout = d_p2;
        for (int it = 0; it < 20; ++it) {
            k_jacobi<<<grid, block>>>(pin, pout, d_div, d_flags, nx, ny, nz, 1.f / dt);
            float* tmp = pin; pin = pout; pout = tmp;
        }
        d_p = pin;
        d_p2 = pout;
        k_project<<<grid, block>>>(d_u, d_v, d_w, d_p, d_flags, nx, ny, nz, dt);
        k_enforce<<<grid, block>>>(d_u, d_v, d_w, d_flags, nx, ny, nz, u_in);
        k_advect<<<grid, block>>>(d_dye, d_dyet, d_u, d_v, d_w, d_flags, nx, ny, nz, dt, 1);
        float* sd = d_dye; d_dye = d_dyet; d_dyet = sd;
        if (step % 20 == 0 || step == steps) {
            k_ke<<<blocks1, 256>>>(d_u, d_v, d_w, d_flags, d_part, n);
            CHECK(cudaMemcpy(partial, d_part, sizeof(float) * blocks1, cudaMemcpyDeviceToHost));
            float ke = 0.f;
            for (int b = 0; b < blocks1; ++b) ke += partial[b];
            printf("step %4d  t %6.2f  KE %.4e\\n", step, dt * (float)step, ke);
        }
    }
    CHECK(cudaDeviceSynchronize());
    printf("done\\n");
    return 0;
}
`;
