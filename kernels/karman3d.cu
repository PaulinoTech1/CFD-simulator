// karman3d.cu
// Authoritative MAC solver for constant-density laminar flow.
// Stored pressure is kinematic: phi = p / rho.
//
// Validation build, no GPU and no fast-math:
//   g++ -O2 -std=c++17 -DKARMAN_CPU -x c++ -o karman3d kernels/karman3d.cu
//   ./karman3d
//
// Device build. Pick the architecture at compile time. sm_86 is the 3090 Ti.
// Do not add --use_fast_math until its effect on these checks has been measured.
//   nvcc -O3 -arch=sm_86 -o karman3d kernels/karman3d.cu
//
// The host path below is what ./karman3d checks. GPU launches are compiled only
// by nvcc and are not executed by the validation main. This environment has
// neither nvcc nor a GPU, so device results are unverified.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#if defined(__CUDACC__) && !defined(KARMAN_CPU)
#define KARMAN_HAS_CUDA 1
#include <cuda_runtime.h>
#else
#define KARMAN_HAS_CUDA 0
#endif

struct Mac {
    int nx, ny, nz;
    double dx, dy, dz, nu;
    std::vector<double> u, v, w, phi, rhs, x, r, p, ap;
    explicit Mac(int nx, int ny, int nz, double lx, double ly, double lz, double nu)
        : nx(nx), ny(ny), nz(nz), dx(lx / nx), dy(ly / ny), dz(lz / nz), nu(nu),
          u((nx + 1) * ny * nz), v(nx * (ny + 1) * nz), w(nx * ny * (nz + 1)),
          phi(nx * ny * nz), rhs(nx * ny * nz), x(nx * ny * nz), r(nx * ny * nz),
          p(nx * ny * nz), ap(nx * ny * nz) {}
    int iu(int i, int j, int k) const { return i + (nx + 1) * (j + ny * k); }
    int iv(int i, int j, int k) const { return i + nx * (j + (ny + 1) * k); }
    int iw(int i, int j, int k) const { return i + nx * (j + ny * k); }
    int ic(int i, int j, int k) const { return i + nx * (j + ny * k); }
};

static void diffuse_shear(Mac& m, double dt) {
    const double inv = 1.0 / (m.dy * m.dy);
    std::vector<double> next = m.u;
    for (int j = 0; j < m.ny; ++j) {
        const int jm = (j + m.ny - 1) % m.ny;
        const int jp = (j + 1) % m.ny;
        for (int i = 0; i < m.nx; ++i) {
            const double c = m.u[m.iu(i, j, 0)];
            const double lap = (m.u[m.iu(i, jm, 0)] - 2 * c + m.u[m.iu(i, jp, 0)]) * inv;
            next[m.iu(i, j, 0)] = c + dt * m.nu * lap;
        }
    }
    for (int j = 0; j < m.ny; ++j) m.u[m.iu(m.nx, j, 0)] = next[m.iu(0, j, 0)];
    m.u.swap(next);
    for (int j = 0; j < m.ny; ++j) m.u[m.iu(m.nx, j, 0)] = m.u[m.iu(0, j, 0)];
}

static void apply_a(const Mac& m, const std::vector<double>& x, std::vector<double>& y) {
    const double invx = 1.0 / (m.dx * m.dx);
    const double invy = 1.0 / (m.dy * m.dy);
    const double invz = 1.0 / (m.dz * m.dz);
    for (int k = 0; k < m.nz; ++k) {
        for (int j = 0; j < m.ny; ++j) {
            for (int i = 0; i < m.nx; ++i) {
                const int c = m.ic(i, j, k);
                const auto nb = [&](int ni, int nj, int nk, double inv, double& diag, double& off) {
                    ni = (ni + m.nx) % m.nx;
                    nj = (nj + m.ny) % m.ny;
                    nk = (nk + m.nz) % m.nz;
                    diag += inv;
                    off += inv * x[m.ic(ni, nj, nk)];
                };
                double diag = 0, off = 0;
                nb(i - 1, j, k, invx, diag, off);
                nb(i + 1, j, k, invx, diag, off);
                nb(i, j - 1, k, invy, diag, off);
                nb(i, j + 1, k, invy, diag, off);
                nb(i, j, k - 1, invz, diag, off);
                nb(i, j, k + 1, invz, diag, off);
                y[c] = diag * x[c] - off;
            }
        }
    }
}

static void deflate(std::vector<double>& v) {
    double s = 0;
    for (double q : v) s += q;
    const double mean = s / static_cast<double>(v.size());
    for (double& q : v) q -= mean;
}

static void sync_periodic(Mac& m) {
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j) m.u[m.iu(m.nx, j, k)] = m.u[m.iu(0, j, k)];
    for (int k = 0; k < m.nz; ++k)
        for (int i = 0; i < m.nx; ++i) m.v[m.iv(i, m.ny, k)] = m.v[m.iv(i, 0, k)];
    for (int j = 0; j < m.ny; ++j)
        for (int i = 0; i < m.nx; ++i) m.w[m.iw(i, j, m.nz)] = m.w[m.iw(i, j, 0)];
}

static bool project(Mac& m, double dt, int max_iter, double rtol, int* iters, double* residual) {
    sync_periodic(m);
    const int n = m.nx * m.ny * m.nz;
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const int c = m.ic(i, j, k);
                const double div =
                    (m.u[m.iu(i + 1, j, k)] - m.u[m.iu(i, j, k)]) / m.dx +
                    (m.v[m.iv(i, j + 1, k)] - m.v[m.iv(i, j, k)]) / m.dy +
                    (m.w[m.iw(i, j, k + 1)] - m.w[m.iw(i, j, k)]) / m.dz;
                m.rhs[c] = -div / dt;
            }
    deflate(m.rhs);
    m.x = m.phi;
    deflate(m.x);
    apply_a(m, m.x, m.ap);
    double bb = 0, rr = 0;
    for (int c = 0; c < n; ++c) {
        m.r[c] = m.rhs[c] - m.ap[c];
        bb += m.rhs[c] * m.rhs[c];
    }
    deflate(m.r);
    for (int c = 0; c < n; ++c) {
        m.p[c] = m.r[c];
        rr += m.r[c] * m.r[c];
    }
    const double tol = rtol * std::sqrt(bb);
    int it = 0;
    double res = std::sqrt(rr);
    bool ok = res <= tol;
    while (!ok && it < max_iter) {
        deflate(m.p);
        apply_a(m, m.p, m.ap);
        double pap = 0;
        for (int c = 0; c < n; ++c) pap += m.p[c] * m.ap[c];
        if (!(pap > 0)) break;
        const double alpha = rr / pap;
        for (int c = 0; c < n; ++c) {
            m.x[c] += alpha * m.p[c];
            m.r[c] -= alpha * m.ap[c];
        }
        deflate(m.r);
        deflate(m.x);
        double rr2 = 0;
        for (int c = 0; c < n; ++c) rr2 += m.r[c] * m.r[c];
        ++it;
        res = std::sqrt(rr2);
        if (res <= tol) {
            ok = true;
            break;
        }
        const double beta = rr2 / rr;
        for (int c = 0; c < n; ++c) m.p[c] = m.r[c] + beta * m.p[c];
        rr = rr2;
    }
    m.phi = m.x;
    deflate(m.phi);
    sync_periodic(m);
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const int c = m.ic(i, j, k);
                const int e = m.ic((i + 1) % m.nx, j, k);
                m.u[m.iu(i, j, k)] -= dt * (m.phi[c] - m.phi[m.ic((i - 1 + m.nx) % m.nx, j, k)]) / m.dx;
                m.v[m.iv(i, j, k)] -= dt * (m.phi[c] - m.phi[m.ic(i, (j - 1 + m.ny) % m.ny, k)]) / m.dy;
                m.w[m.iw(i, j, k)] -= dt * (m.phi[c] - m.phi[m.ic(i, j, (k - 1 + m.nz) % m.nz)]) / m.dz;
                (void)e;
            }
    sync_periodic(m);
    *iters = it;
    *residual = res;
    return ok;
}

static double div_linf(const Mac& m) {
    double mmax = 0;
    for (int k = 0; k < m.nz; ++k)
        for (int j = 0; j < m.ny; ++j)
            for (int i = 0; i < m.nx; ++i) {
                const double div =
                    (m.u[m.iu(i + 1, j, k)] - m.u[m.iu(i, j, k)]) / m.dx +
                    (m.v[m.iv(i, j + 1, k)] - m.v[m.iv(i, j, k)]) / m.dy +
                    (m.w[m.iw(i, j, k + 1)] - m.w[m.iw(i, j, k)]) / m.dz;
                mmax = std::max(mmax, std::abs(div));
            }
    return mmax;
}

#if KARMAN_HAS_CUDA
__global__ void k_diffuse_shear(float* u, int nx, int ny, float dy, float nu, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int jm = (j + ny - 1) % ny;
    int jp = (j + 1) % ny;
    float c = u[i + (nx + 1) * j];
    float lap = (u[i + (nx + 1) * jm] - 2.f * c + u[i + (nx + 1) * jp]) / (dy * dy);
    u[i + (nx + 1) * j] = c + dt * nu * lap;
}
#endif

int main() {
    int failures = 0;
    {
        const int n = 24;
        const double nu = 0.02;
        Mac m(n, n, 1, 1, 1, 1, nu);
        for (int j = 0; j < n; ++j) {
            const double value = std::sin(2 * M_PI * (j + 0.5) / n);
            for (int i = 0; i <= n; ++i) m.u[m.iu(i, j, 0)] = value;
        }
        const double kappa = 2 * (std::cos(2 * M_PI / n) - 1) / (m.dy * m.dy);
        const double inv_h = 2 / (m.dx * m.dx) + 2 / (m.dy * m.dy) + 2 / (m.dz * m.dz);
        const double dt = 0.2 * 0.85 / (nu * inv_h);
        diffuse_shear(m, dt);
        double err = 0;
        for (int j = 0; j < n; ++j) {
            const double expected = std::sin(2 * M_PI * (j + 0.5) / n) * (1 + dt * nu * kappa);
            err = std::max(err, std::abs(m.u[m.iu(0, j, 0)] - expected));
        }
        const bool pass = err < 1e-12;
        std::printf("decay  err %.3e  %s\n", err, pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    {
        Mac m(8, 8, 8, 1, 1, 1, 0);
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j < m.ny; ++j)
                for (int i = 0; i <= m.nx; ++i) m.u[m.iu(i, j, k)] = std::sin(0.4 * i);
        for (int k = 0; k < m.nz; ++k)
            for (int j = 0; j <= m.ny; ++j)
                for (int i = 0; i < m.nx; ++i) m.v[m.iv(i, j, k)] = 0.1 * j;
        int iters = 0;
        double residual = 0;
        const bool converged = project(m, 1e-3, 500, 1e-8, &iters, &residual);
        const double div = div_linf(m);
        const bool pass = converged && div < 1e-6;
        std::printf("project  div %.3e  iters %d  residual %.3e  %s\n", div, iters, residual, pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    {
        const int n = 32;
        const double nu = 0.05;
        const double speed = 1.0;
        const double dy = 1.0 / n;
        std::vector<double> u(n, 0.0);
        double time = 0;
        const double dt = 0.2 * dy * dy / nu;
        while (time < 40.0) {
            std::vector<double> next(n);
            const double south = -u[0];
            const double north = 2.0 * speed - u[n - 1];
            const double coef = nu * dt / (dy * dy);
            for (int j = 0; j < n; ++j) {
                const double um = j == 0 ? south : u[j - 1];
                const double up = j == n - 1 ? north : u[j + 1];
                next[j] = u[j] + coef * (um - 2.0 * u[j] + up);
            }
            u.swap(next);
            time += dt;
        }
        double err = 0;
        for (int j = 0; j < n; ++j) err = std::max(err, std::abs(u[j] - speed * (j + 0.5) * dy));
        const bool pass = err < 1e-8;
        std::printf("couette1d  err %.3e  %s\n", err, pass ? "PASS" : "FAIL");
        if (!pass) ++failures;
    }
    std::printf("gpu  nvcc=%d  device-check=not-run  UNVERIFIED\n", KARMAN_HAS_CUDA);
    std::printf(failures ? "FAILED %d\n" : "OK\n", failures);
    return failures ? 1 : 0;
}
