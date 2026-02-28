#include "SMCKernels.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_OpenMP.H>
#include <AMReX_Gpu.H>
#include <AMReX_Reduce.H>

namespace minismc::kernels {

using namespace amrex;

namespace {
constexpr Real kRu = 8.31446261815324e+07_rt;
constexpr GpuArray<Real, 4> D8Coeffs{
    {0.8_rt, -0.2_rt, 4.0_rt / 105.0_rt, -1.0_rt / 280.0_rt}};

enum ConsComp { URHO = 0, UMX, UMY, UMZ, UEDEN, URY1 };
enum PrimComp { QRHO = 0, QU, QV, QW, QPRES, QTEMP, QEINT, QY = 7 };
constexpr int QX = QY + NSpecies;
constexpr int QH = QX + NSpecies;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real central_diff(const Array4<const Real>& arr,
                  int comp,
                  int i, int j, int k,
                  int dir,
                  const GpuArray<Real, AMREX_SPACEDIM>& dxinv)
{
    Real sum = 0.0_rt;
    for (int m = 0; m < 4; ++m) {
        const int offset = m + 1;
        const Real coeff = D8Coeffs[m];
        if (dir == 0) {
            sum += coeff * (arr(i + offset, j, k, comp) - arr(i - offset, j, k, comp));
        } else if (dir == 1) {
            sum += coeff * (arr(i, j + offset, k, comp) - arr(i, j - offset, k, comp));
        } else {
            sum += coeff * (arr(i, j, k + offset, comp) - arr(i, j, k - offset, comp));
        }
    }
    return sum * dxinv[dir];
}

template <typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real central_diff_fn(int i, int j, int k,
                     int dir,
                     const GpuArray<Real, AMREX_SPACEDIM>& dxinv,
                     const F& func)
{
    Real sum = 0.0_rt;
    for (int m = 0; m < 4; ++m) {
        const int offset = m + 1;
        const Real coeff = D8Coeffs[m];
        if (dir == 0) {
            sum += coeff * (func(i + offset, j, k) - func(i - offset, j, k));
        } else if (dir == 1) {
            sum += coeff * (func(i, j + offset, k) - func(i, j - offset, k));
        } else {
            sum += coeff * (func(i, j, k + offset) - func(i, j, k - offset));
        }
    }
    return sum * dxinv[dir];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real laplacian(const Array4<const Real>& arr,
               int comp,
               int i, int j, int k,
               const GpuArray<Real, AMREX_SPACEDIM>& dxinv)
{
    Real result = 0.0_rt;
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
        Real forward = 0.0_rt;
        Real backward = 0.0_rt;
        Real center = arr(i, j, k, comp);
        if (dir == 0) {
            forward = arr(i + 1, j, k, comp);
            backward = arr(i - 1, j, k, comp);
        } else if (dir == 1) {
            forward = arr(i, j + 1, k, comp);
            backward = arr(i, j - 1, k, comp);
        } else {
            forward = arr(i, j, k + 1, comp);
            backward = arr(i, j, k - 1, comp);
        }
        result += (forward - 2.0_rt * center + backward) * dxinv[dir] * dxinv[dir];
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real div_velocity(const Array4<const Real>& prim,
                  int i, int j, int k,
                  const GpuArray<Real, AMREX_SPACEDIM>& dxinv)
{
    Real divu = central_diff(prim, QU, i, j, k, 0, dxinv)
              + central_diff(prim, QV, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
    divu += central_diff(prim, QW, i, j, k, 2, dxinv);
#endif
    return divu;
}

} // namespace

void InitData(const Geometry& geom,
              MultiFab& state,
              const ProbParm& prob,
              const GpuArray<Real, AMREX_SPACEDIM>& dx)
{
    state.setVal(0.0);

    const GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    const GpuArray<Real, AMREX_SPACEDIM> prob_hi = geom.ProbHiArray();

    const Real kx = 2.0_rt * Math::pi<Real>() / (prob_hi[0] - prob_lo[0]);
    const Real ky = 2.0_rt * Math::pi<Real>() / (prob_hi[1] - prob_lo[1]);
#if (AMREX_SPACEDIM == 3)
    const Real kz = 2.0_rt * Math::pi<Real>() / (prob_hi[2] - prob_lo[2]);
#else
    const Real kz = 0.0_rt;
#endif

    Real Ru, Ruc, Patm;
    CKRP(Ru, Ruc, Patm);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto arr = state.array(mfi);
        const Real rfire = prob.rfire;
        const Real patm = Patm;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            GpuArray<Real, NSpecies> Xt;
            GpuArray<Real, NSpecies> Yt;

            Real x = prob_lo[0] + dx[0] * (static_cast<Real>(i) + 0.5_rt);
            Real y = prob_lo[1] + dx[1] * (static_cast<Real>(j) + 0.5_rt);
#if (AMREX_SPACEDIM == 3)
            Real z = prob_lo[2] + dx[2] * (static_cast<Real>(k) + 0.5_rt);
#else
            Real z = 0.0_rt;
#endif

            Real r = std::sqrt(x * x + y * y + z * z);

            Real Pt = patm;
            Real Tt = 300.0_rt;

            for (int n = 0; n < NSpecies; ++n) {
                Xt[n] = 0.0_rt;
            }
            Xt[0] = 0.10_rt;
            Xt[1] = 0.25_rt;

            Real expfac = std::exp(-std::pow(r / rfire, 2.0_rt));
            Pt += 0.1_rt * patm * expfac;
            Tt += 1100.0_rt * expfac;
            Xt[0] += 0.025_rt * expfac;
            Xt[1] -= 0.050_rt * expfac;
            Xt[8] = 1.0_rt - Xt[0] - Xt[1];

            CKXTY(Xt.data(), Yt.data());

            Real rho;
            CKRHOY(Pt, Tt, Yt.data(), rho);

            Real et;
            CKUBMS(Tt, Yt.data(), et);

            Real uvel = std::sin(kx * x) * std::cos(ky * y) * std::cos(kz * z) * 300.0_rt;
            Real vvel = -std::cos(kx * x) * std::sin(ky * y) * std::cos(kz * z) * 300.0_rt;
            Real wvel = 0.0_rt;

            Real kin = 0.5_rt * (uvel * uvel + vvel * vvel + wvel * wvel);

            arr(i, j, k, URHO) = rho;
            arr(i, j, k, UMX) = rho * uvel;
            arr(i, j, k, UMY) = rho * vvel;
            arr(i, j, k, UMZ) = rho * wvel;
            arr(i, j, k, UEDEN) = rho * (et + kin);

            for (int n = 0; n < NSpecies; ++n) {
                arr(i, j, k, URY1 + n) = rho * Yt[n];
            }
        });
    }
}

void ComputePrimitives(const Geometry& geom,
                       const MultiFab& state,
                       MultiFab& prim)
{
    const auto dxinv = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(prim, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(StencilNG);
        auto q = prim.array(mfi);
        auto u = state.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            GpuArray<Real, NSpecies> Y;
            GpuArray<Real, NSpecies> X;
            GpuArray<Real, NSpecies> h;

            Real rho = u(i, j, k, URHO);
            Real rhoinv = 1.0_rt / rho;

            Real uvel = u(i, j, k, UMX) * rhoinv;
            Real vvel = u(i, j, k, UMY) * rhoinv;
            Real wvel = u(i, j, k, UMZ) * rhoinv;

            q(i, j, k, QRHO) = rho;
            q(i, j, k, QU) = uvel;
            q(i, j, k, QV) = vvel;
            q(i, j, k, QW) = wvel;

            Real kin = 0.5_rt * (uvel * uvel + vvel * vvel + wvel * wvel);
            Real eint = rhoinv * u(i, j, k, UEDEN) - kin;
            q(i, j, k, QEINT) = eint;

            for (int n = 0; n < NSpecies; ++n) {
                Real rhoy = amrex::max(0.0_rt, u(i, j, k, URY1 + n));
                Y[n] = rhoy * rhoinv;
                q(i, j, k, QY + n) = Y[n];
            }

            CKYTX(Y.data(), X.data());
            for (int n = 0; n < NSpecies; ++n) {
                q(i, j, k, QX + n) = X[n];
            }

            Real temp = 300.0_rt;
            int ierr = 0;
            GET_T_GIVEN_EY(eint, Y.data(), temp, ierr);
            q(i, j, k, QTEMP) = temp;

            Real pres;
            CKPY(rho, temp, Y.data(), pres);
            q(i, j, k, QPRES) = pres;

            CKHMS(temp, h.data());
            for (int n = 0; n < NSpecies; ++n) {
                q(i, j, k, QH + n) = h[n];
            }
        });
    }
}

void ResetDensity(MultiFab& state)
{
    const IntVect ng = state.nGrowVect();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(state, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(ng);
        auto arr = state.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real rho = 0.0_rt;
            GpuArray<Real, NSpecies> tmp;
            for (int n = 0; n < NSpecies; ++n) {
                Real val = arr(i, j, k, URY1 + n);
                if (val < 0.0_rt) {
                    val = 0.0_rt;
                }
                tmp[n] = val;
                rho += val;
            }
            arr(i, j, k, URHO) = rho;
            for (int n = 0; n < NSpecies; ++n) {
                arr(i, j, k, URY1 + n) = tmp[n];
            }
        });
    }
}

void AddChemistry(MultiFab& rhs,
                  const MultiFab& prim)
{
    GpuArray<Real, NSpecies> mw;
    CKWT(mw.data());

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto rp = rhs.array(mfi);
        auto qp = prim.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            GpuArray<Real, NSpecies> Y;
            for (int n = 0; n < NSpecies; ++n) {
                Y[n] = qp(i, j, k, QY + n);
            }
            Real T = qp(i, j, k, QTEMP);
            Real P = qp(i, j, k, QPRES);
            GpuArray<Real, NSpecies> wdot;
            CKWYP(P, T, Y.data(), wdot.data());
            for (int n = 0; n < NSpecies; ++n) {
                rp(i, j, k, URY1 + n) += wdot[n] * mw[n];
            }
        });
    }
}

void ComputeTransport(const MultiFab& prim,
                      MultiFab& mu,
                      MultiFab& xi,
                      MultiFab& lam,
                      MultiFab& Ddiag)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(prim, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.growntilebox(StencilNG);
        auto qp = prim.const_array(mfi);
        auto mup = mu.array(mfi);
        auto xip = xi.array(mfi);
        auto lamp = lam.array(mfi);
        auto dp = Ddiag.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            GpuArray<Real, NSpecies> Y;
            for (int n = 0; n < NSpecies; ++n) {
                Y[n] = qp(i, j, k, QY + n);
            }
            Real T = qp(i, j, k, QTEMP);
            Real mu_val = 1.458e-5_rt * std::sqrt(T) * T / (T + 110.4_rt);
            mup(i, j, k) = mu_val;
            xip(i, j, k) = 0.0_rt;

            Real Wbar;
            CKMMWY(Y.data(), Wbar);
            Real cp;
            CKCPBS(T, Y.data(), cp);
            lamp(i, j, k) = mu_val * cp / 0.72_rt;
            Real invW = 1.0_rt / Wbar;
            for (int n = 0; n < NSpecies; ++n) {
                dp(i, j, k, n) = mu_val * invW / 0.72_rt;
            }
        });
    }
}

void AddHyperbolic(const Geometry& geom,
                   const MultiFab& state,
                   const MultiFab& prim,
                   MultiFab& rhs)
{
    const auto dxinv = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto ru = rhs.array(mfi);
        auto u = state.const_array(mfi);
        auto q = prim.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real drho = central_diff(u, UMX, i, j, k, 0, dxinv)
                      + central_diff(u, UMY, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            drho += central_diff(u, UMZ, i, j, k, 2, dxinv);
#endif
            ru(i, j, k, URHO) -= drho;

            auto fx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real uvel = u(ii, jj, kk, UMX) / rho;
                Real pres = q(ii, jj, kk, QPRES);
                return u(ii, jj, kk, UMX) * uvel + pres;
            };
            auto fy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real vvel = u(ii, jj, kk, UMY) / rho;
                return u(ii, jj, kk, UMX) * vvel;
            };
            auto fz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real wvel = u(ii, jj, kk, UMZ) / rho;
                return u(ii, jj, kk, UMX) * wvel;
            };

            Real dmx = central_diff_fn(i, j, k, 0, dxinv, fx)
                     + central_diff_fn(i, j, k, 1, dxinv, fy);
#if (AMREX_SPACEDIM == 3)
            dmx += central_diff_fn(i, j, k, 2, dxinv, fz);
#endif
            ru(i, j, k, UMX) -= dmx;

            auto gx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real uvel = u(ii, jj, kk, UMX) / rho;
                return u(ii, jj, kk, UMY) * uvel;
            };
            auto gy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real vvel = u(ii, jj, kk, UMY) / rho;
                Real pres = q(ii, jj, kk, QPRES);
                return u(ii, jj, kk, UMY) * vvel + pres;
            };
            auto gz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real wvel = u(ii, jj, kk, UMZ) / rho;
                return u(ii, jj, kk, UMY) * wvel;
            };

            Real dmy = central_diff_fn(i, j, k, 0, dxinv, gx)
                     + central_diff_fn(i, j, k, 1, dxinv, gy);
#if (AMREX_SPACEDIM == 3)
            dmy += central_diff_fn(i, j, k, 2, dxinv, gz);
#endif
            ru(i, j, k, UMY) -= dmy;

#if (AMREX_SPACEDIM == 3)
            auto hx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real uvel = u(ii, jj, kk, UMX) / rho;
                return u(ii, jj, kk, UMZ) * uvel;
            };
            auto hy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real vvel = u(ii, jj, kk, UMY) / rho;
                return u(ii, jj, kk, UMZ) * vvel;
            };
            auto hz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real rho = u(ii, jj, kk, URHO);
                Real wvel = u(ii, jj, kk, UMZ) / rho;
                Real pres = q(ii, jj, kk, QPRES);
                return u(ii, jj, kk, UMZ) * wvel + pres;
            };

            Real dmz = central_diff_fn(i, j, k, 0, dxinv, hx)
                     + central_diff_fn(i, j, k, 1, dxinv, hy)
                     + central_diff_fn(i, j, k, 2, dxinv, hz);
            ru(i, j, k, UMZ) -= dmz;
#endif

            auto ex = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real vel = u(ii, jj, kk, UMX) / u(ii, jj, kk, URHO);
                return (u(ii, jj, kk, UEDEN) + q(ii, jj, kk, QPRES)) * vel;
            };
            auto ey = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real vel = u(ii, jj, kk, UMY) / u(ii, jj, kk, URHO);
                return (u(ii, jj, kk, UEDEN) + q(ii, jj, kk, QPRES)) * vel;
            };
#if (AMREX_SPACEDIM == 3)
            auto ez = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real vel = u(ii, jj, kk, UMZ) / u(ii, jj, kk, URHO);
                return (u(ii, jj, kk, UEDEN) + q(ii, jj, kk, QPRES)) * vel;
            };
#endif

            Real dE = central_diff_fn(i, j, k, 0, dxinv, ex)
                     + central_diff_fn(i, j, k, 1, dxinv, ey);
#if (AMREX_SPACEDIM == 3)
            dE += central_diff_fn(i, j, k, 2, dxinv, ez);
#endif
            ru(i, j, k, UEDEN) -= dE;

            for (int n = 0; n < NSpecies; ++n) {
                auto sx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real vel = u(ii, jj, kk, UMX) / u(ii, jj, kk, URHO);
                    return vel * u(ii, jj, kk, URY1 + n);
                };
                auto sy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real vel = u(ii, jj, kk, UMY) / u(ii, jj, kk, URHO);
                    return vel * u(ii, jj, kk, URY1 + n);
                };
#if (AMREX_SPACEDIM == 3)
                auto sz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real vel = u(ii, jj, kk, UMZ) / u(ii, jj, kk, URHO);
                    return vel * u(ii, jj, kk, URY1 + n);
                };
#endif
                Real dY = central_diff_fn(i, j, k, 0, dxinv, sx)
                        + central_diff_fn(i, j, k, 1, dxinv, sy);
#if (AMREX_SPACEDIM == 3)
                dY += central_diff_fn(i, j, k, 2, dxinv, sz);
#endif
                ru(i, j, k, URY1 + n) -= dY;
            }
        });
    }
}

void AddDiffusive(const Geometry& geom,
                  const MultiFab& prim,
                  const MultiFab& mu,
                  const MultiFab& xi,
                  const MultiFab& lam,
                  const MultiFab& Ddiag,
                  MultiFab& rhs)
{
    const auto dxinv = geom.InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto rp = rhs.array(mfi);
        auto qp = prim.const_array(mfi);
        auto mup = mu.const_array(mfi);
        auto xip = xi.const_array(mfi);
        auto lamp = lam.const_array(mfi);
        auto dp = Ddiag.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real mu_loc = mup(i, j, k);
            Real xi_loc = xip(i, j, k);
            Real divu = div_velocity(qp, i, j, k, dxinv);

            Real dudx = central_diff(qp, QU, i, j, k, 0, dxinv);
            Real dudy = central_diff(qp, QU, i, j, k, 1, dxinv);
            Real dvdx = central_diff(qp, QV, i, j, k, 0, dxinv);
            Real dvdy = central_diff(qp, QV, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            Real dwdx = central_diff(qp, QW, i, j, k, 0, dxinv);
            Real dwdy = central_diff(qp, QW, i, j, k, 1, dxinv);
            Real dudz = central_diff(qp, QU, i, j, k, 2, dxinv);
            Real dvdz = central_diff(qp, QV, i, j, k, 2, dxinv);
            Real dwdz = central_diff(qp, QW, i, j, k, 2, dxinv);
#endif

            Real bulk = xi_loc - (2.0_rt / 3.0_rt) * mu_loc;

            auto tau_xx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real bulkv = xip(ii, jj, kk) - (2.0_rt / 3.0_rt) * mup(ii, jj, kk);
                Real divuv = div_velocity(qp, ii, jj, kk, dxinv);
                Real dudxv = central_diff(qp, QU, ii, jj, kk, 0, dxinv);
                return 2.0_rt * muv * dudxv + bulkv * divuv;
            };
            auto tau_xy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dudyv = central_diff(qp, QU, ii, jj, kk, 1, dxinv);
                Real dvdxv = central_diff(qp, QV, ii, jj, kk, 0, dxinv);
                return muv * (dudyv + dvdxv);
            };
#if (AMREX_SPACEDIM == 3)
            auto tau_xz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dudzv = central_diff(qp, QU, ii, jj, kk, 2, dxinv);
                Real dwdxv = central_diff(qp, QW, ii, jj, kk, 0, dxinv);
                return muv * (dudzv + dwdxv);
            };
            auto tau_yy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real bulkv = xip(ii, jj, kk) - (2.0_rt / 3.0_rt) * mup(ii, jj, kk);
                Real divuv = div_velocity(qp, ii, jj, kk, dxinv);
                Real dvdvv = central_diff(qp, QV, ii, jj, kk, 1, dxinv);
                return 2.0_rt * muv * dvdvv + bulkv * divuv;
            };
            auto tau_yz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dvdzv = central_diff(qp, QV, ii, jj, kk, 2, dxinv);
                Real dwdyv = central_diff(qp, QW, ii, jj, kk, 1, dxinv);
                return muv * (dvdzv + dwdyv);
            };
            auto tau_zz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real bulkv = xip(ii, jj, kk) - (2.0_rt / 3.0_rt) * mup(ii, jj, kk);
                Real divuv = div_velocity(qp, ii, jj, kk, dxinv);
                Real dwdzv = central_diff(qp, QW, ii, jj, kk, 2, dxinv);
                return 2.0_rt * muv * dwdzv + bulkv * divuv;
            };
#endif

            Real visc_x = central_diff_fn(i, j, k, 0, dxinv, tau_xx)
                         + central_diff_fn(i, j, k, 1, dxinv, tau_xy);
#if (AMREX_SPACEDIM == 3)
            visc_x += central_diff_fn(i, j, k, 2, dxinv, tau_xz);
#endif
            rp(i, j, k, UMX) += visc_x;

            auto tau_yx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dvdxv = central_diff(qp, QV, ii, jj, kk, 0, dxinv);
                Real dudyv = central_diff(qp, QU, ii, jj, kk, 1, dxinv);
                return muv * (dvdxv + dudyv);
            };
            auto tau_yy2 = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real bulkv = xip(ii, jj, kk) - (2.0_rt / 3.0_rt) * mup(ii, jj, kk);
                Real divuv = div_velocity(qp, ii, jj, kk, dxinv);
                Real dvdyv = central_diff(qp, QV, ii, jj, kk, 1, dxinv);
                return 2.0_rt * muv * dvdyv + bulkv * divuv;
            };
#if (AMREX_SPACEDIM == 3)
            auto tau_yz2 = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dvdzv = central_diff(qp, QV, ii, jj, kk, 2, dxinv);
                Real dwdyv = central_diff(qp, QW, ii, jj, kk, 1, dxinv);
                return muv * (dvdzv + dwdyv);
            };
#endif
            Real visc_y = central_diff_fn(i, j, k, 0, dxinv, tau_yx)
                         + central_diff_fn(i, j, k, 1, dxinv, tau_yy2);
#if (AMREX_SPACEDIM == 3)
            visc_y += central_diff_fn(i, j, k, 2, dxinv, tau_yz2);
#endif
            rp(i, j, k, UMY) += visc_y;

#if (AMREX_SPACEDIM == 3)
            auto tau_zx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dwdxv = central_diff(qp, QW, ii, jj, kk, 0, dxinv);
                Real dudzv = central_diff(qp, QU, ii, jj, kk, 2, dxinv);
                return muv * (dwdxv + dudzv);
            };
            auto tau_zy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real muv = mup(ii, jj, kk);
                Real dwdyv = central_diff(qp, QW, ii, jj, kk, 1, dxinv);
                Real dvdzv = central_diff(qp, QV, ii, jj, kk, 2, dxinv);
                return muv * (dwdyv + dvdzv);
            };
            Real visc_z = central_diff_fn(i, j, k, 0, dxinv, tau_zx)
                         + central_diff_fn(i, j, k, 1, dxinv, tau_zy)
                         + central_diff_fn(i, j, k, 2, dxinv, tau_zz);
            rp(i, j, k, UMZ) += visc_z;
#endif

            Real gradTx = central_diff(qp, QTEMP, i, j, k, 0, dxinv);
            Real gradTy = central_diff(qp, QTEMP, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            Real gradTz = central_diff(qp, QTEMP, i, j, k, 2, dxinv);
#endif
            auto qx = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real lamv = lamp(ii, jj, kk);
                Real grad = central_diff(qp, QTEMP, ii, jj, kk, 0, dxinv);
                return -lamv * grad;
            };
            auto qy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real lamv = lamp(ii, jj, kk);
                Real grad = central_diff(qp, QTEMP, ii, jj, kk, 1, dxinv);
                return -lamv * grad;
            };
#if (AMREX_SPACEDIM == 3)
            auto qz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                Real lamv = lamp(ii, jj, kk);
                Real grad = central_diff(qp, QTEMP, ii, jj, kk, 2, dxinv);
                return -lamv * grad;
            };
#endif
            Real cond = central_diff_fn(i, j, k, 0, dxinv, qx)
                       + central_diff_fn(i, j, k, 1, dxinv, qy);
#if (AMREX_SPACEDIM == 3)
            cond += central_diff_fn(i, j, k, 2, dxinv, qz);
#endif

            Real species_heat = 0.0_rt;
            for (int n = 0; n < NSpecies; ++n) {
                auto flux = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real rho = qp(ii, jj, kk, QRHO);
                    Real grad = central_diff(qp, QY + n, ii, jj, kk, 0, dxinv);
                    return -rho * dp(ii, jj, kk, n) * grad;
                };
                Real div_flux = central_diff_fn(i, j, k, 0, dxinv, flux);
                auto fluxy = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real rho = qp(ii, jj, kk, QRHO);
                    Real grad = central_diff(qp, QY + n, ii, jj, kk, 1, dxinv);
                    return -rho * dp(ii, jj, kk, n) * grad;
                };
                div_flux += central_diff_fn(i, j, k, 1, dxinv, fluxy);
#if (AMREX_SPACEDIM == 3)
                auto fluxz = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real rho = qp(ii, jj, kk, QRHO);
                    Real grad = central_diff(qp, QY + n, ii, jj, kk, 2, dxinv);
                    return -rho * dp(ii, jj, kk, n) * grad;
                };
                div_flux += central_diff_fn(i, j, k, 2, dxinv, fluxz);
#endif
                rp(i, j, k, URY1 + n) += div_flux;
                species_heat += div_flux * qp(i, j, k, QH + n);
            }

            rp(i, j, k, UEDEN) += cond + species_heat;
        });
    }
}

void ComputeCourant(const Geometry& geom,
                    const MultiFab& prim,
                    Real& courno)
{
    const auto dxinv = geom.InvCellSizeArray();

    ReduceOps<ReduceOpMax> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(prim, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto qp = prim.const_array(mfi);

        reduce_op.eval(bx, reduce_data,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept -> ReduceTuple {
                GpuArray<Real, NSpecies> X;
                for (int n = 0; n < NSpecies; ++n) {
                    X[n] = qp(i, j, k, QX + n);
                }
                Real Cv;
                CKCVBL(qp(i, j, k, QTEMP), X.data(), Cv);
                Real Cp = Cv + kRu;
                Real gamma = Cp / Cv;
                Real c = std::sqrt(gamma * qp(i, j, k, QPRES) / qp(i, j, k, QRHO));
                Real cour = (c + Math::abs(qp(i, j, k, QU))) * dxinv[0];
                cour = amrex::max(cour, (c + Math::abs(qp(i, j, k, QV))) * dxinv[1]);
#if (AMREX_SPACEDIM == 3)
                cour = amrex::max(cour, (c + Math::abs(qp(i, j, k, QW))) * dxinv[2]);
#endif
                return {cour};
            });
    }

    ReduceTuple result = reduce_data.value(reduce_op);
    Real local_max = amrex::get<0>(result);
    ParallelDescriptor::ReduceRealMax(local_max);
    courno = local_max;
}

} // namespace minismc::kernels
