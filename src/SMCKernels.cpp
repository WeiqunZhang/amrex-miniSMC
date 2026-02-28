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

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real d8_coeff(int idx) noexcept
{
    constexpr Real coeffs[4] = {
        0.8_rt, -0.2_rt, 4.0_rt / 105.0_rt, -1.0_rt / 280.0_rt};
    return coeffs[idx];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_interp_coeff(int idx) noexcept
{
    constexpr Real coeffs[8] = {
        -5.0_rt / 2048.0_rt,
        49.0_rt / 2048.0_rt,
        -245.0_rt / 2048.0_rt,
        1225.0_rt / 2048.0_rt,
        1225.0_rt / 2048.0_rt,
        -245.0_rt / 2048.0_rt,
        49.0_rt / 2048.0_rt,
        -5.0_rt / 2048.0_rt};
    return coeffs[idx];
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real face_deriv_coeff(int idx) noexcept
{
    constexpr Real coeffs[8] = {
        6.97544642857694e-04_rt,
        -9.57031250000056e-03_rt,
        7.975260416666667e-02_rt,
        -1.1962890625_rt,
        1.1962890625_rt,
        -7.975260416666667e-02_rt,
        9.5703125e-03_rt,
        -6.975446428571429e-04_rt};
    return coeffs[idx];
}

enum VelDerComp {
    DUDX = 0,
    DUDY,
    DUDZ,
    DVDX,
    DVDY,
    DVDZ,
    DWDX,
    DWDY,
    DWDZ
};

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void apply_offset(int dir, int offset, int& i, int& j, int& k) noexcept
{
    if (dir == 0) {
        i += offset;
    } else if (dir == 1) {
        j += offset;
#if (AMREX_SPACEDIM == 3)
    } else {
        k += offset;
#endif
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real face_interp(const Array4<const Real>& arr,
                 int comp,
                 int dir,
                 int i, int j, int k) noexcept
{
    Real sum = 0.0_rt;
    for (int m = 0; m < 8; ++m) {
        int offset = m - 3;
        int ii = i;
        int jj = j;
        int kk = k;
        apply_offset(dir, offset, ii, jj, kk);
        sum += face_interp_coeff(m) * arr(ii, jj, kk, comp);
    }
    return sum;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real face_deriv(const Array4<const Real>& arr,
                int comp,
                int dir,
                int i, int j, int k,
                Real dxinv) noexcept
{
    Real sum = 0.0_rt;
    for (int m = 0; m < 8; ++m) {
        int offset = m - 3;
        int ii = i;
        int jj = j;
        int kk = k;
        apply_offset(dir, offset, ii, jj, kk);
        sum += face_deriv_coeff(m) * arr(ii, jj, kk, comp);
    }
    return sum * dxinv;
}

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
        const Real coeff = d8_coeff(m);
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
        const Real coeff = d8_coeff(m);
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

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_tau_face_x(int i, int j, int k,
                        const Array4<const Real>& qp,
                        const Array4<const Real>& mu,
                        const Array4<const Real>& xi,
                        const Array4<const Real>& grad,
                        const GpuArray<Real, AMREX_SPACEDIM>& dxinv,
                        Real& tau_xx, Real& tau_xy, Real& tau_xz) noexcept
{
    Real mu_face = face_interp(mu, 0, 0, i, j, k);
    Real xi_face = face_interp(xi, 0, 0, i, j, k);
    Real dudx = face_deriv(qp, QU, 0, i, j, k, dxinv[0]);
    Real dvdy = face_interp(grad, DVDY, 0, i, j, k);
#if (AMREX_SPACEDIM == 3)
    Real dwdz = face_interp(grad, DWDZ, 0, i, j, k);
#else
    Real dwdz = 0.0_rt;
#endif
    Real divu = dudx + dvdy + dwdz;
    Real bulk = xi_face - (2.0_rt / 3.0_rt) * mu_face;
    tau_xx = 2.0_rt * mu_face * dudx + bulk * divu;

    Real dvdx = face_deriv(qp, QV, 0, i, j, k, dxinv[0]);
    Real dudy = face_interp(grad, DUDY, 0, i, j, k);
    tau_xy = mu_face * (dudy + dvdx);

#if (AMREX_SPACEDIM == 3)
    Real dwdx = face_deriv(qp, QW, 0, i, j, k, dxinv[0]);
    Real dudz = face_interp(grad, DUDZ, 0, i, j, k);
    tau_xz = mu_face * (dudz + dwdx);
#else
    tau_xz = 0.0_rt;
#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_tau_face_y(int i, int j, int k,
                        const Array4<const Real>& qp,
                        const Array4<const Real>& mu,
                        const Array4<const Real>& xi,
                        const Array4<const Real>& grad,
                        const GpuArray<Real, AMREX_SPACEDIM>& dxinv,
                        Real& tau_yx, Real& tau_yy, Real& tau_yz) noexcept
{
    Real mu_face = face_interp(mu, 0, 1, i, j, k);
    Real xi_face = face_interp(xi, 0, 1, i, j, k);
    Real dvdy = face_deriv(qp, QV, 1, i, j, k, dxinv[1]);
    Real dudx = face_interp(grad, DUDX, 1, i, j, k);
#if (AMREX_SPACEDIM == 3)
    Real dwdz = face_interp(grad, DWDZ, 1, i, j, k);
#else
    Real dwdz = 0.0_rt;
#endif
    Real divu = dudx + dvdy + dwdz;
    Real bulk = xi_face - (2.0_rt / 3.0_rt) * mu_face;
    tau_yy = 2.0_rt * mu_face * dvdy + bulk * divu;

    Real dvdx = face_interp(grad, DVDX, 1, i, j, k);
    Real dudy = face_deriv(qp, QU, 1, i, j, k, dxinv[1]);
    tau_yx = mu_face * (dvdx + dudy);

#if (AMREX_SPACEDIM == 3)
    Real dvdz = face_interp(grad, DVDZ, 1, i, j, k);
    Real dwdy = face_deriv(qp, QW, 1, i, j, k, dxinv[1]);
    tau_yz = mu_face * (dvdz + dwdy);
#else
    tau_yz = 0.0_rt;
#endif
}

#if (AMREX_SPACEDIM == 3)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void compute_tau_face_z(int i, int j, int k,
                        const Array4<const Real>& qp,
                        const Array4<const Real>& mu,
                        const Array4<const Real>& xi,
                        const Array4<const Real>& grad,
                        const GpuArray<Real, AMREX_SPACEDIM>& dxinv,
                        Real& tau_zx, Real& tau_zy, Real& tau_zz) noexcept
{
    Real mu_face = face_interp(mu, 0, 2, i, j, k);
    Real xi_face = face_interp(xi, 0, 2, i, j, k);
    Real dwdz = face_deriv(qp, QW, 2, i, j, k, dxinv[2]);
    Real dudx = face_interp(grad, DUDX, 2, i, j, k);
    Real dvdy = face_interp(grad, DVDY, 2, i, j, k);
    Real divu = dudx + dvdy + dwdz;
    Real bulk = xi_face - (2.0_rt / 3.0_rt) * mu_face;
    tau_zz = 2.0_rt * mu_face * dwdz + bulk * divu;

    Real dwdx = face_interp(grad, DWDX, 2, i, j, k);
    Real dudz = face_deriv(qp, QU, 2, i, j, k, dxinv[2]);
    tau_zx = mu_face * (dwdx + dudz);

    Real dwdy = face_interp(grad, DWDY, 2, i, j, k);
    Real dvdz = face_deriv(qp, QV, 2, i, j, k, dxinv[2]);
    tau_zy = mu_face * (dwdy + dvdz);
}
#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real compute_heat_flux_face(int dir,
                            int i, int j, int k,
                            const Array4<const Real>& qp,
                            const Array4<const Real>& lam,
                            const GpuArray<Real, AMREX_SPACEDIM>& dxinv) noexcept
{
    Real lam_face = face_interp(lam, 0, dir, i, j, k);
    Real gradT = face_deriv(qp, QTEMP, dir, i, j, k, dxinv[dir]);
    return -lam_face * gradT;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
Real compute_species_flux_face(int dir,
                               int i, int j, int k,
                               int n,
                               const Array4<const Real>& qp,
                               const Array4<const Real>& dp,
                               const GpuArray<Real, AMREX_SPACEDIM>& dxinv) noexcept
{
    Real rho_face = face_interp(qp, QRHO, dir, i, j, k);
    Real diff_face = face_interp(dp, n, dir, i, j, k);
    Real gradY = face_deriv(qp, QY + n, dir, i, j, k, dxinv[dir]);
    return -rho_face * diff_face * gradY;
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
        const Box& gbx = mfi.growntilebox(StencilNG);
        auto rp = rhs.array(mfi);
        auto qp = prim.const_array(mfi);
        auto mup = mu.const_array(mfi);
        auto xip = xi.const_array(mfi);
        auto lamp = lam.const_array(mfi);
        auto dp = Ddiag.const_array(mfi);

        FArrayBox grad_fab(gbx, 9);
        auto grad = grad_fab.array();
        Elixir grad_eli = grad_fab.elixir();

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            grad(i, j, k, DUDX) = central_diff(qp, QU, i, j, k, 0, dxinv);
            grad(i, j, k, DUDY) = central_diff(qp, QU, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            grad(i, j, k, DUDZ) = central_diff(qp, QU, i, j, k, 2, dxinv);
#else
            grad(i, j, k, DUDZ) = 0.0_rt;
#endif
            grad(i, j, k, DVDX) = central_diff(qp, QV, i, j, k, 0, dxinv);
            grad(i, j, k, DVDY) = central_diff(qp, QV, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            grad(i, j, k, DVDZ) = central_diff(qp, QV, i, j, k, 2, dxinv);
#else
            grad(i, j, k, DVDZ) = 0.0_rt;
#endif
            grad(i, j, k, DWDX) = central_diff(qp, QW, i, j, k, 0, dxinv);
            grad(i, j, k, DWDY) = central_diff(qp, QW, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            grad(i, j, k, DWDZ) = central_diff(qp, QW, i, j, k, 2, dxinv);
#else
            grad(i, j, k, DWDZ) = 0.0_rt;
#endif
        });

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real tau_xx_hi, tau_xy_hi, tau_xz_hi;
            Real tau_xx_lo, tau_xy_lo, tau_xz_lo;
            compute_tau_face_x(i, j, k, qp, mup, xip, grad, dxinv,
                               tau_xx_hi, tau_xy_hi, tau_xz_hi);
            compute_tau_face_x(i - 1, j, k, qp, mup, xip, grad, dxinv,
                               tau_xx_lo, tau_xy_lo, tau_xz_lo);

            Real tau_yx_hi, tau_yy_hi, tau_yz_hi;
            Real tau_yx_lo, tau_yy_lo, tau_yz_lo;
            compute_tau_face_y(i, j, k, qp, mup, xip, grad, dxinv,
                               tau_yx_hi, tau_yy_hi, tau_yz_hi);
            compute_tau_face_y(i, j - 1, k, qp, mup, xip, grad, dxinv,
                               tau_yx_lo, tau_yy_lo, tau_yz_lo);

#if (AMREX_SPACEDIM == 3)
            Real tau_zx_hi, tau_zy_hi, tau_zz_hi;
            Real tau_zx_lo, tau_zy_lo, tau_zz_lo;
            compute_tau_face_z(i, j, k, qp, mup, xip, grad, dxinv,
                               tau_zx_hi, tau_zy_hi, tau_zz_hi);
            compute_tau_face_z(i, j, k - 1, qp, mup, xip, grad, dxinv,
                               tau_zx_lo, tau_zy_lo, tau_zz_lo);
#else
            Real tau_zx_hi = 0.0_rt, tau_zx_lo = 0.0_rt;
            Real tau_zy_hi = 0.0_rt, tau_zy_lo = 0.0_rt;
            Real tau_zz_hi = 0.0_rt, tau_zz_lo = 0.0_rt;
#endif

            Real visc_x = (tau_xx_hi - tau_xx_lo) * dxinv[0]
                        + (tau_yx_hi - tau_yx_lo) * dxinv[1];
#if (AMREX_SPACEDIM == 3)
            visc_x += (tau_zx_hi - tau_zx_lo) * dxinv[2];
#endif
            rp(i, j, k, UMX) += visc_x;

            Real visc_y = (tau_xy_hi - tau_xy_lo) * dxinv[0]
                        + (tau_yy_hi - tau_yy_lo) * dxinv[1];
#if (AMREX_SPACEDIM == 3)
            visc_y += (tau_zy_hi - tau_zy_lo) * dxinv[2];
#endif
            rp(i, j, k, UMY) += visc_y;

#if (AMREX_SPACEDIM == 3)
            Real visc_z = (tau_xz_hi - tau_xz_lo) * dxinv[0]
                        + (tau_yz_hi - tau_yz_lo) * dxinv[1]
                        + (tau_zz_hi - tau_zz_lo) * dxinv[2];
            rp(i, j, k, UMZ) += visc_z;
#endif

            Real heat_x_hi = compute_heat_flux_face(0, i, j, k, qp, lamp, dxinv);
            Real heat_x_lo = compute_heat_flux_face(0, i - 1, j, k, qp, lamp, dxinv);
            Real heat_y_hi = compute_heat_flux_face(1, i, j, k, qp, lamp, dxinv);
            Real heat_y_lo = compute_heat_flux_face(1, i, j - 1, k, qp, lamp, dxinv);
#if (AMREX_SPACEDIM == 3)
            Real heat_z_hi = compute_heat_flux_face(2, i, j, k, qp, lamp, dxinv);
            Real heat_z_lo = compute_heat_flux_face(2, i, j, k - 1, qp, lamp, dxinv);
#else
            Real heat_z_hi = 0.0_rt;
            Real heat_z_lo = 0.0_rt;
#endif

            Real energy_flux_x_hi = heat_x_hi;
            Real energy_flux_x_lo = heat_x_lo;
            Real energy_flux_y_hi = heat_y_hi;
            Real energy_flux_y_lo = heat_y_lo;
            Real energy_flux_z_hi = heat_z_hi;
            Real energy_flux_z_lo = heat_z_lo;

            for (int n = 0; n < NSpecies; ++n) {
                Real flux_x_hi = compute_species_flux_face(0, i, j, k, n, qp, dp, dxinv);
                Real flux_x_lo = compute_species_flux_face(0, i - 1, j, k, n, qp, dp, dxinv);
                Real flux_y_hi = compute_species_flux_face(1, i, j, k, n, qp, dp, dxinv);
                Real flux_y_lo = compute_species_flux_face(1, i, j - 1, k, n, qp, dp, dxinv);
#if (AMREX_SPACEDIM == 3)
                Real flux_z_hi = compute_species_flux_face(2, i, j, k, n, qp, dp, dxinv);
                Real flux_z_lo = compute_species_flux_face(2, i, j, k - 1, n, qp, dp, dxinv);
#else
                Real flux_z_hi = 0.0_rt;
                Real flux_z_lo = 0.0_rt;
#endif

                Real div_flux = (flux_x_hi - flux_x_lo) * dxinv[0]
                              + (flux_y_hi - flux_y_lo) * dxinv[1];
#if (AMREX_SPACEDIM == 3)
                div_flux += (flux_z_hi - flux_z_lo) * dxinv[2];
#endif
                rp(i, j, k, URY1 + n) += div_flux;

                Real h_x_hi = face_interp(qp, QH + n, 0, i, j, k);
                Real h_x_lo = face_interp(qp, QH + n, 0, i - 1, j, k);
                Real h_y_hi = face_interp(qp, QH + n, 1, i, j, k);
                Real h_y_lo = face_interp(qp, QH + n, 1, i, j - 1, k);
#if (AMREX_SPACEDIM == 3)
                Real h_z_hi = face_interp(qp, QH + n, 2, i, j, k);
                Real h_z_lo = face_interp(qp, QH + n, 2, i, j, k - 1);
#else
                Real h_z_hi = 0.0_rt;
                Real h_z_lo = 0.0_rt;
#endif

                energy_flux_x_hi += h_x_hi * flux_x_hi;
                energy_flux_x_lo += h_x_lo * flux_x_lo;
                energy_flux_y_hi += h_y_hi * flux_y_hi;
                energy_flux_y_lo += h_y_lo * flux_y_lo;
#if (AMREX_SPACEDIM == 3)
                energy_flux_z_hi += h_z_hi * flux_z_hi;
                energy_flux_z_lo += h_z_lo * flux_z_lo;
#endif
            }

            Real energy_div = (energy_flux_x_hi - energy_flux_x_lo) * dxinv[0]
                            + (energy_flux_y_hi - energy_flux_y_lo) * dxinv[1];
#if (AMREX_SPACEDIM == 3)
            energy_div += (energy_flux_z_hi - energy_flux_z_lo) * dxinv[2];
#endif
            rp(i, j, k, UEDEN) += energy_div;
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
