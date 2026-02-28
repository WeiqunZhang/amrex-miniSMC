#include "SMCKernels.H"

#include <AMReX_Arena.H>
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

void add_diffusive_part1(const Geometry& geom,
                         const MultiFab& prim,
                         const MultiFab& mu,
                         const MultiFab& xi,
                         const MultiFab& lam,
                         const MultiFab& Ddiag,
                         MultiFab& rhs);

void add_diffusive_part2(const Geometry& geom,
                         const MultiFab& prim,
                         const MultiFab& mu,
                         const MultiFab& xi,
                         const MultiFab& lam,
                         const MultiFab& Ddiag,
                         MultiFab& rhs);

constexpr Real kRu = 8.31446261815324e+07_rt;
constexpr Real kPrandtl = 0.72_rt;
constexpr Real kSchmidt = 0.72_rt;
constexpr Real kInvPrandtl = 1.0_rt / kPrandtl;
constexpr Real kInvSchmidt = 1.0_rt / kSchmidt;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real d8_coeff(int idx) noexcept
{
    constexpr Real coeffs[4] = {
        0.8_rt, -0.2_rt, 4.0_rt / 105.0_rt, -1.0_rt / 280.0_rt};
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
constexpr int kNumVelDeriv = 9;

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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

AMREX_GPU_MANAGED Real kM8[8][8] = {
    {-0.0028316326530612246_rt, 0.010408163265306122_rt, -0.0078571428571428577_rt, -0.00061224489795918364_rt, 0.00089285714285714283_rt, 0.0_rt, 0.0_rt, 0.0_rt},
    {0.015782312925170069_rt, -0.05859126984126984_rt, 0.045646258503401357_rt, -0.02119047619047619_rt, 0.026371882086167801_rt, -0.0080187074829931974_rt, 0.0_rt, 0.0_rt},
    {-0.0035714285714285713_rt, -0.049931972789115646_rt, 0.32069444444444445_rt, -0.36292517006802721_rt, 0.1680952380952381_rt, -0.093628117913832201_rt, 0.021267006802721089_rt, 0.0_rt},
    {-0.025306122448979593_rt, 0.17642857142857143_rt, -0.44993197278911562_rt, -0.22073412698412698_rt, 0.6227891156462585_rt, -0.16619047619047619_rt, 0.080657596371882093_rt, -0.017712585034013604_rt},
    {0.017712585034013604_rt, -0.080657596371882093_rt, 0.16619047619047619_rt, -0.6227891156462585_rt, 0.22073412698412698_rt, 0.44993197278911562_rt, -0.17642857142857143_rt, 0.025306122448979593_rt},
    {0.0_rt, -0.021267006802721089_rt, 0.093628117913832201_rt, -0.1680952380952381_rt, 0.36292517006802721_rt, -0.32069444444444445_rt, 0.049931972789115646_rt, 0.0035714285714285713_rt},
    {0.0_rt, 0.0_rt, 0.0080187074829931974_rt, -0.026371882086167801_rt, 0.02119047619047619_rt, -0.045646258503401357_rt, 0.05859126984126984_rt, -0.015782312925170069_rt},
    {0.0_rt, 0.0_rt, 0.0_rt, -0.00089285714285714283_rt, 0.00061224489795918364_rt, 0.0078571428571428577_rt, -0.010408163265306122_rt, 0.0028316326530612246_rt}};

AMREX_GPU_MANAGED Real kM8T[8][8] = {
    {-0.0028316326530612246_rt, 0.015782312925170069_rt, -0.0035714285714285713_rt, -0.025306122448979593_rt, 0.017712585034013604_rt, 0.0_rt, 0.0_rt, 0.0_rt},
    {0.010408163265306122_rt, -0.05859126984126984_rt, -0.049931972789115646_rt, 0.17642857142857143_rt, -0.080657596371882093_rt, -0.021267006802721089_rt, 0.0_rt, 0.0_rt},
    {-0.0078571428571428577_rt, 0.045646258503401357_rt, 0.32069444444444445_rt, -0.44993197278911562_rt, 0.16619047619047619_rt, 0.093628117913832201_rt, 0.0080187074829931974_rt, 0.0_rt},
    {-0.00061224489795918364_rt, -0.02119047619047619_rt, -0.36292517006802721_rt, -0.22073412698412698_rt, -0.6227891156462585_rt, -0.1680952380952381_rt, -0.026371882086167801_rt, -0.00089285714285714283_rt},
    {0.00089285714285714283_rt, 0.026371882086167801_rt, 0.1680952380952381_rt, 0.6227891156462585_rt, 0.22073412698412698_rt, 0.36292517006802721_rt, 0.02119047619047619_rt, 0.00061224489795918364_rt},
    {0.0_rt, -0.0080187074829931974_rt, -0.093628117913832201_rt, -0.16619047619047619_rt, 0.44993197278911562_rt, -0.32069444444444445_rt, -0.045646258503401357_rt, 0.0078571428571428577_rt},
    {0.0_rt, 0.0_rt, 0.021267006802721089_rt, 0.080657596371882093_rt, -0.17642857142857143_rt, 0.049931972789115646_rt, 0.05859126984126984_rt, -0.010408163265306122_rt},
    {0.0_rt, 0.0_rt, 0.0_rt, -0.017712585034013604_rt, 0.025306122448979593_rt, 0.0035714285714285713_rt, -0.015782312925170069_rt, 0.0028316326530612246_rt}};
enum ConsComp { URHO = 0, UMX, UMY, UMZ, UEDEN, URY1 };
enum PrimComp { QRHO = 0, QU, QV, QW, QPRES, QTEMP, QEINT, QY = 7 };
constexpr int QX = QY + NSpecies;
constexpr int QH = QX + NSpecies;
AMREX_GPU_MANAGED int kMomentumIndex[AMREX_SPACEDIM] = {AMREX_D_DECL(UMX, UMY, UMZ)};
AMREX_GPU_MANAGED int kVelocityIndex[AMREX_SPACEDIM] = {AMREX_D_DECL(QU, QV, QW)};
constexpr int kInertSpecies = N2_ID;
constexpr Real TwoThirds = 2.0_rt / 3.0_rt;
constexpr Real FourThirds = 4.0_rt / 3.0_rt;

template <typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void gather_line(int dir,
                 int i, int j, int k,
                 const F& func,
                 GpuArray<Real, 8>& data) noexcept
{
    for (int idx = 0; idx < 8; ++idx) {
        int ii = i;
        int jj = j;
        int kk = k;
        apply_offset(dir, idx - 4, ii, jj, kk);
        data[idx] = func(ii, jj, kk);
    }
}

template <typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real, 8>
matmul_M8_line(int dir, int i, int j, int k, const F& func) noexcept
{
    GpuArray<Real, 8> line;
    gather_line(dir, i, j, k, func, line);
    GpuArray<Real, 8> result;
    for (int col = 0; col < 8; ++col) {
        Real sum = 0.0_rt;
        for (int row = 0; row < 8; ++row) {
            sum += kM8[row][col] * line[row];
        }
        result[col] = sum;
    }
    return result;
}

template <typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
GpuArray<Real, 8>
matmul_M8T_line(int dir, int i, int j, int k, const F& func) noexcept
{
    GpuArray<Real, 8> line;
    gather_line(dir, i, j, k, func, line);
    GpuArray<Real, 8> result;
    for (int col = 0; col < 8; ++col) {
        Real sum = 0.0_rt;
        for (int row = 0; row < 8; ++row) {
            sum += kM8T[row][col] * line[row];
        }
        result[col] = sum;
    }
    return result;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real get_face_flux(const Array4<const Real>& flux,
                   int dir,
                   int i, int j, int k,
                   int comp,
                   int shift) noexcept
{
    if (dir == 0) {
        return flux(i + shift, j, k, comp);
    } else if (dir == 1) {
        return flux(i, j + shift, k, comp);
#if (AMREX_SPACEDIM == 3)
    } else {
        return flux(i, j, k + shift, comp);
#else
    } else {
        return 0.0_rt;
#endif
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real compute_vsp(const Array4<const Real>& mu,
                 const Array4<const Real>& xi,
                 int i, int j, int k) noexcept
{
    return xi(i, j, k, 0) + FourThirds * mu(i, j, k, 0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real compute_dpy(const Array4<const Real>& qp,
                 const Array4<const Real>& dp,
                 int i, int j, int k,
                 int n) noexcept
{
    Real diff = dp(i, j, k, n);
    Real pres = qp(i, j, k, QPRES);
    Real xmf = qp(i, j, k, QX + n);
    Real ymf = qp(i, j, k, QY + n);
    return diff / pres * (xmf - ymf);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real compute_dxe(const Array4<const Real>& qp,
                 const Array4<const Real>& dp,
                 int i, int j, int k,
                 int n) noexcept
{
    return dp(i, j, k, n) * qp(i, j, k, QH + n);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real compute_dpe(const Array4<const Real>& qp,
                 const Array4<const Real>& dp,
                 int i, int j, int k) noexcept
{
    Real sum = 0.0_rt;
    for (int n = 0; n < NSpecies; ++n) {
        Real dpy = compute_dpy(qp, dp, i, j, k, n);
        sum += dpy * qp(i, j, k, QH + n);
    }
    return sum;
}


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

} // namespace

namespace {

void add_diffusive_part2(const Geometry& geom,
                         const MultiFab& prim,
                         const MultiFab& mu,
                         const MultiFab& xi,
                         const MultiFab& lam,
                         const MultiFab& Ddiag,
                         MultiFab& rhs)
{
    const auto dxinv = geom.InvCellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> dx2inv;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        dx2inv[d] = dxinv[d] * dxinv[d];
    }

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

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int idir = dir;
            const Box nodal = amrex::surroundingNodes(bx, idir);
            FArrayBox flux_fab(nodal, NCons, The_Async_Arena());
            flux_fab.setVal<RunOn::Device>(0.0);
            auto flux = flux_fab.array();

            ParallelFor(nodal, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                GpuArray<Real, NCons> face_flux;
                for (int comp = 0; comp < NCons; ++comp) {
                    face_flux[comp] = 0.0_rt;
                }

                auto vsp_accessor = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return compute_vsp(mup, xip, ii, jj, kk);
                };
                auto mu_accessor = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mup(ii, jj, kk, 0);
                };
                auto lam_accessor = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return lamp(ii, jj, kk, 0);
                };
                auto pres_accessor = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return qp(ii, jj, kk, QPRES);
                };

                GpuArray<Real, 8> mat_vsp = matmul_M8_line(idir, i, j, k, vsp_accessor);
                GpuArray<Real, 8> mat_mu = matmul_M8_line(idir, i, j, k, mu_accessor);
                GpuArray<Real, 8> mat_lam = matmul_M8_line(idir, i, j, k, lam_accessor);
                GpuArray<Real, 8> mat_pres = matmul_M8T_line(idir, i, j, k, pres_accessor);

                auto dot_with_comp = [&](const GpuArray<Real, 8>& coeffs,
                                         int comp) noexcept -> Real {
                    Real sum = 0.0_rt;
                    for (int idx = 0; idx < 8; ++idx) {
                        int ii = i;
                        int jj = j;
                        int kk = k;
                        apply_offset(idir, idx - 4, ii, jj, kk);
                        sum += coeffs[idx] * qp(ii, jj, kk, comp);
                    }
                    return sum;
                };

                auto dot_arrays = [&](const GpuArray<Real, 8>& a,
                                      const GpuArray<Real, 8>& b) noexcept -> Real {
                    Real sum = 0.0_rt;
                    for (int idx = 0; idx < 8; ++idx) {
                        sum += a[idx] * b[idx];
                    }
                    return sum;
                };

                int normal_mom = kMomentumIndex[idir];
                int normal_vel = kVelocityIndex[idir];
                face_flux[normal_mom] += dot_with_comp(mat_vsp, normal_vel);

                for (int t = 0; t < AMREX_SPACEDIM; ++t) {
                    if (t == idir) {
                        continue;
                    }
                    int mom_comp = kMomentumIndex[t];
                    int vel_comp = kVelocityIndex[t];
                    face_flux[mom_comp] += dot_with_comp(mat_mu, vel_comp);
                }

                face_flux[UEDEN] += dot_with_comp(mat_lam, QTEMP);

                GpuArray<Real, 8> dpe_line;
                gather_line(idir, i, j, k,
                    [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                        return compute_dpe(qp, dp, ii, jj, kk);
                    },
                    dpe_line);
                face_flux[UEDEN] += dot_arrays(dpe_line, mat_pres);

                for (int n = 0; n < NSpecies; ++n) {
                    if (n == kInertSpecies) {
                        continue;
                    }
                    const int comp = URY1 + n;

                    GpuArray<Real, 8> dpy_line;
                    gather_line(idir, i, j, k,
                        [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                            return compute_dpy(qp, dp, ii, jj, kk, n);
                        },
                        dpy_line);
                    face_flux[comp] += dot_arrays(dpy_line, mat_pres);

                    auto qx_accessor = [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                        return qp(ii, jj, kk, QX + n);
                    };
                    GpuArray<Real, 8> mat_qx = matmul_M8T_line(idir, i, j, k, qx_accessor);

                    GpuArray<Real, 8> dxy_line;
                    gather_line(idir, i, j, k,
                        [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                            return dp(ii, jj, kk, n);
                        },
                        dxy_line);
                    face_flux[comp] += dot_arrays(dxy_line, mat_qx);

                    GpuArray<Real, 8> dxe_line;
                    gather_line(idir, i, j, k,
                        [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                            return compute_dxe(qp, dp, ii, jj, kk, n);
                        },
                        dxe_line);
                    face_flux[UEDEN] += dot_arrays(dxe_line, mat_qx);
                }

                for (int comp = 0; comp < NCons; ++comp) {
                    flux(i, j, k, comp) = face_flux[comp];
                }
            });

            FArrayBox sumdry_fab(bx, 1, The_Async_Arena());
            sumdry_fab.setVal<RunOn::Device>(0.0);
            auto sumdry = sumdry_fab.array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                for (int comp = UMX; comp <= UEDEN; ++comp) {
                    Real flux_hi = get_face_flux(flux, idir, i, j, k, comp, 1);
                    Real flux_lo = get_face_flux(flux, idir, i, j, k, comp, 0);
                    rp(i, j, k, comp) += (flux_hi - flux_lo) * dx2inv[idir];
                }
            });

            for (int n = 0; n < NSpecies; ++n) {
                const int comp = URY1 + n;
                const int inert_flag = (n == kInertSpecies) ? 1 : 0;
                ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                    Real flux_hi = get_face_flux(flux, idir, i, j, k, comp, 1);
                    Real flux_lo = get_face_flux(flux, idir, i, j, k, comp, 0);
                    Real div = (flux_hi - flux_lo) * dx2inv[idir];
                    rp(i, j, k, comp) += div;
                    if (inert_flag == 0) {
                        sumdry(i, j, k) += div;
                    }
                });
            }

            FArrayBox gradp_fab(bx, 1, The_Async_Arena());
            gradp_fab.setVal<RunOn::Device>(0.0);
            auto gradp = gradp_fab.array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                gradp(i, j, k) = central_diff(qp, QPRES, i, j, k, idir, dxinv);
            });

            FArrayBox sumryv_fab(bx, 1, The_Async_Arena());
            sumryv_fab.setVal<RunOn::Device>(0.0);
            auto sumryv = sumryv_fab.array();

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Real accum = 0.0_rt;
                for (int n = 0; n < NSpecies; ++n) {
                    if (n == kInertSpecies) {
                        continue;
                    }
                    Real dpy_val = compute_dpy(qp, dp, i, j, k, n);
                    Real grad_qx = central_diff(qp, QX + n, i, j, k, idir, dxinv);
                    accum += dpy_val * gradp(i, j, k) + dp(i, j, k, n) * grad_qx;
                }
                sumryv(i, j, k) = accum;
            });

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                Real sumdry_val = sumdry(i, j, k);
                Real sumryv_val = sumryv(i, j, k);
                Real qh = qp(i, j, k, QH + kInertSpecies);
                Real grad_qh = central_diff(qp, QH + kInertSpecies, i, j, k, idir, dxinv);
                Real corr = sumdry_val * qh + sumryv_val * grad_qh;
                rp(i, j, k, UEDEN) -= corr;
                rp(i, j, k, URY1 + kInertSpecies) -= sumdry_val;
            });
        }

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real corr = rp(i, j, k, UMX) * qp(i, j, k, QU)
                      + rp(i, j, k, UMY) * qp(i, j, k, QV);
#if (AMREX_SPACEDIM == 3)
            corr += rp(i, j, k, UMZ) * qp(i, j, k, QW);
#endif
            rp(i, j, k, UEDEN) += corr;
        });
    }
}

} // namespace

void AddDiffusive(const Geometry& geom,
                  const MultiFab& prim,
                  const MultiFab& mu,
                  const MultiFab& xi,
                  const MultiFab& lam,
                  const MultiFab& Ddiag,
                  MultiFab& rhs)
{
    add_diffusive_part1(geom, prim, mu, xi, lam, Ddiag, rhs);
    add_diffusive_part2(geom, prim, mu, xi, lam, Ddiag, rhs);
}

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
            lamp(i, j, k) = mu_val * cp * kInvPrandtl;
            Real invW = 1.0_rt / Wbar;
            for (int n = 0; n < NSpecies; ++n) {
                Real mw_n = mw(n);
                dp(i, j, k, n) = mu_val * mw_n * invW * kInvSchmidt;
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

namespace {

void add_diffusive_part1(const Geometry& geom,
                         const MultiFab& prim,
                         const MultiFab& mu,
                         const MultiFab& xi,
                         const MultiFab& lam,
                         const MultiFab& Ddiag,
                         MultiFab& rhs)
{
    const auto dxinv = geom.InvCellSizeArray();
    amrex::ignore_unused(lam);
    amrex::ignore_unused(Ddiag);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        const Box gbox = amrex::grow(bx, StencilNG);
        auto rp = rhs.array(mfi);
        auto qp = prim.const_array(mfi);
        auto mup = mu.const_array(mfi);
        auto xip = xi.const_array(mfi);
        FArrayBox gradfab(gbox, kNumVelDeriv, The_Async_Arena());
        auto grad = gradfab.array();

        ParallelFor(gbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            grad(i, j, k, DUDX) = central_diff(qp, QU, i, j, k, 0, dxinv);
            grad(i, j, k, DVDX) = central_diff(qp, QV, i, j, k, 0, dxinv);
            grad(i, j, k, DWDX) = central_diff(qp, QW, i, j, k, 0, dxinv);
            grad(i, j, k, DUDY) = central_diff(qp, QU, i, j, k, 1, dxinv);
            grad(i, j, k, DVDY) = central_diff(qp, QV, i, j, k, 1, dxinv);
            grad(i, j, k, DWDY) = central_diff(qp, QW, i, j, k, 1, dxinv);
#if (AMREX_SPACEDIM == 3)
            grad(i, j, k, DUDZ) = central_diff(qp, QU, i, j, k, 2, dxinv);
            grad(i, j, k, DVDZ) = central_diff(qp, QV, i, j, k, 2, dxinv);
            grad(i, j, k, DWDZ) = central_diff(qp, QW, i, j, k, 2, dxinv);
#else
            grad(i, j, k, DUDZ) = 0.0_rt;
            grad(i, j, k, DVDZ) = 0.0_rt;
            grad(i, j, k, DWDZ) = 0.0_rt;
#endif
        });

        auto vsm_eval = [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            return xip(i, j, k, 0) - TwoThirds * mup(i, j, k, 0);
        };
        auto mu_eval = [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            return mup(i, j, k, 0);
        };

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            Real mx_x = central_diff_fn(i, j, k, 0, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real vy = grad(ii, jj, kk, DVDY);
                    Real wz = grad(ii, jj, kk, DWDZ);
                    return vsm_eval(ii, jj, kk) * (vy + wz);
                });

            Real mx_y = central_diff_fn(i, j, k, 1, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DVDX);
                });

            Real mx_z = 0.0_rt;
#if (AMREX_SPACEDIM == 3)
            mx_z = central_diff_fn(i, j, k, 2, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DWDX);
                });
#endif
            rp(i, j, k, UMX) += mx_x + mx_y + mx_z;

            Real my_x = central_diff_fn(i, j, k, 0, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DUDY);
                });

            Real my_y = central_diff_fn(i, j, k, 1, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real ux = grad(ii, jj, kk, DUDX);
                    Real wz = grad(ii, jj, kk, DWDZ);
                    return vsm_eval(ii, jj, kk) * (ux + wz);
                });

            Real my_z = 0.0_rt;
#if (AMREX_SPACEDIM == 3)
            my_z = central_diff_fn(i, j, k, 2, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DWDY);
                });
#endif
            rp(i, j, k, UMY) += my_x + my_y + my_z;

#if (AMREX_SPACEDIM == 3)
            Real mz_x = central_diff_fn(i, j, k, 0, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DUDZ);
                });

            Real mz_y = central_diff_fn(i, j, k, 1, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    return mu_eval(ii, jj, kk) * grad(ii, jj, kk, DVDZ);
                });

            Real mz_z = central_diff_fn(i, j, k, 2, dxinv,
                [=] AMREX_GPU_DEVICE(int ii, int jj, int kk) noexcept {
                    Real ux = grad(ii, jj, kk, DUDX);
                    Real vy = grad(ii, jj, kk, DVDY);
                    return vsm_eval(ii, jj, kk) * (ux + vy);
                });

            rp(i, j, k, UMZ) += mz_x + mz_y + mz_z;
#endif

            Real mu_c = mu_eval(i, j, k);
            Real vsm_c = vsm_eval(i, j, k);
            Real dudx = grad(i, j, k, DUDX);
            Real dvdy = grad(i, j, k, DVDY);
            Real divu = dudx + dvdy;
#if (AMREX_SPACEDIM == 3)
            Real dwdz = grad(i, j, k, DWDZ);
            divu += dwdz;
#else
            Real dwdz = 0.0_rt;
#endif
            Real divu_term = vsm_c * divu;
            Real tauxx = 2.0_rt * mu_c * dudx + divu_term;
            Real tauyy = 2.0_rt * mu_c * dvdy + divu_term;
            Real tauzz = divu_term;
#if (AMREX_SPACEDIM == 3)
            tauzz = 2.0_rt * mu_c * dwdz + divu_term;
#endif

            Real shear1 = grad(i, j, k, DUDY) + grad(i, j, k, DVDX);
            Real shear2 = grad(i, j, k, DWDX);
#if (AMREX_SPACEDIM == 3)
            shear2 += grad(i, j, k, DUDZ);
            Real shear3 = grad(i, j, k, DVDZ) + grad(i, j, k, DWDY);
#else
            Real shear3 = 0.0_rt;
#endif

            Real visc_ene = tauxx * dudx + tauyy * dvdy;
#if (AMREX_SPACEDIM == 3)
            visc_ene += tauzz * dwdz;
#endif
            visc_ene += mu_c * (shear1 * shear1 + shear2 * shear2
#if (AMREX_SPACEDIM == 3)
                                + shear3 * shear3
#endif
                                );

            rp(i, j, k, UEDEN) += visc_ene;
        });
    }
}

} // namespace

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
