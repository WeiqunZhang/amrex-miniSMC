#include "MiniSMC.H"
#include "SMCKernels.H"

#include <AMReX_FArrayBox.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <limits>

using namespace amrex;

namespace minismc {

namespace {
constexpr int URHO = 0;
constexpr int UMX = 1;
constexpr int UMY = 2;
constexpr int UMZ = 3;
constexpr int UEDEN = 4;
constexpr int URY1 = 5;

constexpr int QRHO = 0;
constexpr int QU = 1;
constexpr int QV = 2;
constexpr int QW = 3;
constexpr int QPRES = 4;
constexpr int QTEMP = 5;
constexpr int QEINT = 6;
constexpr int QY = 7;
constexpr int QX = QY + NSpecies;
constexpr int QH = QX + NSpecies;

constexpr Real Zero = 0.0_rt;
constexpr Real One = 1.0_rt;
constexpr Real OneThird = One / 3.0_rt;
constexpr Real TwoThirds = 2.0_rt / 3.0_rt;
constexpr Real FourThirds = 4.0_rt / 3.0_rt;
constexpr Real OneQuarter = 0.25_rt;
constexpr Real ThreeQuarters = 0.75_rt;
}

MiniSMC::MiniSMC()
{
    read_parameters();
    build_mesh();
}

void MiniSMC::read_parameters()
{
    ParmParse pp("smc");

    Vector<int> n_cell(AMREX_SPACEDIM, -1);
    if (pp.contains("n_cell")) {
        pp.getarr("n_cell", n_cell, 0, AMREX_SPACEDIM);
    } else {
        pp.query("n_cellx", n_cell[0]);
        pp.query("n_celly", n_cell[1]);
#if (AMREX_SPACEDIM == 3)
        pp.query("n_cellz", n_cell[2]);
#endif
    }
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (n_cell[d] > 0) {
            m_prob.ncell[d] = n_cell[d];
        }
    }

    pp.query("max_grid_size", m_prob.max_grid_size);
    pp.query("max_step", m_prob.max_step);
    pp.query("stop_time", m_prob.stop_time);
    pp.query("cflfac", m_prob.cfl);
    pp.query("cfl_int", m_prob.cfl_int);
    pp.query("init_shrink", m_prob.init_shrink);
    pp.query("small_dt", m_prob.small_dt);
    pp.query("max_dt_growth", m_prob.max_dt_growth);
    pp.query("max_dt", m_prob.max_dt);
    pp.query("fixed_dt", m_prob.fixed_dt);
    pp.query("verbose", m_prob.verbose);
    pp.query("rfire", m_prob.rfire);

    Vector<Real> plo(AMREX_SPACEDIM);
    Vector<Real> phi(AMREX_SPACEDIM);
    if (pp.contains("prob_lo")) {
        pp.getarr("prob_lo", plo, 0, AMREX_SPACEDIM);
    } else {
        pp.query("prob_lo_x", plo[0]);
        pp.query("prob_lo_y", plo[1]);
#if (AMREX_SPACEDIM == 3)
        pp.query("prob_lo_z", plo[2]);
#endif
    }
    if (pp.contains("prob_hi")) {
        pp.getarr("prob_hi", phi, 0, AMREX_SPACEDIM);
    } else {
        pp.query("prob_hi_x", phi[0]);
        pp.query("prob_hi_y", phi[1]);
#if (AMREX_SPACEDIM == 3)
        pp.query("prob_hi_z", phi[2]);
#endif
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        if (phi[d] <= plo[d]) {
            amrex::Abort("Invalid prob_lo/prob_hi specification");
        }
    }

    m_prob.prob_domain = amrex::RealBox(plo.data(), phi.data());
}

void MiniSMC::build_mesh()
{
    IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    IntVect dom_hi(m_prob.ncell);
    dom_hi -= IntVect(AMREX_D_DECL(1, 1, 1));

    Box domain(dom_lo, dom_hi);
    Vector<int> is_per(AMREX_SPACEDIM, 1);
    m_geom.define(domain, m_prob.prob_domain, CoordSys::cartesian, is_per.data());

    m_dx = m_geom.CellSizeArray();

    m_ba.define(domain);
    m_ba.maxSize(m_prob.max_grid_size);
    m_dm.define(m_ba);

    m_state.define(m_ba, m_dm, NCons, StencilNG);
    m_stage.define(m_ba, m_dm, NCons, StencilNG);
    m_rhs.define(m_ba, m_dm, NCons, 0);
    m_prim.define(m_ba, m_dm, NPrim, StencilNG);
    m_mu.define(m_ba, m_dm, 1, StencilNG);
    m_xi.define(m_ba, m_dm, 1, StencilNG);
    m_lam.define(m_ba, m_dm, 1, StencilNG);
    m_Ddiag.define(m_ba, m_dm, NSpecies, StencilNG);
}

void MiniSMC::InitData()
{
    init_from_scratch();
}

void MiniSMC::init_from_scratch()
{
    kernels::InitData(m_geom, m_state, m_prob, m_dx);
}

void MiniSMC::Evolve()
{
    if (m_prob.verbose > 0 && ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "Initializing miniSMC" << std::endl;
    }

    InitData();

    const int init_step = 1;
    const Real stoptime = m_prob.stop_time;

    Real courno = -1.0e50_rt;

    Real wt_start = amrex::second();

    if ((m_prob.max_step >= init_step) && (stoptime < 0.0_rt || m_time < stoptime)) {
        for (int istep = init_step; istep <= m_prob.max_step; ++istep) {
            if (ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "Advancing step " << istep << ", time = " << m_time << std::endl;
            }

            advance_step(istep);

            if (ParallelDescriptor::IOProcessor()) {
                amrex::Print() << "End step " << istep << ", time = " << m_time << std::endl;
            }

            if (stoptime >= 0.0_rt && m_time >= stoptime) {
                break;
            }
        }
    }

    Real wt_total = amrex::second() - wt_start;
    ParallelDescriptor::ReduceRealMax(wt_total, ParallelDescriptor::IOProcessorNumber());
    if (ParallelDescriptor::IOProcessor()) {
        amrex::Print() << "miniSMC advance time = " << wt_total << std::endl;
    }
}

void MiniSMC::advance_step(int istep)
{
    bool update_courno = (m_prob.fixed_dt <= 0.0_rt) && (m_prob.cfl_int <= 1 || (istep % m_prob.cfl_int) == 1);
    Real courno = m_last_courno;

    compute_rhs(update_courno, courno);
    if (update_courno) {
        m_last_courno = courno;
    }
    set_dt(m_last_courno, istep);

    // Stage 1
    MultiFab::Copy(m_stage, m_state, 0, 0, NCons, StencilNG);
    MultiFab::Saxpy(m_stage, m_dt, m_rhs, 0, 0, NCons, 0);
    reset_density();

    // Stage 2
    compute_rhs(false, courno);
    MultiFab::LinComb(m_stage, OneQuarter, m_stage, 0, ThreeQuarters, m_state, 0, 0, NCons, StencilNG);
    MultiFab::Saxpy(m_stage, OneQuarter * m_dt, m_rhs, 0, 0, NCons, 0);
    reset_density();

    // Stage 3
    compute_rhs(false, courno);
    MultiFab::LinComb(m_state, OneThird, m_state, 0, TwoThirds, m_stage, 0, 0, NCons, StencilNG);
    MultiFab::Saxpy(m_state, TwoThirds * m_dt, m_rhs, 0, 0, NCons, 0);
    reset_density();

    m_time += m_dt;
}

void MiniSMC::compute_rhs(bool need_courno, Real& courno)
{
    m_state.FillBoundary(m_geom.periodicity());
    m_rhs.setVal(0.0);

    compute_primitives();
    compute_chemistry();
    compute_transport();
    add_hyperbolic_terms();
    add_diffusive_terms();

    if (need_courno) {
        compute_courno(courno);
    }
}

void MiniSMC::compute_primitives()
{
    kernels::ComputePrimitives(m_geom, m_state, m_prim);
}

void MiniSMC::reset_density()
{
    kernels::ResetDensity(m_state);
}

void MiniSMC::compute_chemistry()
{
    kernels::AddChemistry(m_rhs, m_prim);
}

void MiniSMC::compute_transport()
{
    kernels::ComputeTransport(m_prim, m_mu, m_xi, m_lam, m_Ddiag);
}

void MiniSMC::add_hyperbolic_terms()
{
    kernels::AddHyperbolic(m_geom, m_state, m_prim, m_rhs);
}

void MiniSMC::add_diffusive_terms()
{
    kernels::AddDiffusive(m_geom, m_prim, m_mu, m_xi, m_lam, m_Ddiag, m_rhs);
}

void MiniSMC::compute_courno(Real& courno)
{
    kernels::ComputeCourant(m_geom, m_prim, courno);
}

void MiniSMC::set_dt(Real courno, int istep)
{
    Real new_dt = m_dt;
    if (m_prob.fixed_dt > 0.0_rt) {
        new_dt = m_prob.fixed_dt;
    } else {
        new_dt = m_prob.cfl / courno;
        if (istep == 1) {
            new_dt *= m_prob.init_shrink;
        } else {
            const Real growth = new_dt / m_dt;
            if (growth > m_prob.max_dt_growth) {
                new_dt = m_dt * m_prob.max_dt_growth;
            }
        }
        new_dt = std::min(new_dt, m_prob.max_dt);
        if (new_dt < m_prob.small_dt) {
            amrex::Abort("Timestep smaller than small_dt");
        }
        if (m_prob.stop_time > 0.0_rt) {
            if (m_time + new_dt > m_prob.stop_time) {
                new_dt = m_prob.stop_time - m_time;
            }
        }
    }
    m_dt = new_dt;
}

} // namespace minismc
