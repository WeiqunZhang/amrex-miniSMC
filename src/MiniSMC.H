#pragma once

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>
#include <AMReX_RealBox.H>

#include "Mechanism.H"

namespace minismc {

using Real = amrex::Real;

constexpr int NSpecies = 9;
constexpr int NCons = 5 + NSpecies;
constexpr int NPrim = 7 + 3 * NSpecies;
constexpr int StencilNG = 4;

struct ProbParm {
    amrex::IntVect ncell{AMREX_D_DECL(128, 128, 128)};
    amrex::RealBox prob_domain;
    int max_grid_size = 128;
    int max_step = 1;
    int cfl_int = 10;
    int verbose = 0;
    Real stop_time = -1.0;
    Real cfl = 0.5;
    Real init_shrink = 1.0;
    Real small_dt = 1.0e-30;
    Real max_dt_growth = 1.1;
    Real max_dt = 1.0e33;
    Real fixed_dt = -1.0;
    Real rfire = 0.01;
};

struct TimingStats {
    Real advance = 0.0;
    Real fillboundary = 0.0;
    Real ctoprim = 0.0;
    Real chemistry = 0.0;
    Real transport = 0.0;
    Real hypdiff = 0.0;
};

class MiniSMC {
public:
    MiniSMC();
    void InitData();
    void Evolve();

    static void SetInputFilePath(const std::string& path);
    static const std::string& GetInputFilePath();

private:
    void read_parameters();
    void build_mesh();
    void set_dt(Real courno, int istep);
    void advance_step(int istep);
    void compute_rhs(bool need_courno, Real& courno);
    void compute_primitives();
    void reset_density();
    void compute_chemistry();
    void compute_transport();
    void add_hyperbolic_terms();
    void add_diffusive_terms();
    void compute_courno(Real& courno);
    void init_from_scratch();

    ProbParm m_prob;
    TimingStats m_timers;

    Real m_time = 0.0;
    Real m_dt = 1.0e20;
    Real m_last_courno = 1.0;

    amrex::Geometry m_geom;
    amrex::BoxArray m_ba;
    amrex::DistributionMapping m_dm;

    amrex::MultiFab m_state;
    amrex::MultiFab m_stage;
    amrex::MultiFab m_rhs;
    amrex::MultiFab m_prim;
    amrex::MultiFab m_mu;
    amrex::MultiFab m_xi;
    amrex::MultiFab m_lam;
    amrex::MultiFab m_Ddiag;

    amrex::GpuArray<Real, AMREX_SPACEDIM> m_dx{{AMREX_D_DECL(0.0, 0.0, 0.0)}};
};

} // namespace minismc
