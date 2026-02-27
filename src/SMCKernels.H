#pragma once

#include "MiniSMC.H"

namespace minismc::kernels {

void InitData(const amrex::Geometry& geom,
              amrex::MultiFab& state,
              const ProbParm& prob,
              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx);

void ComputePrimitives(const amrex::Geometry& geom,
                       const amrex::MultiFab& state,
                       amrex::MultiFab& prim);

void ResetDensity(amrex::MultiFab& state);

void AddChemistry(amrex::MultiFab& rhs,
                  const amrex::MultiFab& prim);

void ComputeTransport(const amrex::MultiFab& prim,
                      amrex::MultiFab& mu,
                      amrex::MultiFab& xi,
                      amrex::MultiFab& lam,
                      amrex::MultiFab& Ddiag);

void AddHyperbolic(const amrex::Geometry& geom,
                   const amrex::MultiFab& state,
                   const amrex::MultiFab& prim,
                   amrex::MultiFab& rhs);

void AddDiffusive(const amrex::Geometry& geom,
                  const amrex::MultiFab& prim,
                  const amrex::MultiFab& mu,
                  const amrex::MultiFab& xi,
                  const amrex::MultiFab& lam,
                  const amrex::MultiFab& Ddiag,
                  amrex::MultiFab& rhs);

void ComputeCourant(const amrex::Geometry& geom,
                    const amrex::MultiFab& prim,
                    amrex::Real& courno);

}

