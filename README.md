# AMReX miniSMC

This directory contains a C++ port of the legacy `miniSMC` solver that now uses the AMReX library for parallel mesh management and supports MPI + X execution (OpenMP on CPUs and CUDA/HIP/SYCL through AMReX's GPU backend).

## Layout

- `src/MiniSMC.H/.cpp` – driver class that owns the mesh, state multifabs, runtime parameters, and RK3 integrator.
- `src/SMCKernels.H/.cpp` – data-parallel kernels for initialization, primitive variable conversion, chemistry, transport properties, hyperbolic terms, diffusion, and CFL calculation. Kernels rely on AMReX `ParallelFor` so they run on CPUs (OpenMP) or GPUs.
- `src/Mechanism.H/.cpp` – LiDryer mechanism from PelePhysics (BSD-3). Provides thermodynamic/transport/chemistry functions that are GPU-ready.
- `src/main.cpp` – program entry point.
- `inputs/inputs_smc` – example input deck compatible with the original Fortran namelist parameters.
- `CMakeLists.txt` – CMake build script.

## Building

Two workflows are supported: using an existing AMReX installation or running a super-build that compiles AMReX from the checked-out sources that now live alongside this directory.

### Super-build (local AMReX checkout)

```bash
cd amrex-miniSMC
cmake -S . -B build \
      -DMINISMC_USE_SUPERBUILD=ON \
      -DMINISMC_DISABLE_CCACHE=ON \
      -DAMREX_SOURCE_DIR=../amrex \
      -DAMReX_GPU_BACKEND=CUDA -DAMReX_MPI=ON   # pick backend/options
cmake --build build -j
```

You can pass any regular `AMReX_*` cache variable at configure time (e.g., `-DAMReX_OMP=ON`, `-DAMReX_AMRDATA=OFF`, …); they will be forwarded to the internal AMReX configure step.

### Using an existing AMReX installation

```bash
cd amrex-miniSMC
cmake -S . -B build -DAMReX_DIR=/path/to/amrex/install \
      -DMINISMC_DISABLE_CCACHE=ON \
      -DAMReX_GPU_BACKEND=CUDA
cmake --build build -j
```

### Running

```bash
mpirun -n 4 build/miniSMC inputs/inputs_smc
```

AMReX handles OpenMP threading automatically when configured with `-DAMReX_OMP=ON`. GPU backends are enabled through `AMReX_GPU_BACKEND` (values: `NONE`, `CUDA`, `HIP`, `SYCL`). `MINISMC_DISABLE_CCACHE` is ON by default to avoid sandbox permission problems; flip it OFF if you want to re-enable your ccache setup.

## Runtime Parameters

All parameters from the original `inputs_SMC` namelist are supported via plain AMReX `ParmParse` entries (e.g., `n_cellx`, `cflfac`, `stop_time`, etc.). See `inputs/inputs_smc` for defaults.

## Notes and Limitations

- The LiDryer mechanism is imported from the PelePhysics project (BSD-3 license). Cite PelePhysics if you redistribute the mechanism.
- For numerical portability the spatial operators currently use centered second-order finite differences rather than the original eight-order stencils. This makes the port easier to maintain across CPU and GPU backends, but you may wish to raise the order later.
- Viscous heating and species diffusion follow simplified constant-coefficient models. Transport coefficients are still evaluated from local thermodynamic states, so you can refine them if higher fidelity is required.
- Plot/output routines have not been added yet; AMReX `WriteMultiLevelPlotfile` can be wired in once an output cadence is defined.

## Next Steps

- Validate the new solver against the legacy miniSMC benchmarks and tune the `inputs` deck accordingly.
- Reintroduce the 8th-order derivative stencils if that accuracy level is required.
- Hook up plotfile/density diagnostics or checkpoints as needed.
