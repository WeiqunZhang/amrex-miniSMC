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

1. Install AMReX with the desired backend (set `AMReX_GPU_BACKEND=NONE`, `CUDA`, `HIP`, or `SYCL`). Make sure `AMReX_DIR` points to the install location containing the generated `AMReXConfig.cmake`.
2. Configure and build:
   ```bash
   cd amrex-miniSMC
   cmake -S . -B build -DAMReX_GPU_BACKEND=CUDA   # choose backend
   cmake --build build -j
   ```
3. Run with MPI (example uses 4 ranks):
   ```bash
   mpirun -n 4 build/miniSMC inputs/inputs_smc
   ```

AMReX handles OpenMP threading automatically when configured with `-DAMReX_OMP=ON`. GPU backends are enabled via `AMReX_GPU_BACKEND` during configure time; no code changes are required.

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

