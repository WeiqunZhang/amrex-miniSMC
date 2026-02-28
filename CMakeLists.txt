cmake_minimum_required(VERSION 3.24)

option(MINISMC_USE_SUPERBUILD "Build AMReX from an in-tree source checkout" OFF)
option(MINISMC_DISABLE_CCACHE "Force-disable ccache to avoid permission issues on shared runners" OFF)
option(MINISMC_AUTO_GPU_ARCH "Detect GPU compute capability with nvidia-smi when targeting GPUs" ON)
set(AMREX_SOURCE_DIR "" CACHE PATH "Path to the AMReX source tree (used when super-building)")

if(MINISMC_DISABLE_CCACHE)
    set(ENV{CCACHE_DISABLE} 1)
    set(ENV{CMAKE_C_COMPILER_LAUNCHER} "")
    set(ENV{CMAKE_CXX_COMPILER_LAUNCHER} "")
    set(ENV{CMAKE_CUDA_COMPILER_LAUNCHER} "")
    set(ENV{CMAKE_HIP_COMPILER_LAUNCHER} "")
    set(CMAKE_C_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
    set(CMAKE_HIP_COMPILER_LAUNCHER "" CACHE STRING "" FORCE)
endif()

project(amrex_miniSMC LANGUAGES CXX)

set(_gpu_backend "")
if(DEFINED AMReX_GPU_BACKEND)
    string(TOUPPER "${AMReX_GPU_BACKEND}" _gpu_backend)
endif()

if(_gpu_backend STREQUAL "CUDA")
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    if(MINISMC_AUTO_GPU_ARCH)
        find_program(NVIDIA_SMI nvidia-smi)
        if(NVIDIA_SMI)
            execute_process(
                COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE _smcaps
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(_smcaps)
                string(REGEX MATCH "[0-9]+\\.[0-9]+" _first_cap "${_smcaps}")
                if(_first_cap)
                    string(REPLACE "." "" _arch "${_first_cap}")
                    set(_arch_value "${_arch}")
                    if(NOT _arch_value STREQUAL "")
                        set(CMAKE_CUDA_ARCHITECTURES "${_arch_value}-real" CACHE STRING
                            "CUDA SM architectures to target" FORCE)
                        set(AMReX_CUDA_ARCH "${_arch_value}" CACHE STRING
                            "AMReX CUDA architecture" FORCE)
                        message(STATUS "Detected CUDA compute capability ${_first_cap}; using sm_${_arch_value} (no PTX fallback)")
                    endif()
                endif()
            endif()
        else()
            message(WARNING "nvidia-smi not found; unable to auto-detect CUDA architecture. "
                            "Consider setting CMAKE_CUDA_ARCHITECTURES manually.")
        endif()
    endif()
elseif(_gpu_backend STREQUAL "HIP")
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
elseif(_gpu_backend STREQUAL "SYCL")
    # SYCL builds still use the host C++ compiler but rely on Clang extensions,
    # so there is no dedicated language to enable here.  The branch exists so
    # future SYCL-specific setup can be added without touching the rest.
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MINISMC_USE_SUPERBUILD OR (NOT AMReX_DIR AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../amrex/CMakeLists.txt"))
    if(NOT AMREX_SOURCE_DIR)
        set(AMREX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../amrex" CACHE PATH "Path to the AMReX source tree" FORCE)
    endif()
    if(NOT EXISTS "${AMREX_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "AMReX source directory '${AMREX_SOURCE_DIR}' does not contain a CMakeLists.txt")
    endif()

    message(STATUS "Configuring AMReX from source: ${AMREX_SOURCE_DIR}")

    # Users can pass standard AMReX_* cache variables (e.g., AMReX_GPU_BACKEND, AMReX_MPI)
    add_subdirectory(${AMREX_SOURCE_DIR} ${CMAKE_BINARY_DIR}/extern/amrex)
else()
    find_package(AMReX CONFIG REQUIRED)
endif()

set(MINISMC_SOURCES
    src/main.cpp
    src/MiniSMC.cpp
    src/SMCKernels.cpp
    src/Mechanism.cpp)

add_executable(miniSMC ${MINISMC_SOURCES})

if(_gpu_backend STREQUAL "CUDA")
    set_source_files_properties(${MINISMC_SOURCES} PROPERTIES LANGUAGE CUDA)
    set_target_properties(miniSMC PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
elseif(_gpu_backend STREQUAL "HIP")
    set_source_files_properties(${MINISMC_SOURCES} PROPERTIES LANGUAGE HIP)
    set_target_properties(miniSMC PROPERTIES HIP_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(miniSMC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(miniSMC PRIVATE AMReX::amrex)
