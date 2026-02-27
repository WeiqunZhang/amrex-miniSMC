cmake_minimum_required(VERSION 3.24)
project(amrex_miniSMC LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(AMReX CONFIG REQUIRED)

add_executable(miniSMC
    src/main.cpp
    src/MiniSMC.cpp
    src/SMCKernels.cpp
    src/Mechanism.cpp)

target_include_directories(miniSMC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(miniSMC PRIVATE AMReX::amrex)

