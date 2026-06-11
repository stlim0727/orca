cmake_minimum_required(VERSION 3.20)
project(orca LANGUAGES CXX)

# Toolchain choice locked in AGENT.md: C++17; CUDA/DOCA behind OFF-able flags.
option(EMU_WITH_CUDA "Build the CUDA (GPU) backends and kernels" OFF)
option(EMU_WITH_DOCA "Build the DOCA GPUNetIO backend (deferred, ADR 0007)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EMU_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Header-only foundations (common/); compiled modules attach as stages land.
add_library(orca_common INTERFACE)
target_include_directories(orca_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# DSP kernels — host (CPU) implementations; CUDA variants land behind
# EMU_WITH_CUDA with the same signatures (Spec E §E.7).
add_library(orca_dsp STATIC dsp/convert.cpp dsp/precode.cpp dsp/awgn.cpp
                            dsp/channel_apply.cpp dsp/combine.cpp)
target_link_libraries(orca_dsp PUBLIC orca_common)

# Separate ORU process program (ADR 0007) — host build runs an in-memory
# smoke loop until Linux socket bring-up.
add_executable(oru_process oru_process/main.cpp)
target_link_libraries(oru_process PRIVATE orca_common)

# ORCA application — Stage-1 identity hot path over the transport seams.
add_executable(orca_app app/main.cpp)
target_link_libraries(orca_app PRIVATE orca_common orca_dsp)

enable_testing()
add_subdirectory(tests)
