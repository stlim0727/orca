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

# Separate ORU process program (ADR 0007) — host build runs an in-memory
# smoke loop until Linux socket bring-up.
add_executable(oru_process oru_process/main.cpp)
target_link_libraries(oru_process PRIVATE orca_common)

enable_testing()
add_subdirectory(tests)
