cmake_minimum_required(VERSION 3.20)

project(orca LANGUAGES CXX)

option(EMU_WITH_CUDA "Build CUDA-backed DSP kernels" OFF)
option(EMU_WITH_DOCA "Build DOCA-backed transports" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(EMU_WITH_DOCA)
  message(FATAL_ERROR "EMU_WITH_DOCA is deferred for this implementation stage")
endif()

if(EMU_WITH_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

add_library(orca_common INTERFACE)
target_include_directories(orca_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(orca_dsp STATIC dsp/convert.cpp)
target_link_libraries(orca_dsp PUBLIC orca_common)

if(EMU_WITH_CUDA)
  target_sources(orca_dsp PRIVATE dsp/convert.cu)
  set_source_files_properties(dsp/convert.cu PROPERTIES LANGUAGE CUDA)
  target_compile_definitions(orca_dsp PUBLIC ORCA_WITH_CUDA=1)
else()
  target_compile_definitions(orca_dsp PUBLIC ORCA_WITH_CUDA=0)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(test_layout tests/test_layout.cpp)
  target_link_libraries(test_layout PRIVATE orca_dsp)
  add_test(NAME test_layout COMMAND test_layout)

  if(EMU_WITH_CUDA)
    add_executable(test_convert_cuda tests/test_convert_cuda.cu)
    set_source_files_properties(tests/test_convert_cuda.cu PROPERTIES LANGUAGE CUDA)
    target_link_libraries(test_convert_cuda PRIVATE orca_dsp)
    add_test(NAME test_convert_cuda COMMAND test_convert_cuda)
  endif()
endif()
