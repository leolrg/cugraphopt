cmake_minimum_required(VERSION 3.22)

project(CuGraphOpt VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
endif()

include(CTest)

add_library(cugraphopt_core
  src/core.cpp
  src/core.cu
)

set_target_properties(cugraphopt_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(cugraphopt_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(cugraphopt
  src/main.cpp
)

target_link_libraries(cugraphopt
  PRIVATE
    cugraphopt_core
)

add_executable(cugraphopt_smoke
  tests/smoke_test.cpp
)

target_link_libraries(cugraphopt_smoke
  PRIVATE
    cugraphopt_core
)

add_test(NAME cugraphopt_smoke COMMAND cugraphopt_smoke)
