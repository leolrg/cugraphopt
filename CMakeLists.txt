cmake_minimum_required(VERSION 3.22)

project(CuGraphOpt VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
endif()

include(CTest)

add_library(cugraphopt_core
  src/core.cpp
  src/pose_graph.cpp
  src/se3.cpp
  src/linearization.cpp
  src/solver.cpp
  src/bsr.cpp
  src/graph_color.cpp
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

add_test(
  NAME cugraphopt_cli_parser
  COMMAND cugraphopt ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_pose_graph.g2o
)

set_tests_properties(cugraphopt_cli_parser PROPERTIES
  PASS_REGULAR_EXPRESSION "nodes=2 edges=1"
)

add_executable(cugraphopt_parser_test
  tests/parser_test.cpp
)

target_link_libraries(cugraphopt_parser_test
  PRIVATE
    cugraphopt_core
)

target_compile_definitions(cugraphopt_parser_test
  PRIVATE
    CUGRAPHOPT_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
)

add_test(NAME cugraphopt_parser_test COMMAND cugraphopt_parser_test)

add_executable(cugraphopt_se3_test
  tests/se3_test.cpp
)

target_link_libraries(cugraphopt_se3_test
  PRIVATE
    cugraphopt_core
)

add_test(NAME cugraphopt_se3_test COMMAND cugraphopt_se3_test)

add_executable(cugraphopt_linearization_test
  tests/linearization_test.cpp
)

target_link_libraries(cugraphopt_linearization_test
  PRIVATE
    cugraphopt_core
)

add_test(NAME cugraphopt_linearization_test COMMAND cugraphopt_linearization_test)

add_test(
  NAME cugraphopt_cli_linearize
  COMMAND cugraphopt --linearize ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_pose_graph.g2o
)

set_tests_properties(cugraphopt_cli_linearize PROPERTIES
  PASS_REGULAR_EXPRESSION "nodes=2 edges=1 dim=12 error="
)

add_executable(cugraphopt_solver_test
  tests/solver_test.cpp
)

target_link_libraries(cugraphopt_solver_test
  PRIVATE
    cugraphopt_core
)

target_compile_definitions(cugraphopt_solver_test
  PRIVATE
    CUGRAPHOPT_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    CUGRAPHOPT_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/datasets"
)

add_test(NAME cugraphopt_solver_test COMMAND cugraphopt_solver_test)

add_test(
  NAME cugraphopt_cli_solve
  COMMAND cugraphopt --solve ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/sample_pose_graph.g2o
)

set_tests_properties(cugraphopt_cli_solve PROPERTIES
  PASS_REGULAR_EXPRESSION "converged"
)

add_executable(cugraphopt_bsr_test
  tests/bsr_test.cpp
)

target_link_libraries(cugraphopt_bsr_test
  PRIVATE
    cugraphopt_core
)

add_test(NAME cugraphopt_bsr_test COMMAND cugraphopt_bsr_test)

add_executable(cugraphopt_graph_color_test
  tests/graph_color_test.cpp
)

target_link_libraries(cugraphopt_graph_color_test
  PRIVATE
    cugraphopt_core
)

target_compile_definitions(cugraphopt_graph_color_test
  PRIVATE
    CUGRAPHOPT_DATASET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/datasets"
)

add_test(NAME cugraphopt_graph_color_test COMMAND cugraphopt_graph_color_test)
