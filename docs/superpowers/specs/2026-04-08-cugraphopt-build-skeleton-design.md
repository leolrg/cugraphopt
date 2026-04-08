# CuGraphOpt Build Skeleton Design

Date: 2026-04-08

## Goal

Create the initial repository scaffold for CuGraphOpt with CUDA enabled in CMake from the start. The scaffold should build cleanly, provide a stable place for future parser and solver code, and include a minimal verification path.

## Scope

This design covers only the initial build skeleton:

- Top-level CMake configuration with CUDA enabled
- Public include layout under `include/cugraphopt/`
- Source layout under `src/`
- Minimal executable target
- Core library target that can contain both C++ and CUDA sources
- Minimal smoke test registered with CTest

This design does not include:

- `.g2o` parsing
- Pose graph data structures beyond placeholders
- Linearization or Hessian assembly
- PCG or preconditioning
- GTSAM integration

## Recommended Approach

Use a two-target structure:

- `cugraphopt_core`: a reusable library compiled with both C++ and CUDA enabled
- `cugraphopt`: a thin executable linked against the core library

This is preferred over a single mixed executable because it keeps the future parser, math, and GPU kernels isolated from application entry-point code. It is also preferred over splitting CPU and GPU into separate libraries because that boundary is premature at the current repo size.

## Project Structure

Planned layout:

- `CMakeLists.txt`
- `include/cugraphopt/`
- `src/`
- `tests/`
- `datasets/`

Expected initial files:

- `include/cugraphopt/version.hpp`
- `include/cugraphopt/core.hpp`
- `src/core.cpp`
- `src/core.cu`
- `src/main.cpp`
- `tests/smoke_test.cpp`

## Build Design

The root `CMakeLists.txt` will:

- declare `project(CuGraphOpt LANGUAGES CXX CUDA)`
- require C++17 for host code and set the CUDA language standard to 17 for `.cu` compilation units; this refers to `CMAKE_CXX_STANDARD` and `CMAKE_CUDA_STANDARD`, not a CUDA Toolkit version
- enable testing via `include(CTest)`
- define a default CUDA architecture list only when the user has not already provided one
- build `cugraphopt_core` as the primary library target
- build `cugraphopt` as a small executable target linked to `cugraphopt_core`
- build `cugraphopt_smoke` as a minimal test executable

The core library will enable CUDA separable compilation so future kernels can grow without reworking the build setup.

## Initial Interfaces

The initial library surface should stay minimal:

- a version/query header for compile-time constants
- a simple function callable from the executable and smoke test
- a trivial CUDA-backed function to prove host code can link against device-compiled translation units

The point is build validation, not feature completeness.

## Verification

The first verification path should be:

1. Configure with CMake in a `build/` directory
2. Build all targets
3. Run `ctest`
4. Optionally run the main executable directly

Success means:

- CMake configures without CUDA toolchain errors
- the mixed C++/CUDA targets compile and link
- the smoke test passes

## Risks and Mitigations

### CUDA architecture mismatch

Risk:
CUDA builds can fail or produce suboptimal binaries if no architecture is specified.

Mitigation:
Set a conservative default `CMAKE_CUDA_ARCHITECTURES` only when the user has not provided one explicitly.

### Premature build complexity

Risk:
Adding too much structure in the initial scaffold slows down subsequent parser work.

Mitigation:
Keep the first target graph small: one library, one app, one test.

### Unclear boundary between CPU and GPU code

Risk:
Future source placement becomes inconsistent if the scaffold does not establish a pattern now.

Mitigation:
Keep public interfaces in `include/cugraphopt/` and implementation files in `src/`, with CPU and CUDA files coexisting behind the same library target until clearer separation is justified.

## Implementation Notes

The first implementation should avoid external dependencies and keep all code self-contained. The next milestone after this scaffold is a `.g2o` parser and graph representation built on top of the core library target.
