# 15-418 Final Project Proposal
## CuGraphOpt: End-to-End CUDA Accelerated Factor Graph Optimization

**URL:** https://leolrg.github.io/cugraphopt/
**Team:** Ruogu (Leo) Li (Solo project, exception requested).

### Summary
We are going to implement a minimal, fully GPU-accelerated non-linear least squares solver (Gauss-Newton) natively in CUDA for SLAM factor graphs. The system will read a pose-graph dataset, perform parallel linearization and sparse matrix assembly on the device, and solve the resulting linear system using a custom CUDA Preconditioned Conjugate Gradient (PCG) solver to minimize host-device data transfer.

### Background
Factor graph optimization is the mathematical backend for modern robotics and SLAM. The core bottleneck involves iteratively evaluating the error and Jacobians for thousands of factors, assembling them into a massive global sparse Hessian matrix (H = J^T J) and gradient vector (b = J^T e), and solving the linear system H Δx = -b. Currently, state-of-the-art libraries perform much of this on the CPU. Moving the entire inner loop to the GPU avoids the severe latency of transferring the Hessian matrix across the PCIe bus every iteration.

### The Challenge
This project presents distinct parallel systems challenges regarding irregular memory access and divergence:
* **Massive Scatter-Add Contention during Assembly:** To build the global sparse Hessian matrix, local Jacobians from each factor must be scattered and accumulated into global memory. Because many factors observe the same state variables, parallelizing over factors results in massive memory write contention. We must evaluate tradeoffs between naive global `atomicAdd` operations, warp-level reductions, and graph-coloring algorithms for safe concurrent writes.
* **Memory-Bound SpMV on Unstructured Block-Sparse Matrices:** Solving the linear system requires a custom Block-Jacobi PCG solver. The core bottleneck is Sparse Matrix-Vector multiplication (SpMV). SLAM graphs create unstructured global sparsity patterns but contain dense local sub-blocks (e.g., 6x6 blocks for 3D poses). The challenge is optimizing a Block-Compressed Sparse Row (BSR) format to maximize cache hit rates and memory bandwidth utilization.
* **Thread Divergence:** Mapping heterogeneous graph nodes to warps naturally leads to execution divergence, which must be mitigated to maintain high SIMD efficiency.

### Resources
I will utilize the NVIDIA GPUs available in the GHC/latedays clusters. The project will be built from scratch in C++ and CUDA, using standard SLAM benchmark datasets (e.g., .g2o files) for input. I will reference the mathematical foundations outlined in literature surrounding GTSAM and Ceres Solver.

### Goals and Deliverables
**Plan to Achieve:**
* A custom CUDA factor graph structure utilizing a Block-Compressed Sparse Row (BSR) format.
* Parallel linearization and assembly kernels using atomic operations.
* A native CUDA PCG solver with a simple Block-Jacobi preconditioner.
* An evaluation comparing our end-to-end execution time against a single-threaded CPU baseline on standard pose-graph datasets.
* A detailed breakdown of the execution time spent in Linearization vs. Assembly vs. Solving.

**Hope to Achieve:**
* Implement graph-coloring to replace atomic adds during the matrix assembly phase and compare the performance.
* Implement the more complex Levenberg-Marquardt trust-region loop instead of just Gauss-Newton.

### Platform Choice
NVIDIA GPUs accessed via CUDA are the ideal target for this workload. The fine-grained control over shared memory, warp-level primitives, and memory coalescing is strictly necessary to effectively map the irregular, block-sparse structures of factor graphs to parallel hardware without crippling the memory bandwidth.

### Schedule
* **Week 1 (March 25 - March 31):** Parse standard SLAM datasets (.g2o files) and design the GPU data structures (Graph representation, BSR Hessian).
* **Week 2 (April 1 - April 7):** Write and test the parallel Linearization and matrix Assembly kernels.
* **Week 3 (April 8 - April 14):** Implement the CUDA PCG solver and Block-Jacobi preconditioner. Complete Milestone Report (Due April 14).
* **Week 4 (April 15 - April 21):** Tie the loop together (state updates, convergence checks). Ensure correctness against a CPU reference.
* **Week 5 (April 22 - April 28):** Profiling, memory optimization (coalescing, shared memory tuning), and generating graphs for the final report. Final Report Due April 30th.
