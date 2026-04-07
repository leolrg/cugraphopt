# 15-418 Final Project Proposal
## CuGraphOpt: End-to-End CUDA Accelerated Factor Graph Optimization

**URL:** https://leolrg.github.io/cugraphopt/
**Team:** leolrg (Solo project, exception requested).

### Summary
We are going to implement a minimal, fully GPU-accelerated non-linear least squares solver (Gauss-Newton) natively in CUDA, strictly scoped to 3D Pose Graph Optimization (PGO). The system will read standard pose-graph datasets, perform parallel linearization and sparse matrix assembly on the device, and solve the resulting linear system using a custom CUDA Preconditioned Conjugate Gradient (PCG) solver to minimize host-device data transfer.

### Background
Factor graph optimization is the mathematical backend for modern robotics and SLAM. The core bottleneck involves iteratively evaluating the error and Jacobians for thousands of constraints, assembling them into a massive global sparse Hessian matrix ($H = J^T J$) and gradient vector ($b = J^T e$), and solving the linear system $H \Delta x = -b$. Currently, state-of-the-art libraries perform much of this on the CPU. Moving the entire inner loop to the GPU avoids the severe latency of transferring the Hessian matrix across the PCIe bus every iteration. 

By narrowing the scope to PGO, every variable is uniformly structured as a 3D pose, allowing us to focus on the parallel systems challenges of memory coalescing and contention rather than handling dynamic, heterogeneous sensor types.

### The Challenge
This project presents distinct parallel systems challenges regarding irregular memory access and divergence:
* **Massive Scatter-Add Contention during Assembly:** To build the global sparse Hessian matrix, tiny local Jacobians computed from each factor must be scattered and accumulated into global memory. Because many factors observe the same state variables (especially during loop closures), parallelizing over factors results in massive memory write contention. We must evaluate tradeoffs between naive global `atomicAdd` operations and CPU-side graph-coloring algorithms for lock-free concurrent writes.
* **Memory-Bound SpMV on Unstructured Block-Sparse Matrices:** Solving the linear system requires a custom Block-Jacobi PCG solver. The core bottleneck is Sparse Matrix-Vector multiplication (SpMV). SLAM graphs create unstructured global sparsity patterns but contain dense local sub-blocks (specifically, $6 \times 6$ blocks for $SE(3)$ poses). The challenge is optimizing a Block-Compressed Sparse Row (BSR) format to maximize cache hit rates and memory bandwidth utilization.
* **Thread Divergence:** Mapping graph nodes and edge factors to warps naturally leads to execution divergence, which must be mitigated using warp-level reduction primitives to maintain high SIMD efficiency during the iterative solve.

### Resources
I will utilize the NVIDIA GPUs available in the GHC/latedays clusters. The project will be built from scratch in C++ and CUDA. 

For benchmarking and ground-truth validation, I will use **GTSAM** as the single-threaded CPU baseline. I will parse standard PGO benchmark datasets (e.g., `sphere2500.g2o`, `parking-garage.g2o`, `torus3D.g2o`, `M3500.g2o`) to feed the graph topology and initial estimates directly into both solvers. 

### Goals and Deliverables
**Plan to Achieve:**
* A custom CUDA pose-graph structure utilizing a Block-Compressed Sparse Row (BSR) format tailored for $6 \times 6$ matrix blocks.
* A baseline parallel linearization and assembly kernel using naive `atomicAdd` operations.
* An optimized assembly kernel utilizing CPU-computed **graph-coloring** to guarantee lock-free, zero-contention writes to global VRAM.
* A native CUDA PCG solver utilizing warp-level primitives and a simple Block-Jacobi preconditioner.
* An evaluation comparing our end-to-end execution time against GTSAM's CPU batch solver on datasets with varying topological density (e.g., dense spheres vs. sparse real-world garages).
* A detailed breakdown of the execution time spent in Linearization vs. Assembly vs. Solving, highlighting the performance impact of our memory access tricks.

**Hope to Achieve:**
* Implement the more complex Levenberg-Marquardt trust-region loop instead of just Gauss-Newton.
* Compare memory throughput of standard CSR SpMV against our custom BSR SpMV.

### Platform Choice
NVIDIA GPUs accessed via CUDA are the ideal target for this workload. The fine-grained control over shared memory, warp-level primitives, and memory coalescing is strictly necessary to effectively map the irregular, block-sparse structures of factor graphs to parallel hardware without crippling the memory bandwidth.

### Schedule
* **Week 1 (March 25 - March 31):** Implement GTSAM CPU baseline for `.g2o` parsing, outputting ground-truth matrices, and designing the initial GPU data structures (Graph representation, BSR Hessian).
* **Week 2 (April 1 - April 7):** Write and test the parallel Linearization kernels and the initial naive `atomicAdd` matrix Assembly kernel. 
* **Week 3 (April 8 - April 14):** Implement the CPU graph-coloring pre-computation and the lock-free CUDA assembly kernels. Complete Milestone Report (Due April 14).
* **Week 4 (April 15 - April 21):** Implement the CUDA PCG solver and Block-Jacobi preconditioner. Tie the loop together (state updates on the manifold, convergence checks). Ensure correctness against the GTSAM reference.
* **Week 5 (April 22 - April 28):** Profiling (Nsight Compute), memory optimization (coalescing, shared memory tuning), and generating graphs for the final report. Final Report Due April 30th.
