#pragma once

#include "cugraphopt/bsr.hpp"
#include "cugraphopt/graph_color.hpp"
#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/se3_device.cuh"
#include "cugraphopt/solver.hpp"

namespace cugraphopt {

/// Device-side pose graph data (flat arrays on GPU).
struct DevicePoseGraph {
  int num_nodes;
  int num_edges;

  // Per-node: SE3 poses stored as rotation (9 doubles) + translation (3 doubles)
  double* d_poses;      // num_nodes * 12

  // Per-edge: from/to indices, measurement (12 doubles), info (21 doubles)
  int* d_edge_from;     // num_edges
  int* d_edge_to;       // num_edges
  double* d_edge_meas;  // num_edges * 12 (rotation 9 + translation 3)
  double* d_edge_info;  // num_edges * 21 (upper-triangular info matrix)

  // Per-edge outputs: residual (6), Jacobians (2 * 36)
  double* d_residuals;  // num_edges * 6
  double* d_J_i;        // num_edges * 36
  double* d_J_j;        // num_edges * 36

  // Weighted error per edge (for parallel reduction)
  double* d_errors;     // num_edges
};

/// Device-side BSR matrix.
struct DeviceBSR {
  int num_block_rows;
  int nnz_blocks;

  int* d_row_ptr;    // num_block_rows + 1
  int* d_col_idx;    // nnz_blocks
  double* d_values;  // nnz_blocks * 36

  // BSR block lookup table for fast (i,j) -> block index mapping
  int* d_block_map;  // num_block_rows * num_block_rows (or -1 if not present)
};

/// Allocate and transfer pose graph to device.
DevicePoseGraph create_device_pose_graph(const PoseGraph& graph);

/// Update device poses from host graph (after CPU-side retraction, or for
/// GPU-side state sync).
void update_device_poses(DevicePoseGraph& dpg, const PoseGraph& graph);

/// Free device pose graph memory.
void free_device_pose_graph(DevicePoseGraph& dpg);

/// Allocate device BSR from host BSR (transfers sparsity pattern).
DeviceBSR create_device_bsr(const BSRMatrix& bsr);

/// Free device BSR memory.
void free_device_bsr(DeviceBSR& dbsr);

/// Launch CUDA kernel: compute residuals and Jacobians for all edges.
/// One thread per edge.
void cuda_linearize_edges(DevicePoseGraph& dpg);

/// Launch CUDA kernel: assemble BSR Hessian from per-edge Jacobians.
/// Uses graph coloring: one kernel launch per color, lock-free writes.
void cuda_assemble_colored(DevicePoseGraph& dpg, DeviceBSR& dbsr,
                           double* d_gradient,
                           const EdgeColoring& coloring,
                           const int* d_color_edges, const int* d_color_offsets);

/// Compute total weighted error via parallel reduction.
/// Returns error on host.
double cuda_compute_error(DevicePoseGraph& dpg);

/// BSR SpMV on GPU: y = H * x. Operates on device vectors.
void cuda_bsr_spmv(const DeviceBSR& dbsr, const double* d_x, double* d_y);

/// CUDA vector dot product: returns d_a^T * d_b. Length = dim.
double cuda_dot(const double* d_a, const double* d_b, int dim);

/// CUDA vector operations: d_y = alpha * d_x + d_y.
void cuda_axpy(double alpha, const double* d_x, double* d_y, int dim);

/// CUDA set vector: d_x = val for all elements.
void cuda_fill(double* d_x, double val, int dim);

/// CUDA copy: d_dst = d_src.
void cuda_copy(double* d_dst, const double* d_src, int dim);

/// CUDA scale: d_x *= alpha.
void cuda_scale(double alpha, double* d_x, int dim);

/// Apply Block-Jacobi preconditioner on GPU: z = M^{-1} * r.
/// d_diag_inv: N * 36 doubles (inverted diagonal blocks).
void cuda_precond_apply(const double* d_diag_inv, const double* d_r,
                        double* d_z, int num_blocks);

/// Extract and invert diagonal blocks of BSR on GPU.
/// Returns device pointer to N * 36 doubles.
double* cuda_extract_diagonal_inv(const DeviceBSR& dbsr);

/// Full GPU PCG solve: H * dx = rhs. All on device.
/// Returns number of PCG iterations.
int cuda_pcg_solve(const DeviceBSR& dbsr, const double* d_rhs,
                   double* d_dx, int dim, int max_iter, double tol);

/// Apply gauge fix on device BSR and gradient.
void cuda_gauge_fix(DeviceBSR& dbsr, double* d_gradient);

/// Retract poses on GPU: T_new[i] = T_old[i] * Exp(dx[6*i:6*i+6]).
void cuda_retract(DevicePoseGraph& dpg, const double* d_dx);

/// Full GPU Gauss-Newton solver.
GNResult solve_gauss_newton_gpu(PoseGraph& graph, const GNConfig& config);

}  // namespace cugraphopt
