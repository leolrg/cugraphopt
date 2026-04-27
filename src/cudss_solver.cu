/// GPU Gauss-Newton solver using cuDSS (NVIDIA's GPU sparse direct solver)
/// for the linear system instead of PCG.

#include "cugraphopt/cuda_solver.cuh"
#include "cugraphopt/se3_device.cuh"

#include <chrono>
#include <cstdio>
#include <cudss.h>
#include <vector>

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                   __LINE__, cudaGetErrorString(err));                   \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

#define CUDSS_CHECK(call)                                                \
  do {                                                                   \
    cudssStatus_t s = (call);                                            \
    if (s != CUDSS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "cuDSS error at %s:%d: %d\n", __FILE__,      \
                   __LINE__, (int)s);                                    \
      std::exit(1);                                                      \
    }                                                                    \
  } while (0)

namespace cugraphopt {

// ============================================================================
// BSR -> CSR conversion (expand 6x6 blocks into scalar entries)
// ============================================================================

// Build CSR sparsity pattern from BSR on host.
// For SPD matrix, we store only the lower triangle to save memory and let
// cuDSS use CUDSS_MVIEW_LOWER. But cuDSS also supports full matrices.
// We'll use full CSR for simplicity.
struct CSRMatrix {
  int nrows;        // 6 * num_block_rows
  int ncols;        // same
  int64_t nnz;
  std::vector<int64_t> row_ptr;   // nrows + 1  (int64 for cuDSS)
  std::vector<int64_t> col_idx;   // nnz
  // values stored on device
};

static CSRMatrix bsr_to_csr_pattern(const BSRMatrix& bsr) {
  int N = bsr.num_block_rows;
  int dim = 6 * N;

  CSRMatrix csr;
  csr.nrows = dim;
  csr.ncols = dim;
  csr.row_ptr.resize(dim + 1);

  // Count nnz per scalar row.
  // Each block-row i has some non-zero 6x6 blocks.
  // Each scalar row 6*i+r within that block-row has 6 * (num blocks in row) entries.
  for (int bi = 0; bi < N; ++bi) {
    int blocks_in_row = bsr.row_ptr[bi + 1] - bsr.row_ptr[bi];
    for (int r = 0; r < 6; ++r) {
      csr.row_ptr[6 * bi + r + 1] = 6 * blocks_in_row;
    }
  }

  // Prefix sum.
  csr.row_ptr[0] = 0;
  for (int i = 0; i < dim; ++i) {
    csr.row_ptr[i + 1] += csr.row_ptr[i];
  }
  csr.nnz = csr.row_ptr[dim];

  // Fill col_idx.
  csr.col_idx.resize(csr.nnz);
  for (int bi = 0; bi < N; ++bi) {
    for (int r = 0; r < 6; ++r) {
      int scalar_row = 6 * bi + r;
      int64_t pos = csr.row_ptr[scalar_row];
      for (int k = bsr.row_ptr[bi]; k < bsr.row_ptr[bi + 1]; ++k) {
        int bj = bsr.col_idx[k];
        for (int c = 0; c < 6; ++c) {
          csr.col_idx[pos++] = 6 * bj + c;
        }
      }
    }
  }

  return csr;
}

// CUDA kernel: expand BSR values into CSR values on device.
__global__ void bsr_to_csr_values_kernel(
    const int* __restrict__ bsr_row_ptr,
    const int* __restrict__ bsr_col_idx,
    const double* __restrict__ bsr_values,
    const int* __restrict__ csr_row_ptr,
    double* __restrict__ csr_values,
    int num_block_rows) {
  int bi = blockIdx.x;
  if (bi >= num_block_rows) return;

  int r = threadIdx.x;  // 0..5: which row within the 6x6 block
  if (r >= 6) return;

  int scalar_row = 6 * bi + r;
  int csr_pos = csr_row_ptr[scalar_row];

  for (int k = bsr_row_ptr[bi]; k < bsr_row_ptr[bi + 1]; ++k) {
    const double* blk = bsr_values + k * 36;
    for (int c = 0; c < 6; ++c) {
      csr_values[csr_pos++] = blk[r * 6 + c];
    }
  }
}

// ============================================================================
// cuDSS-based GN solver
// ============================================================================

GNResult solve_gauss_newton_cudss(PoseGraph& graph, const GNConfig& config) {
  using Clock = std::chrono::high_resolution_clock;

  GNResult result{};
  int N = static_cast<int>(graph.nodes.size());
  int dim = 6 * N;

  // Build coloring + BSR pattern on CPU.
  EdgeColoring coloring = color_edges(graph);
  BSRMatrix bsr_host = bsr_symbolic(graph);

  // Build CSR pattern from BSR.
  CSRMatrix csr = bsr_to_csr_pattern(bsr_host);

  if (config.verbose) {
    std::printf("cuDSS-GPU: %d nodes, %d edges, BSR %d blocks, CSR nnz=%lld, %d colors\n",
                N, (int)graph.edges.size(), bsr_host.nnz_blocks,
                (long long)csr.nnz, coloring.num_colors);
  }

  // Transfer pose graph and BSR to device.
  DevicePoseGraph dpg = create_device_pose_graph(graph);
  DeviceBSR dbsr = create_device_bsr(bsr_host);

  // Build flat color-edge list.
  std::vector<int> flat_color_edges;
  for (int c = 0; c < coloring.num_colors; ++c)
    for (int e : coloring.color_edges[c])
      flat_color_edges.push_back(e);
  std::vector<int> color_offsets = {0};
  for (int c = 0; c < coloring.num_colors; ++c)
    color_offsets.push_back(color_offsets.back() +
                            (int)coloring.color_edges[c].size());

  int* d_color_edges;
  int* d_color_offsets;
  CUDA_CHECK(cudaMalloc(&d_color_edges, flat_color_edges.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_color_offsets, color_offsets.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_color_edges, flat_color_edges.data(),
                        flat_color_edges.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_color_offsets, color_offsets.data(),
                        color_offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  // Allocate gradient, dx, rhs on device.
  double* d_gradient;
  double* d_dx;
  double* d_rhs;
  CUDA_CHECK(cudaMalloc(&d_gradient, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dx, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_rhs, dim * sizeof(double)));

  // --- Set up CSR on device for cuDSS ---
  // cuDSS needs int32 indices for CSR on some configurations.
  std::vector<int> h_row_ptr_i32(dim + 1);
  std::vector<int> h_col_idx_i32(csr.nnz);
  for (int i = 0; i <= dim; ++i) h_row_ptr_i32[i] = static_cast<int>(csr.row_ptr[i]);
  for (int64_t i = 0; i < csr.nnz; ++i) h_col_idx_i32[i] = static_cast<int>(csr.col_idx[i]);

  int* d_csr_row_ptr;
  int* d_csr_col_idx;
  double* d_csr_values;
  CUDA_CHECK(cudaMalloc(&d_csr_row_ptr, (dim + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_csr_col_idx, csr.nnz * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_csr_values, csr.nnz * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_csr_row_ptr, h_row_ptr_i32.data(),
                        (dim + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr_col_idx, h_col_idx_i32.data(),
                        csr.nnz * sizeof(int), cudaMemcpyHostToDevice));

  // --- Initialize cuDSS ---
  cudssHandle_t handle;
  cudssConfig_t solverConfig;
  cudssData_t solverData;
  CUDSS_CHECK(cudssCreate(&handle));
  CUDSS_CHECK(cudssConfigCreate(&solverConfig));
  CUDSS_CHECK(cudssDataCreate(handle, &solverData));

  // Set up matrix wrappers.
  cudssMatrix_t matA, matX, matB;
  CUDSS_CHECK(cudssMatrixCreateCsr(
      &matA, dim, dim, csr.nnz,
      d_csr_row_ptr, nullptr, d_csr_col_idx, d_csr_values,
      CUDA_R_32I, CUDA_R_64F,
      CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO));

  CUDSS_CHECK(cudssMatrixCreateDn(&matX, dim, 1, dim, d_dx, CUDA_R_64F,
                                   CUDSS_LAYOUT_COL_MAJOR));
  CUDSS_CHECK(cudssMatrixCreateDn(&matB, dim, 1, dim, d_rhs, CUDA_R_64F,
                                   CUDSS_LAYOUT_COL_MAJOR));

  // Analysis phase (symbolic factorization) — done once since pattern is fixed.
  // First need to fill CSR values so cuDSS can do analysis.
  // We'll do a dummy linearize + assemble to get initial values.
  cuda_linearize_edges_analytical(dpg);
  cuda_assemble_colored(dpg, dbsr, d_gradient, coloring,
                        d_color_edges, d_color_offsets);
  cuda_gauge_fix(dbsr, d_gradient);

  // Expand BSR to CSR values.
  bsr_to_csr_values_kernel<<<N, 6>>>(
      dbsr.d_row_ptr, dbsr.d_col_idx, dbsr.d_values,
      d_csr_row_ptr, d_csr_values, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Run analysis (reordering + symbolic factorization).
  auto t_analysis_start = Clock::now();
  CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                            solverData, matA, matX, matB));
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t_analysis_end = Clock::now();
  double analysis_ms = std::chrono::duration<double, std::milli>(
      t_analysis_end - t_analysis_start).count();

  if (config.verbose) {
    std::printf("cuDSS analysis: %.1f ms\n", analysis_ms);
  }

  // --- GN iteration loop ---
  for (int iter = 0; iter < config.max_iterations; ++iter) {
    GNIterationStats stats{};
    stats.iteration = iter;
    auto t_total_start = Clock::now();

    // --- Linearize + assemble ---
    auto t0 = Clock::now();
    cuda_linearize_edges_analytical(dpg);
    cuda_assemble_colored(dpg, dbsr, d_gradient, coloring,
                          d_color_edges, d_color_offsets);
    auto t1 = Clock::now();

    double total_error = cuda_compute_error(dpg);
    if (iter == 0) result.initial_error = total_error;
    stats.error = total_error;

    // Check gradient.
    std::vector<double> h_gradient(dim);
    CUDA_CHECK(cudaMemcpy(h_gradient.data(), d_gradient, dim * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double grad_max = 0.0;
    for (int i = 0; i < dim; ++i) grad_max = std::max(grad_max, std::abs(h_gradient[i]));

    stats.linearize_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (config.verbose) {
      std::printf("iter %3d  error=%.6e  |grad|=%.2e  lin+asm=%.1fms",
                  iter, total_error, grad_max, stats.linearize_ms);
    }

    if (grad_max < config.gradient_tolerance) {
      if (config.verbose) std::printf("  -> converged (gradient)\n");
      result.final_error = total_error;
      result.iterations = iter;
      result.stats.push_back(stats);
      break;
    }

    // --- Gauge fix ---
    cuda_gauge_fix(dbsr, d_gradient);

    // --- Expand BSR to CSR values ---
    bsr_to_csr_values_kernel<<<N, 6>>>(
        dbsr.d_row_ptr, dbsr.d_col_idx, dbsr.d_values,
        d_csr_row_ptr, d_csr_values, N);

    // --- rhs = -gradient ---
    CUDA_CHECK(cudaMemcpy(d_rhs, d_gradient, dim * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    cuda_scale(-1.0, d_rhs, dim);

    // --- cuDSS factorization + solve ---
    auto t2 = Clock::now();

    // Use REFACTORIZATION for iter > 0 (same pattern, new values).
    if (iter == 0) {
      CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig,
                                solverData, matA, matX, matB));
    } else {
      CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_REFACTORIZATION, solverConfig,
                                solverData, matA, matX, matB));
    }

    CUDSS_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                              solverData, matA, matX, matB));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t3 = Clock::now();
    stats.solve_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- Retract ---
    auto t4 = Clock::now();
    cuda_retract(dpg, d_dx);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t5 = Clock::now();
    stats.retract_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    auto t_total_end = Clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start).count();

    if (config.verbose) {
      std::printf("  solve=%.1fms  retract=%.1fms  total=%.1fms\n",
                  stats.solve_ms, stats.retract_ms, stats.total_ms);
    }

    result.stats.push_back(stats);
    result.final_error = total_error;
    result.iterations = iter + 1;

    // Error convergence check.
    if (iter > 0) {
      double prev_error = result.stats[iter - 1].error;
      double rel_change = std::abs(total_error - prev_error) /
                          (std::abs(prev_error) + 1e-30);
      if (rel_change < config.error_tolerance) {
        if (config.verbose) {
          std::printf("  -> converged (error stagnation, rel=%.2e)\n", rel_change);
        }
        break;
      }
    }
  }

  // Read back optimized poses.
  // (reuse the read_back function from cuda_solver.cu via extern linkage)
  // For now, do it inline.
  {
    std::vector<double> h_poses(N * 12);
    CUDA_CHECK(cudaMemcpy(h_poses.data(), dpg.d_poses,
                          N * 12 * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
      auto& n = graph.nodes[i];
      const double* p = h_poses.data() + i * 12;
      n.x = p[9]; n.y = p[10]; n.z = p[11];
      double trace = p[0] + p[4] + p[8];
      double qw, qx, qy, qz;
      if (trace > 0.0) {
        double s = 0.5 / std::sqrt(trace + 1.0);
        qw = 0.25 / s; qx = (p[7]-p[5])*s; qy = (p[2]-p[6])*s; qz = (p[3]-p[1])*s;
      } else if (p[0] > p[4] && p[0] > p[8]) {
        double s = 2.0 * std::sqrt(1.0 + p[0] - p[4] - p[8]);
        qw = (p[7]-p[5])/s; qx = 0.25*s; qy = (p[1]+p[3])/s; qz = (p[2]+p[6])/s;
      } else if (p[4] > p[8]) {
        double s = 2.0 * std::sqrt(1.0 + p[4] - p[0] - p[8]);
        qw = (p[2]-p[6])/s; qx = (p[1]+p[3])/s; qy = 0.25*s; qz = (p[5]+p[7])/s;
      } else {
        double s = 2.0 * std::sqrt(1.0 + p[8] - p[0] - p[4]);
        qw = (p[3]-p[1])/s; qx = (p[2]+p[6])/s; qy = (p[5]+p[7])/s; qz = 0.25*s;
      }
      double norm = std::sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
      n.qx = qx/norm; n.qy = qy/norm; n.qz = qz/norm; n.qw = qw/norm;
    }
  }

  // Cleanup.
  cudssMatrixDestroy(matA);
  cudssMatrixDestroy(matX);
  cudssMatrixDestroy(matB);
  cudssDataDestroy(handle, solverData);
  cudssConfigDestroy(solverConfig);
  cudssDestroy(handle);
  cudaFree(d_gradient); cudaFree(d_dx); cudaFree(d_rhs);
  cudaFree(d_csr_row_ptr); cudaFree(d_csr_col_idx); cudaFree(d_csr_values);
  cudaFree(d_color_edges); cudaFree(d_color_offsets);
  free_device_pose_graph(dpg);
  free_device_bsr(dbsr);

  return result;
}

}  // namespace cugraphopt
