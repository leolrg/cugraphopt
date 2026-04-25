#include "cugraphopt/cuda_solver.cuh"
#include "cugraphopt/se3_device.cuh"

#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,    \
                   __LINE__, cudaGetErrorString(err));                \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

namespace cugraphopt {

using namespace device;

// ============================================================================
// Device data transfer
// ============================================================================

DevicePoseGraph create_device_pose_graph(const PoseGraph& graph) {
  DevicePoseGraph dpg;
  dpg.num_nodes = static_cast<int>(graph.nodes.size());
  dpg.num_edges = static_cast<int>(graph.edges.size());

  // Map node IDs to sequential indices.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < dpg.num_nodes; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  // Poses: rotation (9) + translation (3) = 12 doubles per node.
  std::vector<double> h_poses(dpg.num_nodes * 12);
  for (int i = 0; i < dpg.num_nodes; ++i) {
    const auto& n = graph.nodes[i];
    DSO3 R = dquat_to_SO3(n.qx, n.qy, n.qz, n.qw);
    for (int j = 0; j < 9; ++j) h_poses[i * 12 + j] = R.R.m[j];
    h_poses[i * 12 + 9] = n.x;
    h_poses[i * 12 + 10] = n.y;
    h_poses[i * 12 + 11] = n.z;
  }
  CUDA_CHECK(cudaMalloc(&dpg.d_poses, dpg.num_nodes * 12 * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(dpg.d_poses, h_poses.data(),
                        dpg.num_nodes * 12 * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Edges: from/to, measurement (12), info (21).
  std::vector<int> h_from(dpg.num_edges), h_to(dpg.num_edges);
  std::vector<double> h_meas(dpg.num_edges * 12);
  std::vector<double> h_info(dpg.num_edges * 21);

  for (int e = 0; e < dpg.num_edges; ++e) {
    const auto& edge = graph.edges[e];
    h_from[e] = id_to_idx.at(edge.from);
    h_to[e] = id_to_idx.at(edge.to);

    DSO3 R = dquat_to_SO3(edge.qx, edge.qy, edge.qz, edge.qw);
    for (int j = 0; j < 9; ++j) h_meas[e * 12 + j] = R.R.m[j];
    h_meas[e * 12 + 9] = edge.x;
    h_meas[e * 12 + 10] = edge.y;
    h_meas[e * 12 + 11] = edge.z;

    for (int j = 0; j < 21; ++j) h_info[e * 21 + j] = edge.information[j];
  }

  CUDA_CHECK(cudaMalloc(&dpg.d_edge_from, dpg.num_edges * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dpg.d_edge_to, dpg.num_edges * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dpg.d_edge_meas, dpg.num_edges * 12 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dpg.d_edge_info, dpg.num_edges * 21 * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dpg.d_edge_from, h_from.data(),
                        dpg.num_edges * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpg.d_edge_to, h_to.data(),
                        dpg.num_edges * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpg.d_edge_meas, h_meas.data(),
                        dpg.num_edges * 12 * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpg.d_edge_info, h_info.data(),
                        dpg.num_edges * 21 * sizeof(double), cudaMemcpyHostToDevice));

  // Per-edge output buffers.
  CUDA_CHECK(cudaMalloc(&dpg.d_residuals, dpg.num_edges * 6 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dpg.d_J_i, dpg.num_edges * 36 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dpg.d_J_j, dpg.num_edges * 36 * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dpg.d_errors, dpg.num_edges * sizeof(double)));

  return dpg;
}

void update_device_poses(DevicePoseGraph& dpg, const PoseGraph& graph) {
  std::vector<double> h_poses(dpg.num_nodes * 12);
  for (int i = 0; i < dpg.num_nodes; ++i) {
    const auto& n = graph.nodes[i];
    DSO3 R = dquat_to_SO3(n.qx, n.qy, n.qz, n.qw);
    for (int j = 0; j < 9; ++j) h_poses[i * 12 + j] = R.R.m[j];
    h_poses[i * 12 + 9] = n.x;
    h_poses[i * 12 + 10] = n.y;
    h_poses[i * 12 + 11] = n.z;
  }
  CUDA_CHECK(cudaMemcpy(dpg.d_poses, h_poses.data(),
                        dpg.num_nodes * 12 * sizeof(double),
                        cudaMemcpyHostToDevice));
}

void free_device_pose_graph(DevicePoseGraph& dpg) {
  cudaFree(dpg.d_poses);
  cudaFree(dpg.d_edge_from);
  cudaFree(dpg.d_edge_to);
  cudaFree(dpg.d_edge_meas);
  cudaFree(dpg.d_edge_info);
  cudaFree(dpg.d_residuals);
  cudaFree(dpg.d_J_i);
  cudaFree(dpg.d_J_j);
  cudaFree(dpg.d_errors);
}

DeviceBSR create_device_bsr(const BSRMatrix& bsr) {
  DeviceBSR dbsr;
  dbsr.num_block_rows = bsr.num_block_rows;
  dbsr.nnz_blocks = bsr.nnz_blocks;

  CUDA_CHECK(cudaMalloc(&dbsr.d_row_ptr, (bsr.num_block_rows + 1) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dbsr.d_col_idx, bsr.nnz_blocks * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dbsr.d_values, bsr.nnz_blocks * 36 * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(dbsr.d_row_ptr, bsr.row_ptr.data(),
                        (bsr.num_block_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbsr.d_col_idx, bsr.col_idx.data(),
                        bsr.nnz_blocks * sizeof(int), cudaMemcpyHostToDevice));

  // Build block lookup map on host.
  int N = bsr.num_block_rows;
  std::vector<int> block_map(N * N, -1);
  for (int i = 0; i < N; ++i) {
    for (int k = bsr.row_ptr[i]; k < bsr.row_ptr[i + 1]; ++k) {
      block_map[i * N + bsr.col_idx[k]] = k;
    }
  }
  CUDA_CHECK(cudaMalloc(&dbsr.d_block_map, N * N * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dbsr.d_block_map, block_map.data(),
                        N * N * sizeof(int), cudaMemcpyHostToDevice));

  return dbsr;
}

void free_device_bsr(DeviceBSR& dbsr) {
  cudaFree(dbsr.d_row_ptr);
  cudaFree(dbsr.d_col_idx);
  cudaFree(dbsr.d_values);
  cudaFree(dbsr.d_block_map);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

// Load an SE3 pose from the flat device array.
__device__ DSE3 load_pose(const double* poses, int idx) {
  DSE3 T;
  const double* p = poses + idx * 12;
  for (int i = 0; i < 9; ++i) T.R.R.m[i] = p[i];
  T.t[0] = p[9]; T.t[1] = p[10]; T.t[2] = p[11];
  return T;
}

// Load an edge measurement from the flat device array.
__device__ DSE3 load_measurement(const double* meas, int idx) {
  DSE3 Z;
  const double* p = meas + idx * 12;
  for (int i = 0; i < 9; ++i) Z.R.R.m[i] = p[i];
  Z.t[0] = p[9]; Z.t[1] = p[10]; Z.t[2] = p[11];
  return Z;
}

// ---- Linearization kernel: one thread per edge ----------------------------

__global__ void linearize_edges_kernel(
    const double* __restrict__ poses,
    const int* __restrict__ edge_from,
    const int* __restrict__ edge_to,
    const double* __restrict__ edge_meas,
    const double* __restrict__ edge_info,
    double* __restrict__ residuals,
    double* __restrict__ J_i_out,
    double* __restrict__ J_j_out,
    double* __restrict__ errors,
    int num_edges) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_edges) return;

  int i = edge_from[e];
  int j = edge_to[e];

  DSE3 T_i = load_pose(poses, i);
  DSE3 T_j = load_pose(poses, j);
  DSE3 Z_ij = load_measurement(edge_meas, e);

  // Compute residual.
  DVec6 r = dcompute_residual(T_i, T_j, Z_ij);
  for (int d = 0; d < 6; ++d) residuals[e * 6 + d] = r[d];

  // Compute Jacobians.
  DMat6 Ji, Jj;
  dcompute_jacobians(T_i, T_j, Z_ij, Ji, Jj);
  for (int a = 0; a < 36; ++a) J_i_out[e * 36 + a] = Ji.m[a];
  for (int a = 0; a < 36; ++a) J_j_out[e * 36 + a] = Jj.m[a];

  // Compute weighted error: r^T Omega r.
  DMat6 omega;
  dexpand_information(edge_info + e * 21, omega);
  DVec6 omega_r = dmat6_vec(omega, r);
  double err = 0.0;
  for (int d = 0; d < 6; ++d) err += r[d] * omega_r[d];
  errors[e] = err;
}

void cuda_linearize_edges(DevicePoseGraph& dpg) {
  int threads = 256;
  int blocks = (dpg.num_edges + threads - 1) / threads;
  linearize_edges_kernel<<<blocks, threads>>>(
      dpg.d_poses, dpg.d_edge_from, dpg.d_edge_to,
      dpg.d_edge_meas, dpg.d_edge_info,
      dpg.d_residuals, dpg.d_J_i, dpg.d_J_j, dpg.d_errors,
      dpg.num_edges);
  CUDA_CHECK(cudaGetLastError());
}

// ---- Analytical linearization kernel (no finite differences) ---------------

__global__ void linearize_edges_analytical_kernel(
    const double* __restrict__ poses,
    const int* __restrict__ edge_from,
    const int* __restrict__ edge_to,
    const double* __restrict__ edge_meas,
    const double* __restrict__ edge_info,
    double* __restrict__ residuals,
    double* __restrict__ J_i_out,
    double* __restrict__ J_j_out,
    double* __restrict__ errors,
    int num_edges) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_edges) return;

  int i = edge_from[e];
  int j = edge_to[e];

  DSE3 T_i = load_pose(poses, i);
  DSE3 T_j = load_pose(poses, j);
  DSE3 Z_ij = load_measurement(edge_meas, e);

  // Compute residual.
  DVec6 r = dcompute_residual(T_i, T_j, Z_ij);
  for (int d = 0; d < 6; ++d) residuals[e * 6 + d] = r[d];

  // Compute analytical Jacobians.
  DMat6 Ji, Jj;
  dcompute_jacobians_analytical(Z_ij, r, Ji, Jj);
  for (int a = 0; a < 36; ++a) J_i_out[e * 36 + a] = Ji.m[a];
  for (int a = 0; a < 36; ++a) J_j_out[e * 36 + a] = Jj.m[a];

  // Compute weighted error.
  DMat6 omega;
  dexpand_information(edge_info + e * 21, omega);
  DVec6 omega_r = dmat6_vec(omega, r);
  double err = 0.0;
  for (int d = 0; d < 6; ++d) err += r[d] * omega_r[d];
  errors[e] = err;
}

static void cuda_linearize_edges_analytical(DevicePoseGraph& dpg) {
  int threads = 256;
  int blocks = (dpg.num_edges + threads - 1) / threads;
  linearize_edges_analytical_kernel<<<blocks, threads>>>(
      dpg.d_poses, dpg.d_edge_from, dpg.d_edge_to,
      dpg.d_edge_meas, dpg.d_edge_info,
      dpg.d_residuals, dpg.d_J_i, dpg.d_J_j, dpg.d_errors,
      dpg.num_edges);
  CUDA_CHECK(cudaGetLastError());
}

// ---- Assembly kernel: one thread per edge, process one color at a time ----

__global__ void assemble_edges_kernel(
    const int* __restrict__ color_edge_list,
    int num_color_edges,
    const int* __restrict__ edge_from,
    const int* __restrict__ edge_to,
    const double* __restrict__ residuals,
    const double* __restrict__ J_i_all,
    const double* __restrict__ J_j_all,
    const double* __restrict__ edge_info,
    double* __restrict__ bsr_values,
    double* __restrict__ gradient,
    const int* __restrict__ block_map,
    int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_color_edges) return;

  int e = color_edge_list[idx];
  int i = edge_from[e];
  int j = edge_to[e];

  // Load Jacobians.
  DMat6 Ji, Jj;
  for (int a = 0; a < 36; ++a) Ji.m[a] = J_i_all[e * 36 + a];
  for (int a = 0; a < 36; ++a) Jj.m[a] = J_j_all[e * 36 + a];

  // Load residual.
  DVec6 r;
  for (int d = 0; d < 6; ++d) r[d] = residuals[e * 6 + d];

  // Load information matrix.
  DMat6 omega;
  dexpand_information(edge_info + e * 21, omega);

  // Precompute Omega * J_i, Omega * J_j.
  DMat6 omega_Ji = dmat6_multiply(omega, Ji);
  DMat6 omega_Jj = dmat6_multiply(omega, Jj);
  DVec6 omega_r = dmat6_vec(omega, r);

  // H blocks: J^T * Omega * J.
  // Since this is lock-free (one color at a time), direct writes.
  auto write_block = [&](int bi, int bj, const DMat6& block) {
    int k = block_map[bi * N + bj];
    double* dst = bsr_values + k * 36;
    for (int a = 0; a < 36; ++a) dst[a] += block.m[a];
  };

  // J_i^T * Omega * J_i
  DMat6 JitOJi = dmat6_multiply(dmat6_transpose(Ji), omega_Ji);
  write_block(i, i, JitOJi);

  // J_i^T * Omega * J_j
  DMat6 JitOJj = dmat6_multiply(dmat6_transpose(Ji), omega_Jj);
  write_block(i, j, JitOJj);

  // J_j^T * Omega * J_i
  DMat6 JjtOJi = dmat6_multiply(dmat6_transpose(Jj), omega_Ji);
  write_block(j, i, JjtOJi);

  // J_j^T * Omega * J_j
  DMat6 JjtOJj = dmat6_multiply(dmat6_transpose(Jj), omega_Jj);
  write_block(j, j, JjtOJj);

  // Gradient: b_i += J_i^T * Omega * r, b_j += J_j^T * Omega * r.
  for (int d = 0; d < 6; ++d) {
    double gi = 0.0, gj = 0.0;
    for (int k = 0; k < 6; ++k) {
      gi += Ji.m[k * 6 + d] * omega_r[k];
      gj += Jj.m[k * 6 + d] * omega_r[k];
    }
    gradient[6 * i + d] += gi;
    gradient[6 * j + d] += gj;
  }
}

void cuda_assemble_colored(DevicePoseGraph& dpg, DeviceBSR& dbsr,
                           double* d_gradient,
                           const EdgeColoring& coloring,
                           const int* d_color_edges, const int* d_color_offsets) {
  int N = dpg.num_nodes;

  // Zero BSR values and gradient.
  CUDA_CHECK(cudaMemset(dbsr.d_values, 0, dbsr.nnz_blocks * 36 * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_gradient, 0, N * 6 * sizeof(double)));

  int threads = 256;
  int offset = 0;
  for (int c = 0; c < coloring.num_colors; ++c) {
    int num = static_cast<int>(coloring.color_edges[c].size());
    if (num == 0) continue;
    int blocks = (num + threads - 1) / threads;

    assemble_edges_kernel<<<blocks, threads>>>(
        d_color_edges + offset, num,
        dpg.d_edge_from, dpg.d_edge_to,
        dpg.d_residuals, dpg.d_J_i, dpg.d_J_j, dpg.d_edge_info,
        dbsr.d_values, d_gradient, dbsr.d_block_map, N);
    CUDA_CHECK(cudaGetLastError());

    // Sync between color classes to avoid races.
    CUDA_CHECK(cudaDeviceSynchronize());

    offset += num;
  }
}

// ---- Warp-level reduction -------------------------------------------------

__device__ double warp_reduce_sum_impl(double val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// ---- Error reduction ------------------------------------------------------

__global__ void sum_reduce_kernel(const double* __restrict__ input,
                                   double* output, int n) {
  double sum = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; i < n; i += stride) {
    sum += input[i];
  }

  sum = warp_reduce_sum_impl(sum);

  __shared__ double warp_sums[8];
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  if (lane == 0) warp_sums[warp_id] = sum;
  __syncthreads();

  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    sum = (lane < num_warps) ? warp_sums[lane] : 0.0;
    sum = warp_reduce_sum_impl(sum);
    if (lane == 0) atomicAdd(output, sum);
  }
}

double cuda_compute_error(DevicePoseGraph& dpg) {
  double zero = 0.0;
  double* d_total;
  CUDA_CHECK(cudaMalloc(&d_total, sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_total, &zero, sizeof(double), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (dpg.num_edges + threads - 1) / threads;
  sum_reduce_kernel<<<blocks, threads>>>(
      dpg.d_errors, d_total, dpg.num_edges);
  CUDA_CHECK(cudaGetLastError());

  double total;
  CUDA_CHECK(cudaMemcpy(&total, d_total, sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_total);
  return total;
}

// ---- BSR SpMV kernel: one warp per block-row ------------------------------

// BSR SpMV: 6 threads per block-row for full utilization.
// Block size = 192 = 32 virtual rows * 6 threads/row.
// Each group of 6 threads computes one row of y = H * x.
__global__ void bsr_spmv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_block_rows) {
  // 6 threads per row: threadIdx.x % 6 = element, threadIdx.x / 6 = local row
  int elem = threadIdx.x % 6;
  int local_row = threadIdx.x / 6;
  int row = blockIdx.x * (blockDim.x / 6) + local_row;
  if (row >= num_block_rows) return;

  double acc = 0.0;

  int start = row_ptr[row];
  int end = row_ptr[row + 1];

  for (int k = start; k < end; ++k) {
    int col = col_idx[k];
    const double* blk = values + k * 36;
    const double* xj = x + col * 6;

    // This thread computes row 'elem' of the 6x6 block-vector product.
    double s = 0.0;
    for (int c = 0; c < 6; ++c) {
      s += blk[elem * 6 + c] * xj[c];
    }
    acc += s;
  }

  y[row * 6 + elem] = acc;
}

void cuda_bsr_spmv(const DeviceBSR& dbsr, const double* d_x, double* d_y) {
  // 192 threads per block = 32 rows per block, 6 threads per row.
  const int threads = 192;
  const int rows_per_block = threads / 6;
  int blocks = (dbsr.num_block_rows + rows_per_block - 1) / rows_per_block;
  bsr_spmv_kernel<<<blocks, threads>>>(
      dbsr.d_row_ptr, dbsr.d_col_idx, dbsr.d_values, d_x, d_y,
      dbsr.num_block_rows);
  CUDA_CHECK(cudaGetLastError());
}

// ---- Vector operations ----------------------------------------------------

__global__ void dot_kernel(const double* __restrict__ a,
                           const double* __restrict__ b,
                           double* result, int n) {
  double sum = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Grid-stride loop for large arrays.
  for (; i < n; i += stride) {
    sum += a[i] * b[i];
  }

  // Warp-level reduction.
  sum = warp_reduce_sum_impl(sum);

  // Write warp results to shared memory.
  __shared__ double warp_sums[8]; // max 256 threads / 32 = 8 warps
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  if (lane == 0) warp_sums[warp_id] = sum;
  __syncthreads();

  // First warp reduces all warp sums.
  if (warp_id == 0) {
    int num_warps = blockDim.x / 32;
    sum = (lane < num_warps) ? warp_sums[lane] : 0.0;
    sum = warp_reduce_sum_impl(sum);
    if (lane == 0) atomicAdd(result, sum);
  }
}

double cuda_dot(const double* d_a, const double* d_b, int dim) {
  double zero = 0.0;
  double* d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_result, &zero, sizeof(double), cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  dot_kernel<<<blocks, threads>>>(
      d_a, d_b, d_result, dim);
  CUDA_CHECK(cudaGetLastError());

  double result;
  CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
  cudaFree(d_result);
  return result;
}

__global__ void axpy_kernel(double alpha, const double* __restrict__ x,
                            double* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += alpha * x[i];
}

void cuda_axpy(double alpha, const double* d_x, double* d_y, int dim) {
  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  axpy_kernel<<<blocks, threads>>>(alpha, d_x, d_y, dim);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void fill_kernel(double* __restrict__ x, double val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = val;
}

void cuda_fill(double* d_x, double val, int dim) {
  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  fill_kernel<<<blocks, threads>>>(d_x, val, dim);
  CUDA_CHECK(cudaGetLastError());
}

void cuda_copy(double* d_dst, const double* d_src, int dim) {
  CUDA_CHECK(cudaMemcpy(d_dst, d_src, dim * sizeof(double),
                        cudaMemcpyDeviceToDevice));
}

__global__ void scale_kernel(double alpha, double* __restrict__ x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= alpha;
}

void cuda_scale(double alpha, double* d_x, int dim) {
  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  scale_kernel<<<blocks, threads>>>(alpha, d_x, dim);
  CUDA_CHECK(cudaGetLastError());
}

// ---- Block-Jacobi preconditioner ------------------------------------------

__global__ void extract_diag_inv_kernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    double* __restrict__ diag_inv,
    int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // Find diagonal block.
  int diag_k = -1;
  for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
    if (col_idx[k] == i) { diag_k = k; break; }
  }
  if (diag_k < 0) return;

  const double* blk = values + diag_k * 36;
  double L[36];
  for (int a = 0; a < 36; ++a) L[a] = blk[a];

  // In-place Cholesky.
  bool ok = true;
  for (int j = 0; j < 6 && ok; ++j) {
    double sum = 0.0;
    for (int p = 0; p < j; ++p) sum += L[j * 6 + p] * L[j * 6 + p];
    double diag = L[j * 6 + j] - sum;
    if (diag <= 1e-30) { ok = false; break; }
    L[j * 6 + j] = sqrt(diag);
    for (int row = j + 1; row < 6; ++row) {
      double s = 0.0;
      for (int p = 0; p < j; ++p) s += L[row * 6 + p] * L[j * 6 + p];
      L[row * 6 + j] = (L[row * 6 + j] - s) / L[j * 6 + j];
    }
  }

  double* out = diag_inv + i * 36;
  if (!ok) {
    // Fallback to identity.
    for (int a = 0; a < 36; ++a) out[a] = 0.0;
    for (int a = 0; a < 6; ++a) out[a * 6 + a] = 1.0;
    return;
  }

  // Invert: solve L L^T X = I column by column.
  for (int col = 0; col < 6; ++col) {
    double z[6] = {0, 0, 0, 0, 0, 0};
    z[col] = 1.0;

    for (int row = 0; row < 6; ++row) {
      double s = 0.0;
      for (int p = 0; p < row; ++p) s += L[row * 6 + p] * z[p];
      z[row] = (z[row] - s) / L[row * 6 + row];
    }
    for (int row = 5; row >= 0; --row) {
      double s = 0.0;
      for (int p = row + 1; p < 6; ++p) s += L[p * 6 + row] * z[p];
      z[row] = (z[row] - s) / L[row * 6 + row];
    }

    for (int row = 0; row < 6; ++row) out[row * 6 + col] = z[row];
  }
}

double* cuda_extract_diagonal_inv(const DeviceBSR& dbsr) {
  double* d_diag_inv;
  CUDA_CHECK(cudaMalloc(&d_diag_inv, dbsr.num_block_rows * 36 * sizeof(double)));

  int threads = 256;
  int blocks = (dbsr.num_block_rows + threads - 1) / threads;
  extract_diag_inv_kernel<<<blocks, threads>>>(
      dbsr.d_row_ptr, dbsr.d_col_idx, dbsr.d_values, d_diag_inv,
      dbsr.num_block_rows);
  CUDA_CHECK(cudaGetLastError());

  return d_diag_inv;
}

__global__ void precond_apply_kernel(const double* __restrict__ diag_inv,
                                      const double* __restrict__ r,
                                      double* __restrict__ z,
                                      int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const double* blk = diag_inv + i * 36;
  const double* ri = r + i * 6;
  double* zi = z + i * 6;

  for (int row = 0; row < 6; ++row) {
    double s = 0.0;
    for (int c = 0; c < 6; ++c) s += blk[row * 6 + c] * ri[c];
    zi[row] = s;
  }
}

void cuda_precond_apply(const double* d_diag_inv, const double* d_r,
                        double* d_z, int num_blocks) {
  int threads = 256;
  int blocks = (num_blocks + threads - 1) / threads;
  precond_apply_kernel<<<blocks, threads>>>(d_diag_inv, d_r, d_z, num_blocks);
  CUDA_CHECK(cudaGetLastError());
}

// ---- Fast dot product with pre-allocated buffer --------------------------

// Uses a persistent device scalar to avoid per-call malloc.
static double fast_dot(const double* d_a, const double* d_b, int dim,
                       double* d_scratch) {
  CUDA_CHECK(cudaMemsetAsync(d_scratch, 0, sizeof(double)));
  int threads = 256;
  int blocks = (dim + threads - 1) / threads;
  dot_kernel<<<blocks, threads>>>(
      d_a, d_b, d_scratch, dim);
  double result;
  CUDA_CHECK(cudaMemcpy(&result, d_scratch, sizeof(double),
                        cudaMemcpyDeviceToHost));
  return result;
}

// ---- Fused PCG update kernel ----------------------------------------------
// Combines: dx += alpha*p, r -= alpha*Ap, and computes r^T*r in one pass.
// Eliminates separate axpy + dot kernel launches.

__global__ void pcg_update_xr_kernel(
    double* __restrict__ dx, const double* __restrict__ p,
    double* __restrict__ r, const double* __restrict__ Ap,
    double alpha, double* __restrict__ r_dot_r, int n) {
  double local_rr = 0.0;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; i < n; i += stride) {
    dx[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
    local_rr += r[i] * r[i];
  }

  // Warp reduction.
  local_rr = warp_reduce_sum_impl(local_rr);
  __shared__ double warp_sums[8];
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  if (lane == 0) warp_sums[warp_id] = local_rr;
  __syncthreads();
  if (warp_id == 0) {
    int nw = blockDim.x / 32;
    local_rr = (lane < nw) ? warp_sums[lane] : 0.0;
    local_rr = warp_reduce_sum_impl(local_rr);
    if (lane == 0) atomicAdd(r_dot_r, local_rr);
  }
}

// Fused: z = M^{-1}*r, then compute r^T*z (for rz_new).
__global__ void pcg_precond_rz_kernel(
    const double* __restrict__ diag_inv,
    const double* __restrict__ r,
    double* __restrict__ z,
    double* __restrict__ rz_out,
    int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const double* blk = diag_inv + i * 36;
  const double* ri = r + i * 6;
  double* zi = z + i * 6;

  double local_rz = 0.0;
  for (int row = 0; row < 6; ++row) {
    double s = 0.0;
    for (int c = 0; c < 6; ++c) s += blk[row * 6 + c] * ri[c];
    zi[row] = s;
    local_rz += ri[row] * s;  // r_i * z_i
  }

  // Warp reduction for rz.
  local_rz = warp_reduce_sum_impl(local_rz);
  int lane = threadIdx.x % 32;
  if (lane == 0) atomicAdd(rz_out, local_rz);
}

// Fused: p = z + beta*p, and compute p^T*(H*p_old) is not needed if we
// use the standard CG recurrence. But we still need pAp = p^T * Ap.
// We keep dot_kernel for pAp since it's the only remaining sync point.

// ---- GPU PCG solver (fused) -----------------------------------------------

int cuda_pcg_solve(const DeviceBSR& dbsr, const double* d_rhs,
                   double* d_dx, int dim, int max_iter, double tol) {
  int N = dbsr.num_block_rows;

  // Pre-allocate all buffers.
  double* d_r;
  double* d_z;
  double* d_p;
  double* d_Ap;
  double* d_scratch;
  CUDA_CHECK(cudaMalloc(&d_r, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_z, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_p, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_Ap, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_scratch, 3 * sizeof(double))); // rr, rz, pAp

  cuda_fill(d_dx, 0.0, dim);

  double* d_diag_inv = cuda_extract_diagonal_inv(dbsr);

  cuda_copy(d_r, d_rhs, dim);
  cuda_precond_apply(d_diag_inv, d_r, d_z, N);
  cuda_copy(d_p, d_z, dim);

  double rz = fast_dot(d_r, d_z, dim, d_scratch);
  double rhs_norm2 = fast_dot(d_rhs, d_rhs, dim, d_scratch);
  double rhs_norm = sqrt(rhs_norm2);

  if (rhs_norm < 1e-30) {
    cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
    cudaFree(d_scratch); cudaFree(d_diag_inv);
    return 0;
  }

  int threads = 256;
  int blocks_vec = (dim + threads - 1) / threads;
  int blocks_node = (N + threads - 1) / threads;

  // Check convergence every check_interval iterations to reduce host syncs.
  const int check_interval = 10;

  int iters;
  for (iters = 0; iters < max_iter; ++iters) {
    // Ap = H * p
    cuda_bsr_spmv(dbsr, d_p, d_Ap);

    // pAp = p^T * Ap (need this on host for alpha)
    double pAp = fast_dot(d_p, d_Ap, dim, d_scratch);
    if (fabs(pAp) < 1e-30) break;
    double alpha = rz / pAp;

    // Fused: dx += alpha*p, r -= alpha*Ap, compute r^T*r
    CUDA_CHECK(cudaMemsetAsync(d_scratch, 0, sizeof(double)));
    pcg_update_xr_kernel<<<blocks_vec, threads>>>(
        d_dx, d_p, d_r, d_Ap, alpha, d_scratch, dim);
    CUDA_CHECK(cudaGetLastError());

    // Check convergence periodically
    if ((iters + 1) % check_interval == 0 || iters == max_iter - 1) {
      double r_norm2;
      CUDA_CHECK(cudaMemcpy(&r_norm2, d_scratch, sizeof(double),
                            cudaMemcpyDeviceToHost));
      if (sqrt(r_norm2) / rhs_norm < tol) { ++iters; break; }
    }

    // Fused: z = M^{-1}*r and compute rz_new = r^T*z
    CUDA_CHECK(cudaMemsetAsync(d_scratch + 1, 0, sizeof(double)));
    pcg_precond_rz_kernel<<<blocks_node, threads>>>(
        d_diag_inv, d_r, d_z, d_scratch + 1, N);
    CUDA_CHECK(cudaGetLastError());

    double rz_new;
    CUDA_CHECK(cudaMemcpy(&rz_new, d_scratch + 1, sizeof(double),
                          cudaMemcpyDeviceToHost));

    double beta = rz_new / rz;
    rz = rz_new;

    // p = z + beta * p
    cuda_scale(beta, d_p, dim);
    cuda_axpy(1.0, d_z, d_p, dim);
  }

  cudaFree(d_r); cudaFree(d_z); cudaFree(d_p); cudaFree(d_Ap);
  cudaFree(d_scratch); cudaFree(d_diag_inv);
  return iters;
}

// ---- Gauge fix on device --------------------------------------------------

__global__ void gauge_fix_kernel(double* __restrict__ bsr_values,
                                  double* __restrict__ gradient,
                                  const int* __restrict__ row_ptr,
                                  const int* __restrict__ col_idx,
                                  const int* __restrict__ block_map,
                                  int N) {
  int tid = threadIdx.x;

  // Zero gradient[0..5]
  if (tid < 6) gradient[tid] = 0.0;

  // Zero all blocks in row 0.
  if (tid == 0) {
    for (int k = row_ptr[0]; k < row_ptr[1]; ++k) {
      double* blk = bsr_values + k * 36;
      for (int a = 0; a < 36; ++a) blk[a] = 0.0;
      if (col_idx[k] == 0) {
        for (int a = 0; a < 6; ++a) blk[a * 6 + a] = 1.0;
      }
    }
    // Zero column-0 blocks in other rows.
    for (int i = 1; i < N; ++i) {
      int k = block_map[i * N + 0];
      if (k >= 0) {
        double* blk = bsr_values + k * 36;
        for (int a = 0; a < 36; ++a) blk[a] = 0.0;
      }
    }
  }
}

void cuda_gauge_fix(DeviceBSR& dbsr, double* d_gradient) {
  gauge_fix_kernel<<<1, 32>>>(dbsr.d_values, d_gradient,
                                dbsr.d_row_ptr, dbsr.d_col_idx,
                                dbsr.d_block_map, dbsr.num_block_rows);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---- Retraction kernel ----------------------------------------------------

__global__ void retract_kernel(double* __restrict__ poses,
                                const double* __restrict__ dx, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  DSE3 T_old;
  double* p = poses + i * 12;
  for (int j = 0; j < 9; ++j) T_old.R.R.m[j] = p[j];
  T_old.t[0] = p[9]; T_old.t[1] = p[10]; T_old.t[2] = p[11];

  DVec6 delta;
  for (int d = 0; d < 6; ++d) delta[d] = dx[6 * i + d];

  DSE3 T_new = dse3_compose(T_old, dse3_exp(delta));

  for (int j = 0; j < 9; ++j) p[j] = T_new.R.R.m[j];
  p[9] = T_new.t[0]; p[10] = T_new.t[1]; p[11] = T_new.t[2];
}

void cuda_retract(DevicePoseGraph& dpg, const double* d_dx) {
  int threads = 256;
  int blocks = (dpg.num_nodes + threads - 1) / threads;
  retract_kernel<<<blocks, threads>>>(dpg.d_poses, d_dx, dpg.num_nodes);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Full GPU Gauss-Newton solver
// ============================================================================

// Helper: read poses back from GPU to update the host graph.
static void read_back_poses(DevicePoseGraph& dpg, PoseGraph& graph) {
  int N = dpg.num_nodes;
  std::vector<double> h_poses(N * 12);
  CUDA_CHECK(cudaMemcpy(h_poses.data(), dpg.d_poses,
                        N * 12 * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    auto& n = graph.nodes[i];
    const double* p = h_poses.data() + i * 12;

    n.x = p[9]; n.y = p[10]; n.z = p[11];

    // Rotation matrix to quaternion.
    double trace = p[0] + p[4] + p[8];
    double qw, qx, qy, qz;
    if (trace > 0.0) {
      double s = 0.5 / sqrt(trace + 1.0);
      qw = 0.25 / s;
      qx = (p[7] - p[5]) * s;
      qy = (p[2] - p[6]) * s;
      qz = (p[3] - p[1]) * s;
    } else if (p[0] > p[4] && p[0] > p[8]) {
      double s = 2.0 * sqrt(1.0 + p[0] - p[4] - p[8]);
      qw = (p[7] - p[5]) / s;
      qx = 0.25 * s;
      qy = (p[1] + p[3]) / s;
      qz = (p[2] + p[6]) / s;
    } else if (p[4] > p[8]) {
      double s = 2.0 * sqrt(1.0 + p[4] - p[0] - p[8]);
      qw = (p[2] - p[6]) / s;
      qx = (p[1] + p[3]) / s;
      qy = 0.25 * s;
      qz = (p[5] + p[7]) / s;
    } else {
      double s = 2.0 * sqrt(1.0 + p[8] - p[0] - p[4]);
      qw = (p[3] - p[1]) / s;
      qx = (p[2] + p[6]) / s;
      qy = (p[5] + p[7]) / s;
      qz = 0.25 * s;
    }
    double norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    n.qx = qx / norm; n.qy = qy / norm; n.qz = qz / norm; n.qw = qw / norm;
  }
}

GNResult solve_gauss_newton_gpu(PoseGraph& graph, const GNConfig& config) {
  using Clock = std::chrono::high_resolution_clock;

  GNResult result{};
  int N = static_cast<int>(graph.nodes.size());
  int dim = 6 * N;

  // Build graph coloring on CPU (one-time cost).
  EdgeColoring coloring = color_edges(graph);

  // Build BSR sparsity pattern on CPU.
  BSRMatrix bsr_host = bsr_symbolic(graph);

  // Transfer to GPU.
  DevicePoseGraph dpg = create_device_pose_graph(graph);
  DeviceBSR dbsr = create_device_bsr(bsr_host);

  // Build flat color-edge list for GPU.
  std::vector<int> flat_color_edges;
  std::vector<int> color_offsets = {0};
  for (int c = 0; c < coloring.num_colors; ++c) {
    for (int e : coloring.color_edges[c]) {
      flat_color_edges.push_back(e);
    }
    color_offsets.push_back(static_cast<int>(flat_color_edges.size()));
  }

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

  // Gradient on device.
  double* d_gradient;
  CUDA_CHECK(cudaMalloc(&d_gradient, dim * sizeof(double)));

  // dx on device.
  double* d_dx;
  CUDA_CHECK(cudaMalloc(&d_dx, dim * sizeof(double)));

  // rhs on device.
  double* d_rhs;
  CUDA_CHECK(cudaMalloc(&d_rhs, dim * sizeof(double)));

  if (config.verbose) {
    std::printf("GPU: %d nodes, %d edges, %d BSR blocks, %d colors\n",
                N, dpg.num_edges, dbsr.nnz_blocks, coloring.num_colors);
  }

  cudaEvent_t ev_start, ev_lin, ev_asm, ev_solve, ev_retract;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_lin);
  cudaEventCreate(&ev_asm);
  cudaEventCreate(&ev_solve);
  cudaEventCreate(&ev_retract);

  for (int iter = 0; iter < config.max_iterations; ++iter) {
    GNIterationStats stats{};
    stats.iteration = iter;
    auto t_total_start = Clock::now();

    cudaEventRecord(ev_start);

    // --- Linearize (GPU) ---
    cuda_linearize_edges_analytical(dpg);
    cudaEventRecord(ev_lin);

    // --- Assemble (GPU, colored) ---
    cuda_assemble_colored(dpg, dbsr, d_gradient, coloring,
                          d_color_edges, d_color_offsets);
    cudaEventRecord(ev_asm);

    // Get error.
    double total_error = cuda_compute_error(dpg);

    if (iter == 0) result.initial_error = total_error;
    stats.error = total_error;

    // Check gradient on host (transfer gradient for norm check).
    std::vector<double> h_gradient(dim);
    CUDA_CHECK(cudaMemcpy(h_gradient.data(), d_gradient, dim * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double grad_max = 0.0;
    for (int i = 0; i < dim; ++i) grad_max = std::max(grad_max, std::abs(h_gradient[i]));

    cudaEventSynchronize(ev_asm);
    float lin_ms, asm_ms;
    cudaEventElapsedTime(&lin_ms, ev_start, ev_lin);
    cudaEventElapsedTime(&asm_ms, ev_lin, ev_asm);
    stats.linearize_ms = lin_ms + asm_ms;  // combine linearize + assemble

    if (config.verbose) {
      std::printf("iter %3d  error=%.6e  |grad|_inf=%.6e  lin=%.1fms asm=%.1fms",
                  iter, total_error, grad_max, lin_ms, asm_ms);
    }

    if (grad_max < config.gradient_tolerance) {
      if (config.verbose) std::printf("  -> converged (gradient)\n");
      result.final_error = total_error;
      result.iterations = iter;
      result.stats.push_back(stats);
      break;
    }

    // --- Solve (GPU PCG) ---
    cuda_gauge_fix(dbsr, d_gradient);

    // rhs = -gradient
    cuda_copy(d_rhs, d_gradient, dim);
    cuda_scale(-1.0, d_rhs, dim);

    cudaEventRecord(ev_solve);  // Start of solve timing is after gauge fix
    // We record solve start before PCG but that's approximate.

    int pcg_iters = cuda_pcg_solve(dbsr, d_rhs, d_dx, dim, 300, 1e-8);
    cudaEventRecord(ev_solve);
    cudaEventSynchronize(ev_solve);

    auto t_solve_end = Clock::now();
    stats.solve_ms = std::chrono::duration<double, std::milli>(
        t_solve_end - Clock::now() + std::chrono::milliseconds(0)).count();
    // Approximate solve time from wall clock since CUDA events around PCG are tricky
    {
      auto now = Clock::now();
      stats.solve_ms = asm_ms; // Will be updated below properly
    }

    // --- Retract (GPU) ---
    cuda_retract(dpg, d_dx);
    cudaEventRecord(ev_retract);
    cudaEventSynchronize(ev_retract);

    float solve_ms_approx;
    cudaEventElapsedTime(&solve_ms_approx, ev_asm, ev_retract);
    stats.solve_ms = solve_ms_approx;  // includes solve + retract

    auto t_total_end = Clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start).count();
    stats.retract_ms = 0.0;  // included in solve_ms for now

    if (config.verbose) {
      std::printf("  pcg=%d  solve+retract=%.1fms  total=%.1fms\n",
                  pcg_iters, solve_ms_approx, stats.total_ms);
    }

    result.stats.push_back(stats);
    result.final_error = total_error;
    result.iterations = iter + 1;

    // Check error convergence.
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

  // Read optimized poses back to host.
  read_back_poses(dpg, graph);

  // Cleanup.
  cudaFree(d_gradient);
  cudaFree(d_dx);
  cudaFree(d_rhs);
  cudaFree(d_color_edges);
  cudaFree(d_color_offsets);
  free_device_pose_graph(dpg);
  free_device_bsr(dbsr);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_lin);
  cudaEventDestroy(ev_asm);
  cudaEventDestroy(ev_solve);
  cudaEventDestroy(ev_retract);

  return result;
}

// ============================================================================
// Levenberg-Marquardt damping kernel
// ============================================================================

__global__ void add_damping_kernel(double* __restrict__ bsr_values,
                                    const int* __restrict__ row_ptr,
                                    const int* __restrict__ col_idx,
                                    double lambda, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  // Find diagonal block for row i.
  for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
    if (col_idx[k] == i) {
      double* blk = bsr_values + k * 36;
      for (int d = 0; d < 6; ++d) {
        blk[d * 6 + d] += lambda * blk[d * 6 + d];
      }
      return;
    }
  }
}

static void cuda_add_damping(DeviceBSR& dbsr, double lambda) {
  int threads = 256;
  int blocks = (dbsr.num_block_rows + threads - 1) / threads;
  add_damping_kernel<<<blocks, threads>>>(
      dbsr.d_values, dbsr.d_row_ptr, dbsr.d_col_idx, lambda,
      dbsr.num_block_rows);
  CUDA_CHECK(cudaGetLastError());
}

GNResult solve_lm_gpu(PoseGraph& graph, const GNConfig& config) {
  using Clock = std::chrono::high_resolution_clock;

  GNResult result{};
  int N = static_cast<int>(graph.nodes.size());
  int dim = 6 * N;

  EdgeColoring coloring = color_edges(graph);
  BSRMatrix bsr_host = bsr_symbolic(graph);
  DevicePoseGraph dpg = create_device_pose_graph(graph);
  DeviceBSR dbsr = create_device_bsr(bsr_host);

  // Build flat color-edge list.
  std::vector<int> flat_color_edges;
  for (int c = 0; c < coloring.num_colors; ++c) {
    for (int e : coloring.color_edges[c]) {
      flat_color_edges.push_back(e);
    }
  }
  int* d_color_edges;
  int* d_color_offsets;
  std::vector<int> color_offsets = {0};
  for (int c = 0; c < coloring.num_colors; ++c) {
    color_offsets.push_back(color_offsets.back() +
                            static_cast<int>(coloring.color_edges[c].size()));
  }
  CUDA_CHECK(cudaMalloc(&d_color_edges, flat_color_edges.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_color_offsets, color_offsets.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_color_edges, flat_color_edges.data(),
                        flat_color_edges.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_color_offsets, color_offsets.data(),
                        color_offsets.size() * sizeof(int),
                        cudaMemcpyHostToDevice));

  double* d_gradient;
  double* d_dx;
  double* d_rhs;
  CUDA_CHECK(cudaMalloc(&d_gradient, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dx, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_rhs, dim * sizeof(double)));

  // LM parameters.
  double lambda = 1e-4;
  double nu = 2.0;
  double prev_error = 1e30;

  if (config.verbose) {
    std::printf("LM-GPU: %d nodes, %d edges, %d BSR blocks, %d colors\n",
                N, dpg.num_edges, dbsr.nnz_blocks, coloring.num_colors);
  }

  // Save poses for potential rollback.
  std::vector<double> h_poses_backup(N * 12);

  for (int iter = 0; iter < config.max_iterations; ++iter) {
    GNIterationStats stats{};
    stats.iteration = iter;
    auto t_total_start = Clock::now();

    // --- Linearize + assemble ---
    cuda_linearize_edges_analytical(dpg);
    cuda_assemble_colored(dpg, dbsr, d_gradient, coloring,
                          d_color_edges, d_color_offsets);

    double total_error = cuda_compute_error(dpg);

    if (iter == 0) {
      result.initial_error = total_error;
      prev_error = total_error;
    }
    stats.error = total_error;

    // Check gradient.
    std::vector<double> h_gradient(dim);
    CUDA_CHECK(cudaMemcpy(h_gradient.data(), d_gradient, dim * sizeof(double),
                          cudaMemcpyDeviceToHost));
    double grad_max = 0.0;
    for (int i = 0; i < dim; ++i) grad_max = std::max(grad_max, std::abs(h_gradient[i]));

    if (config.verbose) {
      std::printf("iter %3d  error=%.6e  |grad|=%.2e  lambda=%.2e",
                  iter, total_error, grad_max, lambda);
    }

    if (grad_max < config.gradient_tolerance) {
      if (config.verbose) std::printf("  -> converged (gradient)\n");
      result.final_error = total_error;
      result.iterations = iter;
      result.stats.push_back(stats);
      break;
    }

    // Save poses for rollback.
    CUDA_CHECK(cudaMemcpy(h_poses_backup.data(), dpg.d_poses,
                          N * 12 * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Add LM damping ---
    cuda_add_damping(dbsr, lambda);

    // --- Gauge fix + solve ---
    cuda_gauge_fix(dbsr, d_gradient);
    cuda_copy(d_rhs, d_gradient, dim);
    cuda_scale(-1.0, d_rhs, dim);

    int pcg_iters = cuda_pcg_solve(dbsr, d_rhs, d_dx, dim, 300, 1e-8);

    // --- Retract (tentative) ---
    cuda_retract(dpg, d_dx);

    // --- Evaluate new error ---
    cuda_linearize_edges_analytical(dpg);
    double new_error = cuda_compute_error(dpg);

    auto t_total_end = Clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(
        t_total_end - t_total_start).count();

    if (new_error < total_error) {
      // Accept step, decrease lambda.
      lambda = std::max(lambda / 3.0, 1e-10);
      nu = 2.0;
      prev_error = new_error;

      if (config.verbose) {
        std::printf("  pcg=%d  new_err=%.4e  ACCEPT  total=%.1fms\n",
                    pcg_iters, new_error, stats.total_ms);
      }
    } else {
      // Reject step, increase lambda, restore poses.
      CUDA_CHECK(cudaMemcpy(dpg.d_poses, h_poses_backup.data(),
                            N * 12 * sizeof(double), cudaMemcpyHostToDevice));
      lambda *= nu;
      nu *= 2.0;

      if (config.verbose) {
        std::printf("  pcg=%d  new_err=%.4e  REJECT  total=%.1fms\n",
                    pcg_iters, new_error, stats.total_ms);
      }
    }

    result.stats.push_back(stats);
    result.final_error = std::min(total_error, new_error);
    result.iterations = iter + 1;

    // Error convergence.
    if (iter > 0) {
      double rel_change = std::abs(total_error - prev_error) /
                          (std::abs(prev_error) + 1e-30);
      if (rel_change < config.error_tolerance && new_error <= total_error) {
        if (config.verbose) {
          std::printf("  -> converged (error stagnation, rel=%.2e)\n", rel_change);
        }
        break;
      }
    }
    prev_error = total_error;
  }

  // Read back optimized poses.
  read_back_poses(dpg, graph);

  cudaFree(d_gradient);
  cudaFree(d_dx);
  cudaFree(d_rhs);
  cudaFree(d_color_edges);
  cudaFree(d_color_offsets);
  free_device_pose_graph(dpg);
  free_device_bsr(dbsr);

  return result;
}

}  // namespace cugraphopt
