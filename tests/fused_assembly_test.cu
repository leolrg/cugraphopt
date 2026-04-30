/// Verify fused analytical linearization + atomic assembly matches the
/// existing two-kernel analytical linearization followed by atomic assembly.

#include "cugraphopt/bsr.hpp"
#include "cugraphopt/cuda_solver.cuh"
#include "cugraphopt/pose_graph.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t e = (call);                                                \
    if (e != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));    \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

using namespace cugraphopt;

static double max_abs_diff(const std::vector<double>& a,
                           const std::vector<double>& b) {
  assert(a.size() == b.size());
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

int main() {
  PoseGraph graph = load_pose_graph(
      std::string(CUGRAPHOPT_TEST_FIXTURE_DIR) + "/sample_pose_graph.g2o");
  BSRMatrix bsr_host = bsr_symbolic(graph);
  DevicePoseGraph dpg = create_device_pose_graph(graph);
  DeviceBSR dbsr = create_device_bsr(bsr_host);

  const int dim = static_cast<int>(graph.nodes.size()) * 6;
  double* d_gradient = nullptr;
  CUDA_CHECK(cudaMalloc(&d_gradient, dim * sizeof(double)));

  cuda_linearize_edges_analytical(dpg);
  cuda_assemble_atomic(dpg, dbsr, d_gradient);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> ref_bsr(dbsr.nnz_blocks * 36);
  std::vector<double> ref_grad(dim);
  std::vector<double> ref_errors(dpg.num_edges);
  CUDA_CHECK(cudaMemcpy(ref_bsr.data(), dbsr.d_values,
                        ref_bsr.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_grad.data(), d_gradient,
                        ref_grad.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref_errors.data(), dpg.d_errors,
                        ref_errors.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  cuda_linearize_assemble_atomic(dpg, dbsr, d_gradient);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> fused_bsr(dbsr.nnz_blocks * 36);
  std::vector<double> fused_grad(dim);
  std::vector<double> fused_errors(dpg.num_edges);
  CUDA_CHECK(cudaMemcpy(fused_bsr.data(), dbsr.d_values,
                        fused_bsr.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fused_grad.data(), d_gradient,
                        fused_grad.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fused_errors.data(), dpg.d_errors,
                        fused_errors.size() * sizeof(double),
                        cudaMemcpyDeviceToHost));

  const double h_diff = max_abs_diff(ref_bsr, fused_bsr);
  const double b_diff = max_abs_diff(ref_grad, fused_grad);
  const double e_diff = max_abs_diff(ref_errors, fused_errors);

  std::printf("fused assembly diffs: H=%.3e b=%.3e err=%.3e\n",
              h_diff, b_diff, e_diff);

  assert(h_diff < 1e-9);
  assert(b_diff < 1e-9);
  assert(e_diff < 1e-9);

  cudaFree(d_gradient);
  free_device_pose_graph(dpg);
  free_device_bsr(dbsr);

  std::printf("PASS: fused linearize+assembly matches two-kernel path\n");
  return 0;
}
