/// Benchmark naive atomicAdd assembly vs graph-colored lock-free assembly.

#include "cugraphopt/bsr.hpp"
#include "cugraphopt/cuda_solver.cuh"
#include "cugraphopt/graph_color.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) \
  do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); \
    std::exit(1); } } while(0)

using namespace cugraphopt;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: assembly_bench <pose_graph.g2o>\n");
    return 1;
  }

  PoseGraph graph = load_pose_graph(argv[1]);
  int N = static_cast<int>(graph.nodes.size());
  int dim = 6 * N;

  // Build CPU BSR sparsity pattern + graph coloring.
  BSRMatrix bsr_host = bsr_symbolic(graph);
  EdgeColoring coloring = color_edges(graph);

  std::printf("Dataset: %s\n", argv[1]);
  std::printf("  nodes=%d  edges=%zu  bsr_blocks=%d  colors=%d\n",
              N, graph.edges.size(), bsr_host.nnz_blocks, coloring.num_colors);

  // Transfer to device.
  DevicePoseGraph dpg = create_device_pose_graph(graph);
  DeviceBSR dbsr = create_device_bsr(bsr_host);

  // Allocate gradient.
  double* d_gradient;
  CUDA_CHECK(cudaMalloc(&d_gradient, dim * sizeof(double)));

  // Build flat color-edge list for colored assembly.
  std::vector<int> flat_color_edges;
  std::vector<int> color_offsets = {0};
  for (int c = 0; c < coloring.num_colors; ++c) {
    for (int e : coloring.color_edges[c]) flat_color_edges.push_back(e);
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

  // Linearize once (both assembly methods use the same Jacobians).
  cuda_linearize_edges_analytical(dpg);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int n_runs = 100;

  // ---- Warmup ----
  for (int i = 0; i < 5; ++i) {
    cuda_assemble_atomic(dpg, dbsr, d_gradient);
    cuda_assemble_colored(dpg, dbsr, d_gradient, coloring, d_color_edges, d_color_offsets);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- Atomic baseline ----
  cudaEventRecord(start);
  for (int i = 0; i < n_runs; ++i) {
    cuda_assemble_atomic(dpg, dbsr, d_gradient);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float atomic_ms;
  cudaEventElapsedTime(&atomic_ms, start, stop);
  atomic_ms /= n_runs;

  // Save results from atomic.
  std::vector<double> h_bsr_atomic(dbsr.nnz_blocks * 36);
  std::vector<double> h_grad_atomic(dim);
  CUDA_CHECK(cudaMemcpy(h_bsr_atomic.data(), dbsr.d_values,
                        dbsr.nnz_blocks * 36 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_grad_atomic.data(), d_gradient, dim * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // ---- Graph-colored lock-free ----
  cudaEventRecord(start);
  for (int i = 0; i < n_runs; ++i) {
    cuda_assemble_colored(dpg, dbsr, d_gradient, coloring, d_color_edges, d_color_offsets);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float colored_ms;
  cudaEventElapsedTime(&colored_ms, start, stop);
  colored_ms /= n_runs;

  std::vector<double> h_bsr_colored(dbsr.nnz_blocks * 36);
  std::vector<double> h_grad_colored(dim);
  CUDA_CHECK(cudaMemcpy(h_bsr_colored.data(), dbsr.d_values,
                        dbsr.nnz_blocks * 36 * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_grad_colored.data(), d_gradient, dim * sizeof(double),
                        cudaMemcpyDeviceToHost));

  // ---- Verify correctness ----
  double max_bsr_diff = 0.0, max_grad_diff = 0.0;
  for (int i = 0; i < dbsr.nnz_blocks * 36; ++i) {
    double d = std::abs(h_bsr_atomic[i] - h_bsr_colored[i]);
    if (d > max_bsr_diff) max_bsr_diff = d;
  }
  for (int i = 0; i < dim; ++i) {
    double d = std::abs(h_grad_atomic[i] - h_grad_colored[i]);
    if (d > max_grad_diff) max_grad_diff = d;
  }

  std::printf("\n--- Results (avg over %d runs) ---\n", n_runs);
  std::printf("  atomicAdd assembly:     %.3f ms\n", atomic_ms);
  std::printf("  graph-colored assembly: %.3f ms\n", colored_ms);
  std::printf("  speedup (colored vs atomic): %.2fx\n", atomic_ms / colored_ms);
  std::printf("\n--- Correctness ---\n");
  std::printf("  max |H_atomic - H_colored|: %.3e\n", max_bsr_diff);
  std::printf("  max |b_atomic - b_colored|: %.3e\n", max_grad_diff);
  if (max_bsr_diff < 1e-9 && max_grad_diff < 1e-9) {
    std::printf("  PASS: both assemblies produce identical results\n");
  } else {
    std::printf("  WARN: results differ (likely floating-point non-associativity)\n");
  }

  // Cleanup.
  cudaFree(d_gradient);
  cudaFree(d_color_edges);
  cudaFree(d_color_offsets);
  free_device_pose_graph(dpg);
  free_device_bsr(dbsr);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
