/// Benchmark our custom BSR SpMV vs cuSPARSE BSR SpMV.

#include "cugraphopt/bsr.hpp"
#include "cugraphopt/cuda_solver.cuh"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"

#include <cstdio>
#include <cstdlib>
#include <cusparse.h>
#include <string>
#include <vector>

using namespace cugraphopt;

#define CUDA_CHECK(call) \
  do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); std::exit(1); \
  } } while(0)

#define CUSPARSE_CHECK(call) \
  do { cusparseStatus_t s = (call); if (s != CUSPARSE_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuSPARSE error: %d\n", (int)s); std::exit(1); \
  } } while(0)

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: spmv_bench <pose_graph.g2o>\n");
    return 1;
  }

  PoseGraph graph = load_pose_graph(argv[1]);
  int N = static_cast<int>(graph.nodes.size());
  int dim = 6 * N;

  std::printf("Loaded: %d nodes, %zu edges, dim=%d\n",
              N, graph.edges.size(), dim);

  // Build BSR on CPU.
  BSRMatrix bsr = bsr_symbolic(graph);
  std::vector<double> gradient;
  bsr_assemble(bsr, gradient, graph);

  std::printf("BSR: %d block-rows, %d nnz blocks\n",
              bsr.num_block_rows, bsr.nnz_blocks);

  // Transfer to GPU.
  DeviceBSR dbsr = create_device_bsr(bsr);
  CUDA_CHECK(cudaMemcpy(dbsr.d_values, bsr.values.data(),
                        bsr.nnz_blocks * 36 * sizeof(double),
                        cudaMemcpyHostToDevice));

  // Create test vector x.
  std::vector<double> h_x(dim);
  for (int i = 0; i < dim; ++i) h_x[i] = 1.0 / (i + 1);

  double* d_x;
  double* d_y_custom;
  double* d_y_cusparse;
  CUDA_CHECK(cudaMalloc(&d_x, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_y_custom, dim * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_y_cusparse, dim * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), dim * sizeof(double),
                        cudaMemcpyHostToDevice));

  // ---- Benchmark custom BSR SpMV ----
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup.
  for (int i = 0; i < 10; ++i) {
    cuda_bsr_spmv(dbsr, d_x, d_y_custom);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  int n_iters = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < n_iters; ++i) {
    cuda_bsr_spmv(dbsr, d_x, d_y_custom);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float custom_ms;
  cudaEventElapsedTime(&custom_ms, start, stop);
  std::printf("Custom BSR SpMV: %.3f ms / call (avg over %d calls)\n",
              custom_ms / n_iters, n_iters);

  // ---- Benchmark cuSPARSE BSR SpMV ----
  cusparseHandle_t handle;
  CUSPARSE_CHECK(cusparseCreate(&handle));

  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;

  CUSPARSE_CHECK(cusparseCreateBsr(
      &matA,
      bsr.num_block_rows, bsr.num_block_rows, bsr.nnz_blocks,
      6, 6,
      dbsr.d_row_ptr, dbsr.d_col_idx, dbsr.d_values,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      CUDA_R_64F, CUSPARSE_ORDER_ROW));

  CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, dim, d_x, CUDA_R_64F));
  CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, dim, d_y_cusparse, CUDA_R_64F));

  double alpha = 1.0, beta = 0.0;
  size_t bufferSize;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

  void* dBuffer;
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

  // Warmup.
  for (int i = 0; i < 10; ++i) {
    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEventRecord(start);
  for (int i = 0; i < n_iters; ++i) {
    CUSPARSE_CHECK(cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float cusparse_ms;
  cudaEventElapsedTime(&cusparse_ms, start, stop);
  std::printf("cuSPARSE BSR SpMV: %.3f ms / call (avg over %d calls)\n",
              cusparse_ms / n_iters, n_iters);

  // ---- Compare results ----
  std::vector<double> h_y_custom(dim), h_y_cusparse(dim);
  CUDA_CHECK(cudaMemcpy(h_y_custom.data(), d_y_custom, dim * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_y_cusparse.data(), d_y_cusparse, dim * sizeof(double),
                        cudaMemcpyDeviceToHost));

  double max_diff = 0.0;
  for (int i = 0; i < dim; ++i) {
    max_diff = std::max(max_diff, std::abs(h_y_custom[i] - h_y_cusparse[i]));
  }

  std::printf("\nResults comparison: max |custom - cuSPARSE| = %.2e\n", max_diff);
  std::printf("Speedup (custom/cuSPARSE): %.2fx\n",
              cusparse_ms / custom_ms);

  // Cleanup.
  cusparseDestroySpMat(matA);
  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroy(handle);
  cudaFree(dBuffer);
  cudaFree(d_x);
  cudaFree(d_y_custom);
  cudaFree(d_y_cusparse);
  free_device_bsr(dbsr);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
