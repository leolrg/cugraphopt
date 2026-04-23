#include "cugraphopt/core.hpp"
#include "cugraphopt/cuda_solver.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/solver.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << cugraphopt::build_banner() << '\n';
    return 0;
  }

  if (argc == 2) {
    const cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[1]);
    std::cout << "nodes=" << graph.nodes.size()
              << " edges=" << graph.edges.size() << '\n';
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--linearize") {
    const cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[2]);
    const cugraphopt::LinearizationResult result =
        cugraphopt::linearize(graph);
    std::cout << "nodes=" << graph.nodes.size()
              << " edges=" << graph.edges.size()
              << " dim=" << result.dim
              << " error=" << result.total_error << '\n';
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--solve") {
    cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[2]);
    std::printf("Loaded: nodes=%zu edges=%zu\n", graph.nodes.size(),
                graph.edges.size());

    cugraphopt::GNConfig cfg;
    cfg.max_iterations = 30;
    cfg.verbose = true;

    cugraphopt::GNResult res =
        cugraphopt::solve_gauss_newton_sparse(graph, cfg);
    std::printf("converged: %d iterations, error %.6e -> %.6e\n",
                res.iterations, res.initial_error, res.final_error);
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--solve-dense") {
    cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[2]);
    std::printf("Loaded: nodes=%zu edges=%zu\n", graph.nodes.size(),
                graph.edges.size());

    cugraphopt::GNConfig cfg;
    cfg.max_iterations = 30;
    cfg.verbose = true;

    cugraphopt::GNResult res = cugraphopt::solve_gauss_newton(graph, cfg);
    std::printf("converged: %d iterations, error %.6e -> %.6e\n",
                res.iterations, res.initial_error, res.final_error);
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--solve-lm") {
    cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[2]);
    std::printf("Loaded: nodes=%zu edges=%zu\n", graph.nodes.size(),
                graph.edges.size());

    cugraphopt::GNConfig cfg;
    cfg.max_iterations = 50;
    cfg.verbose = true;

    cugraphopt::GNResult res = cugraphopt::solve_lm_gpu(graph, cfg);
    std::printf("converged: %d iterations, error %.6e -> %.6e\n",
                res.iterations, res.initial_error, res.final_error);
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--solve-gpu") {
    cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[2]);
    std::printf("Loaded: nodes=%zu edges=%zu\n", graph.nodes.size(),
                graph.edges.size());

    cugraphopt::GNConfig cfg;
    cfg.max_iterations = 30;
    cfg.verbose = true;

    cugraphopt::GNResult res =
        cugraphopt::solve_gauss_newton_gpu(graph, cfg);
    std::printf("converged: %d iterations, error %.6e -> %.6e\n",
                res.iterations, res.initial_error, res.final_error);
    return 0;
  }

  if (argc == 3 && std::string(argv[1]) == "--benchmark") {
    cugraphopt::PoseGraph graph_cpu = cugraphopt::load_pose_graph(argv[2]);
    cugraphopt::PoseGraph graph_gpu = graph_cpu;  // copy for GPU run

    std::printf("=== BENCHMARK: %zu nodes, %zu edges ===\n\n",
                graph_cpu.nodes.size(), graph_cpu.edges.size());

    cugraphopt::GNConfig cfg;
    cfg.max_iterations = 30;
    cfg.verbose = true;

    // CPU sparse
    std::printf("--- CPU Sparse (BSR + PCG) ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    cugraphopt::GNResult res_cpu =
        cugraphopt::solve_gauss_newton_sparse(graph_cpu, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("CPU: %d iters, error %.6e -> %.6e, total %.1f ms\n\n",
                res_cpu.iterations, res_cpu.initial_error,
                res_cpu.final_error, cpu_ms);

    // GPU
    std::printf("--- GPU (CUDA Kernels + PCG) ---\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    cugraphopt::GNResult res_gpu =
        cugraphopt::solve_gauss_newton_gpu(graph_gpu, cfg);
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("GPU: %d iters, error %.6e -> %.6e, total %.1f ms\n\n",
                res_gpu.iterations, res_gpu.initial_error,
                res_gpu.final_error, gpu_ms);

    // Summary
    std::printf("=== SUMMARY ===\n");
    std::printf("CPU total: %.1f ms\n", cpu_ms);
    std::printf("GPU total: %.1f ms\n", gpu_ms);
    std::printf("Speedup:   %.1fx\n", cpu_ms / gpu_ms);
    std::printf("CPU final error: %.6e\n", res_cpu.final_error);
    std::printf("GPU final error: %.6e\n", res_gpu.final_error);

    // Per-iteration breakdown
    if (!res_cpu.stats.empty() && !res_gpu.stats.empty()) {
      std::printf("\n=== PER-ITERATION COMPARISON (last iteration) ===\n");
      const auto& cs = res_cpu.stats.back();
      const auto& gs = res_gpu.stats.back();
      std::printf("CPU: linearize=%.1fms solve=%.1fms retract=%.1fms total=%.1fms\n",
                  cs.linearize_ms, cs.solve_ms, cs.retract_ms, cs.total_ms);
      std::printf("GPU: total=%.1fms\n", gs.total_ms);
      std::printf("Per-iter speedup: %.1fx\n", cs.total_ms / gs.total_ms);
    }

    return 0;
  }

  std::cerr << "Usage: cugraphopt [--linearize|--solve|--solve-dense|"
               "--solve-gpu|--benchmark] [pose_graph.g2o]\n";
  return 1;
}
