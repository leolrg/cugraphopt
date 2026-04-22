#include "cugraphopt/core.hpp"
#include "cugraphopt/cuda_solver.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/solver.hpp"

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

  std::cerr << "Usage: cugraphopt [--linearize|--solve|--solve-dense|--solve-gpu] "
               "[pose_graph.g2o]\n";
  return 1;
}
