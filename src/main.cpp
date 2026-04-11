#include "cugraphopt/core.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"

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

  std::cerr << "Usage: cugraphopt [--linearize] [pose_graph.g2o]\n";
  return 1;
}
