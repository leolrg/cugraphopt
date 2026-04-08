#include "cugraphopt/core.hpp"
#include "cugraphopt/pose_graph.hpp"

#include <iostream>

int main(int argc, char** argv) {
  if (argc == 1) {
    std::cout << cugraphopt::build_banner() << '\n';
    return 0;
  }

  if (argc == 2) {
    const cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(argv[1]);
    std::cout << "nodes=" << graph.nodes.size() << " edges=" << graph.edges.size()
              << '\n';
    return 0;
  }

  std::cerr << "Usage: cugraphopt [pose_graph.g2o]\n";
  return 1;
}
