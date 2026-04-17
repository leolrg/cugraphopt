#pragma once

#include "cugraphopt/pose_graph.hpp"

#include <vector>

namespace cugraphopt {

/// Result of edge coloring: maps each edge to a color, and provides the list
/// of edges per color for batch processing.
struct EdgeColoring {
  int num_colors;
  std::vector<int> edge_color;               // color of each edge
  std::vector<std::vector<int>> color_edges;  // edges grouped by color
};

/// Greedy edge coloring: assign colors so no two edges sharing a node have the
/// same color. Uses greedy first-available strategy.
/// Returns the coloring with num_colors <= max_degree + 1 (Vizing's bound).
EdgeColoring color_edges(const PoseGraph& graph);

}  // namespace cugraphopt
