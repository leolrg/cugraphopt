#include "cugraphopt/graph_color.hpp"

#include <algorithm>
#include <set>
#include <unordered_map>
#include <vector>

namespace cugraphopt {

EdgeColoring color_edges(const PoseGraph& graph) {
  const int E = static_cast<int>(graph.edges.size());
  const int N = static_cast<int>(graph.nodes.size());

  // Map node IDs to sequential indices.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < N; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  // Build adjacency: for each node, list of edge indices that touch it.
  std::vector<std::vector<int>> node_edges(N);
  for (int e = 0; e < E; ++e) {
    int i = id_to_idx.at(graph.edges[e].from);
    int j = id_to_idx.at(graph.edges[e].to);
    node_edges[i].push_back(e);
    node_edges[j].push_back(e);
  }

  EdgeColoring result;
  result.edge_color.assign(E, -1);

  int max_color = -1;

  for (int e = 0; e < E; ++e) {
    int i = id_to_idx.at(graph.edges[e].from);
    int j = id_to_idx.at(graph.edges[e].to);

    // Collect colors used by adjacent edges (edges sharing node i or j).
    std::set<int> used;
    for (int adj : node_edges[i]) {
      if (result.edge_color[adj] >= 0) {
        used.insert(result.edge_color[adj]);
      }
    }
    for (int adj : node_edges[j]) {
      if (result.edge_color[adj] >= 0) {
        used.insert(result.edge_color[adj]);
      }
    }

    // Find smallest available color.
    int color = 0;
    while (used.count(color)) {
      ++color;
    }

    result.edge_color[e] = color;
    if (color > max_color) max_color = color;
  }

  result.num_colors = max_color + 1;

  // Group edges by color.
  result.color_edges.resize(result.num_colors);
  for (int e = 0; e < E; ++e) {
    result.color_edges[result.edge_color[e]].push_back(e);
  }

  return result;
}

}  // namespace cugraphopt
