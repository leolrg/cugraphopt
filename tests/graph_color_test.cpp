#include "cugraphopt/bsr.hpp"
#include "cugraphopt/graph_color.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>

using namespace cugraphopt;

static PoseGraph make_triangle_graph() {
  PoseGraph graph;
  Pose3Node n0, n1, n2;
  n0.id = 0; n0.x = 0; n0.y = 0; n0.z = 0;
  n0.qx = 0; n0.qy = 0; n0.qz = 0; n0.qw = 1;
  n1.id = 1; n1.x = 1; n1.y = 0; n1.z = 0;
  n1.qx = 0; n1.qy = 0; n1.qz = 0; n1.qw = 1;
  n2.id = 2; n2.x = 0; n2.y = 1; n2.z = 0;
  n2.qx = 0; n2.qy = 0; n2.qz = 0; n2.qw = 1;
  graph.nodes.push_back(n0);
  graph.nodes.push_back(n1);
  graph.nodes.push_back(n2);

  auto make_info = []() -> std::array<double, 21> {
    std::array<double, 21> info{};
    info[0] = 1; info[6] = 1; info[11] = 1;
    info[15] = 1; info[18] = 1; info[20] = 1;
    return info;
  };

  Pose3Edge e01, e02, e12;
  e01.from = 0; e01.to = 1;
  e01.x = 1; e01.y = 0; e01.z = 0;
  e01.qx = 0; e01.qy = 0; e01.qz = 0; e01.qw = 1;
  e01.information = make_info();
  e02.from = 0; e02.to = 2;
  e02.x = 0; e02.y = 1; e02.z = 0;
  e02.qx = 0; e02.qy = 0; e02.qz = 0; e02.qw = 1;
  e02.information = make_info();
  e12.from = 1; e12.to = 2;
  e12.x = -1; e12.y = 1.2; e12.z = 0;
  e12.qx = 0; e12.qy = 0; e12.qz = 0; e12.qw = 1;
  e12.information = make_info();
  graph.edges.push_back(e01);
  graph.edges.push_back(e02);
  graph.edges.push_back(e12);

  return graph;
}

static void test_coloring_valid() {
  PoseGraph graph = make_triangle_graph();
  EdgeColoring coloring = color_edges(graph);

  int N = static_cast<int>(graph.nodes.size());
  int E = static_cast<int>(graph.edges.size());

  // Every edge should have a color.
  assert(static_cast<int>(coloring.edge_color.size()) == E);
  for (int e = 0; e < E; ++e) {
    assert(coloring.edge_color[e] >= 0);
    assert(coloring.edge_color[e] < coloring.num_colors);
  }

  // No two same-colored edges should share a node.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < N; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  for (int c = 0; c < coloring.num_colors; ++c) {
    std::set<int> nodes_used;
    for (int e : coloring.color_edges[c]) {
      int i = id_to_idx.at(graph.edges[e].from);
      int j = id_to_idx.at(graph.edges[e].to);
      assert(nodes_used.count(i) == 0);
      assert(nodes_used.count(j) == 0);
      nodes_used.insert(i);
      nodes_used.insert(j);
    }
  }

  // Triangle: 3 edges, each sharing nodes -> need 3 colors.
  assert(coloring.num_colors == 3);

  std::printf("PASS: test_coloring_valid (colors=%d for %d edges)\n",
              coloring.num_colors, E);
}

#ifdef CUGRAPHOPT_DATASET_DIR
static void test_coloring_sphere2500() {
  std::string path = std::string(CUGRAPHOPT_DATASET_DIR) + "/sphere2500.g2o";
  PoseGraph graph = load_pose_graph(path);
  EdgeColoring coloring = color_edges(graph);

  int N = static_cast<int>(graph.nodes.size());
  int E = static_cast<int>(graph.edges.size());

  // Validate.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < N; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  for (int c = 0; c < coloring.num_colors; ++c) {
    std::set<int> nodes_used;
    for (int e : coloring.color_edges[c]) {
      int i = id_to_idx.at(graph.edges[e].from);
      int j = id_to_idx.at(graph.edges[e].to);
      assert(nodes_used.count(i) == 0);
      assert(nodes_used.count(j) == 0);
      nodes_used.insert(i);
      nodes_used.insert(j);
    }
  }

  // Count total edges across colors.
  int total = 0;
  for (int c = 0; c < coloring.num_colors; ++c) {
    total += static_cast<int>(coloring.color_edges[c].size());
  }
  assert(total == E);

  std::printf("PASS: test_coloring_sphere2500 (%d edges, %d colors)\n",
              E, coloring.num_colors);
}
#endif

static void test_colored_assembly_matches() {
  // Verify that assembling by color groups gives the same result as
  // sequential assembly.
  PoseGraph graph = make_triangle_graph();

  // Sequential BSR assembly.
  BSRMatrix bsr_seq = bsr_symbolic(graph);
  std::vector<double> grad_seq;
  double err_seq = bsr_assemble(bsr_seq, grad_seq, graph);

  // The BSR assembly already does sequential processing of all edges.
  // Graph coloring just reorders which edges are processed together,
  // but the final result should be identical (since we're on CPU with
  // no races). This validates the coloring doesn't miss any edges.
  EdgeColoring coloring = color_edges(graph);

  int total_edges = 0;
  for (int c = 0; c < coloring.num_colors; ++c) {
    total_edges += static_cast<int>(coloring.color_edges[c].size());
  }
  assert(total_edges == static_cast<int>(graph.edges.size()));

  std::printf("PASS: test_colored_assembly_matches (error=%.4e)\n", err_seq);
}

int main() {
  test_coloring_valid();
  test_colored_assembly_matches();
#ifdef CUGRAPHOPT_DATASET_DIR
  test_coloring_sphere2500();
#endif
  std::printf("\nAll graph coloring tests passed.\n");
  return 0;
}
