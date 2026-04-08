#include "cugraphopt/pose_graph.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

void test_loads_sample_graph() {
  const auto fixture =
      std::filesystem::path(CUGRAPHOPT_TEST_FIXTURE_DIR) /
      "sample_pose_graph.g2o";

  const cugraphopt::PoseGraph graph = cugraphopt::load_pose_graph(fixture);
  assert(graph.nodes.size() == 2);
  assert(graph.edges.size() == 1);

  const cugraphopt::Pose3Node& node = graph.nodes.at(1);
  assert(node.id == 1);
  assert(node.x == 1.0);
  assert(node.y == 2.0);
  assert(node.z == 3.0);
  assert(node.qw == 1.0);

  const cugraphopt::Pose3Edge& edge = graph.edges.at(0);
  assert(edge.from == 0);
  assert(edge.to == 1);
  assert(edge.x == 1.0);
  assert(edge.y == 2.0);
  assert(edge.z == 3.0);
  assert(edge.information.at(0) == 1.0);
  assert(edge.information.at(20) == 21.0);
}

void test_rejects_unsupported_records() {
  const auto path = std::filesystem::temp_directory_path() /
                    "cugraphopt_unsupported_record.g2o";

  {
    std::ofstream output(path);
    output << "VERTEX_XY 0 1 2\n";
  }

  bool threw = false;
  try {
    (void)cugraphopt::load_pose_graph(path);
  } catch (const std::runtime_error&) {
    threw = true;
  }

  std::filesystem::remove(path);
  assert(threw);
}

}  // namespace

int main() {
  test_loads_sample_graph();
  test_rejects_unsupported_records();
  return 0;
}
