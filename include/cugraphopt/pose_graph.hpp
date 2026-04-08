#pragma once

#include <array>
#include <filesystem>
#include <vector>

namespace cugraphopt {

struct Pose3Node {
  int id = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
};

struct Pose3Edge {
  int from = 0;
  int to = 0;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
  double qw = 1.0;
  std::array<double, 21> information{};
};

struct PoseGraph {
  std::vector<Pose3Node> nodes;
  std::vector<Pose3Edge> edges;
};

PoseGraph load_pose_graph(const std::filesystem::path& path);

}  // namespace cugraphopt
