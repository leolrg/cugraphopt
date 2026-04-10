#pragma once

#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/se3.hpp"

#include <vector>

namespace cugraphopt {

struct LinearizationResult {
  std::vector<double> H;
  std::vector<double> b;
  double total_error = 0.0;
  int dim = 0;
};

SE3 to_SE3(const Pose3Node& node);
SE3 to_SE3(const Pose3Edge& edge);

void expand_information(const std::array<double, 21>& upper, Mat6& omega);

se3 compute_residual(const SE3& T_i, const SE3& T_j, const SE3& Z_ij);

void compute_jacobians(const SE3& T_i, const SE3& T_j, const SE3& Z_ij,
                       Mat6& J_i, Mat6& J_j);

LinearizationResult linearize(const PoseGraph& graph);

}  // namespace cugraphopt
