#include "cugraphopt/linearization.hpp"

#include <cmath>
#include <unordered_map>

namespace cugraphopt {

SE3 to_SE3(const Pose3Node& node) {
  return {quat_to_SO3(node.qx, node.qy, node.qz, node.qw),
          {node.x, node.y, node.z}};
}

SE3 to_SE3(const Pose3Edge& edge) {
  return {quat_to_SO3(edge.qx, edge.qy, edge.qz, edge.qw),
          {edge.x, edge.y, edge.z}};
}

void expand_information(const std::array<double, 21>& upper, Mat6& omega) {
  omega = {};
  int k = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      omega[i * 6 + j] = upper[k];
      omega[j * 6 + i] = upper[k];
      ++k;
    }
  }
}

se3 compute_residual(const SE3& T_i, const SE3& T_j, const SE3& Z_ij) {
  const SE3 T_ij_pred = compose(inverse(T_i), T_j);
  const SE3 E = compose(inverse(Z_ij), T_ij_pred);
  return log(E);
}

void compute_jacobians(const SE3& /*T_i*/, const SE3& /*T_j*/,
                       const SE3& /*Z_ij*/, Mat6& J_i, Mat6& J_j) {
  // Stub — filled in Task 3.
  J_i = {};
  J_j = {};
}

LinearizationResult linearize(const PoseGraph& /*graph*/) {
  // Stub — filled in Task 4.
  return {};
}

}  // namespace cugraphopt
