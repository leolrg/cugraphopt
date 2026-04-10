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

namespace {
constexpr double kFiniteDiffEps = 1e-8;
}  // namespace

void compute_jacobians(const SE3& T_i, const SE3& T_j, const SE3& Z_ij,
                       Mat6& J_i, Mat6& J_j) {
  J_i = {};
  J_j = {};

  for (int k = 0; k < 6; ++k) {
    se3 delta{};

    // J_i column k: right-perturb T_i
    delta[k] = kFiniteDiffEps;
    const SE3 T_i_plus = compose(T_i, exp(delta));
    delta[k] = -kFiniteDiffEps;
    const SE3 T_i_minus = compose(T_i, exp(delta));

    const se3 ri_plus = compute_residual(T_i_plus, T_j, Z_ij);
    const se3 ri_minus = compute_residual(T_i_minus, T_j, Z_ij);

    for (int row = 0; row < 6; ++row) {
      J_i[row * 6 + k] =
          (ri_plus[row] - ri_minus[row]) / (2.0 * kFiniteDiffEps);
    }

    // J_j column k: right-perturb T_j
    delta[k] = kFiniteDiffEps;
    const SE3 T_j_plus = compose(T_j, exp(delta));
    delta[k] = -kFiniteDiffEps;
    const SE3 T_j_minus = compose(T_j, exp(delta));

    const se3 rj_plus = compute_residual(T_i, T_j_plus, Z_ij);
    const se3 rj_minus = compute_residual(T_i, T_j_minus, Z_ij);

    for (int row = 0; row < 6; ++row) {
      J_j[row * 6 + k] =
          (rj_plus[row] - rj_minus[row]) / (2.0 * kFiniteDiffEps);
    }
  }
}

LinearizationResult linearize(const PoseGraph& /*graph*/) {
  // Stub — filled in Task 4.
  return {};
}

}  // namespace cugraphopt
