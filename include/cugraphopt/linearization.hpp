#pragma once

#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/se3.hpp"

#include <vector>

namespace cugraphopt {

/// Output of one Gauss-Newton linearization pass.
///
/// For a graph with N nodes:
///   - H:  dense 6N x 6N approximate Hessian, row-major
///   - b:  dense 6N gradient vector
///   - total_error:  sum of weighted squared residuals (r^T Omega r)
///   - dim:  6N (the dimension of H and b)
struct LinearizationResult {
  std::vector<double> H;
  std::vector<double> b;
  double total_error = 0.0;
  int dim = 0;
};

/// Convert a parsed node/edge to an SE3 group element.
SE3 to_SE3(const Pose3Node& node);
SE3 to_SE3(const Pose3Edge& edge);

/// Expand 21 upper-triangular values (g2o format) into a full 6x6 symmetric
/// information matrix Omega.
void expand_information(const std::array<double, 21>& upper, Mat6& omega);

/// Compute the SE(3) residual for one edge: r = Log(Z_ij^{-1} * T_i^{-1} * T_j).
/// Returns a 6-vector in se(3) using [rho; phi] convention (translation first).
/// Zero when the poses are consistent with the measurement.
se3 compute_residual(const SE3& T_i, const SE3& T_j, const SE3& Z_ij);

/// Compute 6x6 Jacobians dr/d(xi_i) and dr/d(xi_j) via central finite
/// differences with right-perturbation: T * Exp(epsilon * e_k).
void compute_jacobians(const SE3& T_i, const SE3& T_j, const SE3& Z_ij,
                       Mat6& J_i, Mat6& J_j);

/// Run one Gauss-Newton linearization pass over the full pose graph.
/// Assembles dense H, b, and total squared error from all edges.
LinearizationResult linearize(const PoseGraph& graph);

}  // namespace cugraphopt
