#include "cugraphopt/linearization.hpp"

#include <cmath>
#include <unordered_map>

namespace cugraphopt {

/// Convert a parsed Pose3Node (quaternion + translation) to an SE3 group element.
SE3 to_SE3(const Pose3Node& node) {
  return {quat_to_SO3(node.qx, node.qy, node.qz, node.qw),
          {node.x, node.y, node.z}};
}

/// Convert a parsed Pose3Edge measurement (quaternion + translation) to an SE3
/// group element representing the measured relative transform Z_ij.
SE3 to_SE3(const Pose3Edge& edge) {
  return {quat_to_SO3(edge.qx, edge.qy, edge.qz, edge.qw),
          {edge.x, edge.y, edge.z}};
}

/// Expand 21 upper-triangular values (g2o format) into a full 6x6 symmetric
/// information matrix.  The g2o storage order is row-major upper triangle:
///   (0,0), (0,1), ..., (0,5), (1,1), (1,2), ..., (1,5), (2,2), ..., (5,5)
/// totaling 6+5+4+3+2+1 = 21 values.
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

/// Compute the SE(3) residual for one edge using the right-perturbation
/// convention.
///
/// Given current pose estimates T_i, T_j and measured relative transform Z_ij:
///   1. Predicted relative transform:  T_ij_pred = T_i^{-1} * T_j
///   2. Error transform:               E = Z_ij^{-1} * T_ij_pred
///   3. Residual (tangent space):       r = Log(E)  ∈ se(3)
///
/// The residual is a 6-vector in [rho; phi] convention (translation first,
/// rotation second).  When the poses are consistent with the measurement,
/// E = I and r = 0.
se3 compute_residual(const SE3& T_i, const SE3& T_j, const SE3& Z_ij) {
  const SE3 T_ij_pred = compose(inverse(T_i), T_j);
  const SE3 E = compose(inverse(Z_ij), T_ij_pred);
  return log(E);
}

namespace {

/// Step size for central finite-difference Jacobian computation.
constexpr double kFiniteDiffEps = 1e-8;

/// Accumulate a 6x6 block into the global dense Hessian H at the given
/// row/column offset.  H is stored row-major with stride `dim`.
void add_block(std::vector<double>& H, int dim, int row_offset, int col_offset,
               const Mat6& block) {
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      H[(row_offset + i) * dim + (col_offset + j)] += block[i * 6 + j];
    }
  }
}

/// Accumulate a 6-vector into the global dense gradient b at the given offset.
void add_vec(std::vector<double>& b, int offset, const Vec6& v) {
  for (int i = 0; i < 6; ++i) {
    b[offset + i] += v[i];
  }
}

/// Compute J^T * Omega * J  (Gauss-Newton Hessian contribution from one pose).
Mat6 jtoj(const Mat6& J, const Mat6& omega) {
  return mat6_multiply(mat6_transpose(J), mat6_multiply(omega, J));
}

/// Compute J^T * Omega * r  (Gauss-Newton gradient contribution from one pose).
Vec6 jtor(const Mat6& J, const Mat6& omega, const Vec6& r) {
  return mat6_vec(mat6_transpose(J), mat6_vec(omega, r));
}

}  // namespace

/// Compute the 6x6 Jacobians of the residual with respect to right-perturbations
/// of T_i and T_j using central finite differences.
///
/// For each degree of freedom k ∈ [0,6):
///   J_i[:, k] = (r(T_i * Exp(ε e_k), T_j) - r(T_i * Exp(-ε e_k), T_j)) / 2ε
///   J_j[:, k] = (r(T_i, T_j * Exp(ε e_k)) - r(T_i, T_j * Exp(-ε e_k))) / 2ε
///
/// where e_k is the k-th standard basis vector in se(3) and ε = kFiniteDiffEps.
/// This gives second-order accurate Jacobians without deriving closed-form
/// analytical expressions.
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

/// Run one Gauss-Newton linearization pass over the full pose graph.
///
/// For a graph with N nodes, builds:
///   - H:  dense 6N x 6N Hessian approximation (row-major)
///   - b:  dense 6N gradient vector
///   - total_error:  sum of r^T Omega r over all edges
///
/// Each edge (i, j) with measurement Z_ij and information matrix Omega
/// contributes four 6x6 blocks to H and two 6-vectors to b:
///   H_ii += J_i^T Ω J_i      H_ij += J_i^T Ω J_j
///   H_ji += J_j^T Ω J_i      H_jj += J_j^T Ω J_j
///   b_i  += J_i^T Ω r        b_j  += J_j^T Ω r
///
/// Node IDs from the .g2o file are mapped to sequential indices [0, N) so that
/// node i occupies rows/columns [6i, 6i+6) in H and b.
LinearizationResult linearize(const PoseGraph& graph) {
  const int N = static_cast<int>(graph.nodes.size());
  const int dim = 6 * N;

  LinearizationResult result;
  result.dim = dim;
  result.H.assign(static_cast<std::size_t>(dim) * dim, 0.0);
  result.b.assign(dim, 0.0);
  result.total_error = 0.0;

  std::unordered_map<int, int> id_to_index;
  for (int idx = 0; idx < N; ++idx) {
    id_to_index[graph.nodes[idx].id] = idx;
  }

  std::vector<SE3> poses(N);
  for (int idx = 0; idx < N; ++idx) {
    poses[idx] = to_SE3(graph.nodes[idx]);
  }

  for (const auto& edge : graph.edges) {
    const int idx_i = id_to_index.at(edge.from);
    const int idx_j = id_to_index.at(edge.to);

    const SE3& T_i = poses[idx_i];
    const SE3& T_j = poses[idx_j];
    const SE3 Z_ij = to_SE3(edge);

    const se3 r = compute_residual(T_i, T_j, Z_ij);

    Mat6 J_i{}, J_j{};
    compute_jacobians(T_i, T_j, Z_ij, J_i, J_j);

    Mat6 omega{};
    expand_information(edge.information, omega);

    // r^T Omega r
    const Vec6 omega_r = mat6_vec(omega, r.v);
    for (int d = 0; d < 6; ++d) {
      result.total_error += r[d] * omega_r[d];
    }

    const int off_i = 6 * idx_i;
    const int off_j = 6 * idx_j;

    add_block(result.H, dim, off_i, off_i, jtoj(J_i, omega));
    add_block(result.H, dim, off_i, off_j,
              mat6_multiply(mat6_transpose(J_i), mat6_multiply(omega, J_j)));
    add_block(result.H, dim, off_j, off_i,
              mat6_multiply(mat6_transpose(J_j), mat6_multiply(omega, J_i)));
    add_block(result.H, dim, off_j, off_j, jtoj(J_j, omega));

    add_vec(result.b, off_i, jtor(J_i, omega, r.v));
    add_vec(result.b, off_j, jtor(J_j, omega, r.v));
  }

  return result;
}

}  // namespace cugraphopt
