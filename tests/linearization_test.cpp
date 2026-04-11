#include "cugraphopt/linearization.hpp"

#include <cassert>
#include <cmath>

namespace {

constexpr double kTol = 1e-6;

bool near(double a, double b, double tol = kTol) {
  return std::abs(a - b) < tol;
}

void test_expand_information_diagonal() {
  using namespace cugraphopt;
  std::array<double, 21> upper{};
  upper[0] = 1;
  upper[6] = 2;
  upper[11] = 3;
  upper[15] = 4;
  upper[18] = 5;
  upper[20] = 6;

  Mat6 omega{};
  expand_information(upper, omega);

  assert(near(omega[0 * 6 + 0], 1));
  assert(near(omega[1 * 6 + 1], 2));
  assert(near(omega[2 * 6 + 2], 3));
  assert(near(omega[3 * 6 + 3], 4));
  assert(near(omega[4 * 6 + 4], 5));
  assert(near(omega[5 * 6 + 5], 6));

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      assert(near(omega[i * 6 + j], omega[j * 6 + i]));
    }
  }
}

void test_expand_information_full() {
  using namespace cugraphopt;
  std::array<double, 21> upper;
  for (int i = 0; i < 21; ++i) {
    upper[i] = i + 1.0;
  }

  Mat6 omega{};
  expand_information(upper, omega);

  assert(near(omega[0 * 6 + 0], 1));
  assert(near(omega[0 * 6 + 1], 2));
  assert(near(omega[0 * 6 + 5], 6));
  assert(near(omega[1 * 6 + 0], 2));
  assert(near(omega[1 * 6 + 1], 7));
  assert(near(omega[5 * 6 + 5], 21));
}

void test_residual_identity_consistent() {
  using namespace cugraphopt;
  const SE3 T_i;
  const SE3 T_j = {SO3{}, {1, 0, 0}};
  const SE3 Z_ij = T_j;

  const se3 r = compute_residual(T_i, T_j, Z_ij);
  for (int i = 0; i < 6; ++i) {
    assert(near(r[i], 0.0, 1e-10));
  }
}

void test_residual_with_error() {
  using namespace cugraphopt;
  const SE3 T_i;
  const SE3 T_j = {SO3{}, {2, 0, 0}};
  const SE3 Z_ij = {SO3{}, {1, 0, 0}};

  const se3 r = compute_residual(T_i, T_j, Z_ij);

  for (int i = 0; i < 6; ++i) {
    assert(std::isfinite(r[i]));
  }
  assert(near(r[0], 1.0, 1e-9));
  assert(near(r[1], 0.0, 1e-9));
  assert(near(r[2], 0.0, 1e-9));
}

void test_jacobians_dimensions() {
  using namespace cugraphopt;
  const SE3 T_i;
  const SE3 T_j = {SO3{}, {1, 0, 0}};
  const SE3 Z_ij = T_j;

  Mat6 J_i{}, J_j{};
  compute_jacobians(T_i, T_j, Z_ij, J_i, J_j);

  for (int i = 0; i < 36; ++i) {
    assert(std::isfinite(J_i[i]));
    assert(std::isfinite(J_j[i]));
  }

  bool has_nonzero = false;
  for (int i = 0; i < 6; ++i) {
    if (std::abs(J_j[i * 6 + i]) > 0.1) has_nonzero = true;
  }
  assert(has_nonzero);
}

void test_jacobians_identity_case() {
  using namespace cugraphopt;
  // T_i = T_j = Z = identity => J_i = -I, J_j = I
  const SE3 T_i;
  const SE3 T_j;
  const SE3 Z_ij;

  Mat6 J_i{}, J_j{};
  compute_jacobians(T_i, T_j, Z_ij, J_i, J_j);

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      const double expected_i = (i == j) ? -1.0 : 0.0;
      const double expected_j = (i == j) ? 1.0 : 0.0;
      assert(near(J_i[i * 6 + j], expected_i, 1e-5));
      assert(near(J_j[i * 6 + j], expected_j, 1e-5));
    }
  }
}

void test_linearize_two_nodes_consistent() {
  using namespace cugraphopt;

  PoseGraph graph;
  graph.nodes.push_back({0, 0, 0, 0, 0, 0, 0, 1});
  graph.nodes.push_back({1, 1, 0, 0, 0, 0, 0, 1});

  Pose3Edge edge;
  edge.from = 0;
  edge.to = 1;
  edge.x = 1;
  edge.y = 0;
  edge.z = 0;
  edge.qx = 0;
  edge.qy = 0;
  edge.qz = 0;
  edge.qw = 1;
  edge.information = {};
  edge.information[0] = 1;
  edge.information[6] = 1;
  edge.information[11] = 1;
  edge.information[15] = 1;
  edge.information[18] = 1;
  edge.information[20] = 1;
  graph.edges.push_back(edge);

  const auto result = linearize(graph);

  assert(result.dim == 12);
  assert(static_cast<int>(result.H.size()) == 12 * 12);
  assert(static_cast<int>(result.b.size()) == 12);
  assert(near(result.total_error, 0.0, 1e-10));

  double h_diag_sum = 0;
  for (int i = 0; i < 12; ++i) {
    h_diag_sum += std::abs(result.H[i * 12 + i]);
  }
  assert(h_diag_sum > 0.1);
}

void test_linearize_two_nodes_with_error() {
  using namespace cugraphopt;

  PoseGraph graph;
  graph.nodes.push_back({0, 0, 0, 0, 0, 0, 0, 1});
  graph.nodes.push_back({1, 2, 0, 0, 0, 0, 0, 1});

  Pose3Edge edge;
  edge.from = 0;
  edge.to = 1;
  edge.x = 1;
  edge.y = 0;
  edge.z = 0;
  edge.qx = 0;
  edge.qy = 0;
  edge.qz = 0;
  edge.qw = 1;
  edge.information = {};
  edge.information[0] = 1;
  edge.information[6] = 1;
  edge.information[11] = 1;
  edge.information[15] = 1;
  edge.information[18] = 1;
  edge.information[20] = 1;
  graph.edges.push_back(edge);

  const auto result = linearize(graph);

  assert(result.dim == 12);
  assert(near(result.total_error, 1.0, 1e-6));
  assert(std::isfinite(result.total_error));

  double b_norm_sq = 0;
  for (int i = 0; i < 12; ++i) {
    b_norm_sq += result.b[i] * result.b[i];
  }
  assert(b_norm_sq > 1e-10);
}

}  // namespace

int main() {
  test_expand_information_diagonal();
  test_expand_information_full();
  test_residual_identity_consistent();
  test_residual_with_error();
  test_jacobians_dimensions();
  test_jacobians_identity_case();
  test_linearize_two_nodes_consistent();
  test_linearize_two_nodes_with_error();
  return 0;
}
