#include "cugraphopt/solver.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

using namespace cugraphopt;

static void test_cholesky_solve() {
  // 2x2 SPD system: H = [[4, 2], [2, 3]], b = [1, 2]
  // Solution: x = [-0.125, 0.75]
  std::vector<double> H = {4, 2, 2, 3};
  std::vector<double> b = {1, 2};
  bool ok = dense_cholesky_solve(H, b, 2);
  assert(ok);
  assert(std::abs(b[0] - (-0.125)) < 1e-12);
  assert(std::abs(b[1] - 0.75) < 1e-12);
  std::printf("PASS: test_cholesky_solve\n");
}

static void test_cholesky_not_pd() {
  // Not positive definite: H = [[1, 2], [2, 1]]
  std::vector<double> H = {1, 2, 2, 1};
  std::vector<double> b = {1, 1};
  bool ok = dense_cholesky_solve(H, b, 2);
  assert(!ok);
  std::printf("PASS: test_cholesky_not_pd\n");
}

static void test_gauge_fix() {
  // 12x12 identity matrix, b = all 1s.
  int dim = 12;
  std::vector<double> H(dim * dim, 0.0);
  std::vector<double> b(dim, 1.0);
  for (int i = 0; i < dim; ++i) H[i * dim + i] = 1.0;

  apply_gauge_fix(H, b, dim);

  // First 6 rows/cols should be identity (diagonal=1, off-diag=0).
  for (int i = 0; i < 6; ++i) {
    assert(b[i] == 0.0);
    for (int j = 0; j < dim; ++j) {
      double expected = (i == j) ? 1.0 : 0.0;
      assert(H[i * dim + j] == expected);
      assert(H[j * dim + i] == expected);
    }
  }
  // Remaining should be untouched.
  for (int i = 6; i < dim; ++i) {
    assert(b[i] == 1.0);
    assert(H[i * dim + i] == 1.0);
  }
  std::printf("PASS: test_gauge_fix\n");
}

static void test_retract_identity() {
  // Retract with dx = 0 should leave poses unchanged.
  PoseGraph graph;
  Pose3Node n;
  n.id = 0;
  n.x = 1.0; n.y = 2.0; n.z = 3.0;
  n.qx = 0.0; n.qy = 0.0; n.qz = 0.0; n.qw = 1.0;
  graph.nodes.push_back(n);

  std::vector<double> dx(6, 0.0);
  retract_poses(graph, dx);

  assert(std::abs(graph.nodes[0].x - 1.0) < 1e-12);
  assert(std::abs(graph.nodes[0].y - 2.0) < 1e-12);
  assert(std::abs(graph.nodes[0].z - 3.0) < 1e-12);
  assert(std::abs(graph.nodes[0].qw - 1.0) < 1e-10);
  std::printf("PASS: test_retract_identity\n");
}

static void test_solve_consistent_graph() {
  // Two poses connected by a consistent edge -> error should be ~0 already.
  PoseGraph graph;
  Pose3Node n0, n1;
  n0.id = 0; n0.x = 0; n0.y = 0; n0.z = 0;
  n0.qx = 0; n0.qy = 0; n0.qz = 0; n0.qw = 1;
  n1.id = 1; n1.x = 1; n1.y = 0; n1.z = 0;
  n1.qx = 0; n1.qy = 0; n1.qz = 0; n1.qw = 1;
  graph.nodes.push_back(n0);
  graph.nodes.push_back(n1);

  Pose3Edge e;
  e.from = 0; e.to = 1;
  e.x = 1; e.y = 0; e.z = 0;
  e.qx = 0; e.qy = 0; e.qz = 0; e.qw = 1;
  e.information = {};
  // Identity information: diagonal = 1
  e.information[0] = 1; e.information[6] = 1; e.information[11] = 1;
  e.information[15] = 1; e.information[18] = 1; e.information[20] = 1;
  graph.edges.push_back(e);

  GNConfig cfg;
  cfg.max_iterations = 5;
  cfg.verbose = false;
  GNResult res = solve_gauss_newton(graph, cfg);

  assert(res.initial_error < 1e-10);
  assert(res.final_error < 1e-10);
  std::printf("PASS: test_solve_consistent_graph (error=%.2e)\n",
              res.final_error);
}

static void test_solve_inconsistent_graph() {
  // Three poses in a triangle with a slight inconsistency. The solver should
  // reduce the total error.
  PoseGraph graph;

  // Pose 0 at origin
  Pose3Node n0;
  n0.id = 0; n0.x = 0; n0.y = 0; n0.z = 0;
  n0.qx = 0; n0.qy = 0; n0.qz = 0; n0.qw = 1;
  graph.nodes.push_back(n0);

  // Pose 1 at (1, 0, 0)
  Pose3Node n1;
  n1.id = 1; n1.x = 1; n1.y = 0; n1.z = 0;
  n1.qx = 0; n1.qy = 0; n1.qz = 0; n1.qw = 1;
  graph.nodes.push_back(n1);

  // Pose 2 at (0, 1, 0) — but measurement from 1->2 says (0, 1.5, 0) relative
  Pose3Node n2;
  n2.id = 2; n2.x = 0; n2.y = 1; n2.z = 0;
  n2.qx = 0; n2.qy = 0; n2.qz = 0; n2.qw = 1;
  graph.nodes.push_back(n2);

  auto make_info = []() -> std::array<double, 21> {
    std::array<double, 21> info{};
    info[0] = 1; info[6] = 1; info[11] = 1;
    info[15] = 1; info[18] = 1; info[20] = 1;
    return info;
  };

  // Edge 0->1: consistent measurement (1, 0, 0)
  Pose3Edge e01;
  e01.from = 0; e01.to = 1;
  e01.x = 1; e01.y = 0; e01.z = 0;
  e01.qx = 0; e01.qy = 0; e01.qz = 0; e01.qw = 1;
  e01.information = make_info();
  graph.edges.push_back(e01);

  // Edge 0->2: consistent measurement (0, 1, 0)
  Pose3Edge e02;
  e02.from = 0; e02.to = 2;
  e02.x = 0; e02.y = 1; e02.z = 0;
  e02.qx = 0; e02.qy = 0; e02.qz = 0; e02.qw = 1;
  e02.information = make_info();
  graph.edges.push_back(e02);

  // Edge 1->2: INCONSISTENT measurement (-1, 1.5, 0) instead of (-1, 1, 0)
  Pose3Edge e12;
  e12.from = 1; e12.to = 2;
  e12.x = -1; e12.y = 1.5; e12.z = 0;
  e12.qx = 0; e12.qy = 0; e12.qz = 0; e12.qw = 1;
  e12.information = make_info();
  graph.edges.push_back(e12);

  double initial_error_check = linearize(graph).total_error;

  GNConfig cfg;
  cfg.max_iterations = 30;
  cfg.verbose = false;
  GNResult res = solve_gauss_newton(graph, cfg);

  assert(res.initial_error > 0.0);
  // Error should decrease.
  assert(res.final_error < initial_error_check);
  std::printf("PASS: test_solve_inconsistent_graph (%.4e -> %.4e in %d iters)\n",
              res.initial_error, res.final_error, res.iterations);
}

int main() {
  test_cholesky_solve();
  test_cholesky_not_pd();
  test_gauge_fix();
  test_retract_identity();
  test_solve_consistent_graph();
  test_solve_inconsistent_graph();
  std::printf("\nAll solver tests passed.\n");
  return 0;
}
