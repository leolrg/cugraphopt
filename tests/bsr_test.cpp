#include "cugraphopt/bsr.hpp"
#include "cugraphopt/linearization.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

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

  // Slightly inconsistent loop closure
  e12.from = 1; e12.to = 2;
  e12.x = -1; e12.y = 1.2; e12.z = 0;
  e12.qx = 0; e12.qy = 0; e12.qz = 0; e12.qw = 1;
  e12.information = make_info();

  graph.edges.push_back(e01);
  graph.edges.push_back(e02);
  graph.edges.push_back(e12);

  return graph;
}

static void test_bsr_symbolic() {
  PoseGraph graph = make_triangle_graph();
  BSRMatrix bsr = bsr_symbolic(graph);

  // 3 nodes -> 3 block-rows.
  assert(bsr.num_block_rows == 3);

  // Triangle: each node connects to all others.
  // Row 0: blocks at cols 0, 1, 2 (3 blocks)
  // Row 1: blocks at cols 0, 1, 2 (3 blocks)
  // Row 2: blocks at cols 0, 1, 2 (3 blocks)
  // Total: 9 blocks.
  assert(bsr.nnz_blocks == 9);
  assert(bsr.row_ptr[0] == 0);
  assert(bsr.row_ptr[1] == 3);
  assert(bsr.row_ptr[2] == 6);
  assert(bsr.row_ptr[3] == 9);

  // Check col_idx sorted within each row.
  for (int i = 0; i < 3; ++i) {
    for (int k = bsr.row_ptr[i]; k < bsr.row_ptr[i + 1] - 1; ++k) {
      assert(bsr.col_idx[k] < bsr.col_idx[k + 1]);
    }
  }

  // find_block should work.
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      assert(bsr.find_block(i, j) >= 0);
    }
  }

  std::printf("PASS: test_bsr_symbolic (nnz_blocks=%d)\n", bsr.nnz_blocks);
}

static void test_bsr_assemble_matches_dense() {
  PoseGraph graph = make_triangle_graph();

  // Dense assembly.
  LinearizationResult dense = linearize(graph);

  // BSR assembly.
  BSRMatrix bsr = bsr_symbolic(graph);
  std::vector<double> gradient;
  double bsr_error = bsr_assemble(bsr, gradient, graph);

  // Errors should match.
  assert(std::abs(bsr_error - dense.total_error) < 1e-10);

  // Gradient should match.
  assert(static_cast<int>(gradient.size()) == dense.dim);
  for (int i = 0; i < dense.dim; ++i) {
    assert(std::abs(gradient[i] - dense.b[i]) < 1e-10);
  }

  // BSR H blocks should match corresponding dense H entries.
  int N = bsr.num_block_rows;
  for (int bi = 0; bi < N; ++bi) {
    for (int k = bsr.row_ptr[bi]; k < bsr.row_ptr[bi + 1]; ++k) {
      int bj = bsr.col_idx[k];
      const double* blk = bsr.block(k);
      for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 6; ++c) {
          double bsr_val = blk[r * 6 + c];
          double dense_val = dense.H[(6 * bi + r) * dense.dim + (6 * bj + c)];
          if (std::abs(bsr_val - dense_val) > 1e-10) {
            std::printf("FAIL: block(%d,%d) r=%d c=%d bsr=%.12e dense=%.12e\n",
                        bi, bj, r, c, bsr_val, dense_val);
            assert(false);
          }
        }
      }
    }
  }

  std::printf("PASS: test_bsr_assemble_matches_dense (error=%.4e)\n", bsr_error);
}

static void test_bsr_spmv() {
  PoseGraph graph = make_triangle_graph();
  BSRMatrix bsr = bsr_symbolic(graph);
  std::vector<double> gradient;
  bsr_assemble(bsr, gradient, graph);

  int dim = 6 * bsr.num_block_rows;

  // Create a test vector x = [1, 2, 3, ...].
  std::vector<double> x(dim);
  for (int i = 0; i < dim; ++i) x[i] = i + 1.0;

  // BSR SpMV.
  std::vector<double> y_bsr(dim);
  bsr_spmv(bsr, x.data(), y_bsr.data());

  // Dense SpMV for comparison.
  LinearizationResult dense = linearize(graph);
  std::vector<double> y_dense(dim, 0.0);
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      y_dense[i] += dense.H[i * dim + j] * x[j];
    }
  }

  // Compare.
  for (int i = 0; i < dim; ++i) {
    if (std::abs(y_bsr[i] - y_dense[i]) > 1e-8) {
      std::printf("FAIL: spmv[%d] bsr=%.12e dense=%.12e\n", i, y_bsr[i],
                  y_dense[i]);
      assert(false);
    }
  }

  std::printf("PASS: test_bsr_spmv\n");
}

static void test_diagonal_inv() {
  PoseGraph graph = make_triangle_graph();
  BSRMatrix bsr = bsr_symbolic(graph);
  std::vector<double> gradient;
  bsr_assemble(bsr, gradient, graph);

  std::vector<double> diag_inv = bsr_extract_diagonal_inv(bsr);

  // For each diagonal block, verify that block * inv ≈ I.
  for (int i = 0; i < bsr.num_block_rows; ++i) {
    int k = bsr.find_block(i, i);
    const double* blk = bsr.block(k);
    const double* inv = diag_inv.data() + i * 36;

    // Multiply blk * inv.
    for (int r = 0; r < 6; ++r) {
      for (int c = 0; c < 6; ++c) {
        double s = 0.0;
        for (int p = 0; p < 6; ++p) {
          s += blk[r * 6 + p] * inv[p * 6 + c];
        }
        double expected = (r == c) ? 1.0 : 0.0;
        if (std::abs(s - expected) > 1e-8) {
          std::printf("FAIL: diag_inv block %d (%d,%d) = %.12e, expected %.1f\n",
                      i, r, c, s, expected);
          assert(false);
        }
      }
    }
  }

  std::printf("PASS: test_diagonal_inv\n");
}

static void test_pcg_solve() {
  PoseGraph graph = make_triangle_graph();
  BSRMatrix bsr = bsr_symbolic(graph);
  std::vector<double> gradient;
  bsr_assemble(bsr, gradient, graph);

  int N = bsr.num_block_rows;
  int dim = 6 * N;

  // Apply gauge fix: zero first 6 rows/cols of H, set diagonal to I.
  // Also zero first 6 entries of gradient.
  for (int i = 0; i < 6; ++i) {
    gradient[i] = 0.0;
  }
  // Zero all blocks in row 0 and column 0.
  for (int k = bsr.row_ptr[0]; k < bsr.row_ptr[0 + 1]; ++k) {
    double* blk = bsr.block(k);
    for (int a = 0; a < 36; ++a) blk[a] = 0.0;
    if (bsr.col_idx[k] == 0) {
      // Set diagonal to identity.
      for (int a = 0; a < 6; ++a) blk[a * 6 + a] = 1.0;
    }
  }
  // Zero column 0 blocks in other rows.
  for (int i = 1; i < N; ++i) {
    int k = bsr.find_block(i, 0);
    if (k >= 0) {
      double* blk = bsr.block(k);
      for (int a = 0; a < 36; ++a) blk[a] = 0.0;
    }
  }

  // Negate gradient to form rhs = -b.
  std::vector<double> rhs(dim);
  for (int i = 0; i < dim; ++i) rhs[i] = -gradient[i];

  // Solve with PCG.
  std::vector<double> dx;
  int iters = pcg_solve(bsr, rhs, dx, 200, 1e-10);

  // Verify: H * dx ≈ rhs.
  std::vector<double> check(dim);
  bsr_spmv(bsr, dx.data(), check.data());

  double max_err = 0.0;
  for (int i = 0; i < dim; ++i) {
    max_err = std::max(max_err, std::abs(check[i] - rhs[i]));
  }

  std::printf("PASS: test_pcg_solve (iters=%d, max_residual=%.2e)\n",
              iters, max_err);
  assert(max_err < 1e-6);
}

int main() {
  test_bsr_symbolic();
  test_bsr_assemble_matches_dense();
  test_bsr_spmv();
  test_diagonal_inv();
  test_pcg_solve();
  std::printf("\nAll BSR tests passed.\n");
  return 0;
}
