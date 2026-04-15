#include "cugraphopt/bsr.hpp"
#include "cugraphopt/linearization.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <unordered_map>

namespace cugraphopt {

int BSRMatrix::find_block(int block_row, int block_col) const {
  // Binary search within the block-row.
  int lo = row_ptr[block_row];
  int hi = row_ptr[block_row + 1];
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (col_idx[mid] < block_col) {
      lo = mid + 1;
    } else if (col_idx[mid] > block_col) {
      hi = mid;
    } else {
      return mid;
    }
  }
  return -1;
}

BSRMatrix bsr_symbolic(const PoseGraph& graph) {
  const int N = static_cast<int>(graph.nodes.size());

  // Map node IDs to sequential indices.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < N; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  // Collect non-zero block positions using a set per row for sorted order.
  std::vector<std::set<int>> row_blocks(N);

  // Diagonal blocks always present.
  for (int i = 0; i < N; ++i) {
    row_blocks[i].insert(i);
  }

  // Each edge (i,j) contributes blocks (i,i), (i,j), (j,i), (j,j).
  for (const auto& edge : graph.edges) {
    int i = id_to_idx.at(edge.from);
    int j = id_to_idx.at(edge.to);
    row_blocks[i].insert(i);
    row_blocks[i].insert(j);
    row_blocks[j].insert(i);
    row_blocks[j].insert(j);
  }

  BSRMatrix bsr;
  bsr.num_block_rows = N;
  bsr.row_ptr.resize(N + 1);

  // Build row_ptr and col_idx.
  bsr.row_ptr[0] = 0;
  for (int i = 0; i < N; ++i) {
    bsr.row_ptr[i + 1] = bsr.row_ptr[i] + static_cast<int>(row_blocks[i].size());
    for (int col : row_blocks[i]) {
      bsr.col_idx.push_back(col);
    }
  }

  bsr.nnz_blocks = static_cast<int>(bsr.col_idx.size());
  bsr.values.assign(bsr.nnz_blocks * 36, 0.0);

  return bsr;
}

namespace {

/// Add a 6x6 block contribution to the BSR block at position k.
void add_block_bsr(BSRMatrix& bsr, int k, const Mat6& block) {
  double* dst = bsr.block(k);
  for (int i = 0; i < 36; ++i) {
    dst[i] += block[i];
  }
}

/// Multiply: C = A^T * B for 6x6 matrices (row-major).
Mat6 mat6_atb(const Mat6& A, const Mat6& B) {
  Mat6 C{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      double s = 0.0;
      for (int k = 0; k < 6; ++k) {
        s += A[k * 6 + i] * B[k * 6 + j];  // A^T[i][k] * B[k][j]
      }
      C[i * 6 + j] = s;
    }
  }
  return C;
}

}  // namespace

double bsr_assemble(BSRMatrix& bsr, std::vector<double>& gradient,
                    const PoseGraph& graph) {
  const int N = static_cast<int>(graph.nodes.size());
  const int dim = 6 * N;

  // Zero out values and gradient.
  std::fill(bsr.values.begin(), bsr.values.end(), 0.0);
  gradient.assign(dim, 0.0);

  // Map node IDs to sequential indices.
  std::unordered_map<int, int> id_to_idx;
  for (int i = 0; i < N; ++i) {
    id_to_idx[graph.nodes[i].id] = i;
  }

  // Convert all poses.
  std::vector<SE3> poses(N);
  for (int i = 0; i < N; ++i) {
    poses[i] = to_SE3(graph.nodes[i]);
  }

  double total_error = 0.0;

  for (const auto& edge : graph.edges) {
    const int idx_i = id_to_idx.at(edge.from);
    const int idx_j = id_to_idx.at(edge.to);

    const SE3& T_i = poses[idx_i];
    const SE3& T_j = poses[idx_j];
    const SE3 Z_ij = to_SE3(edge);

    // Residual.
    const se3 r = compute_residual(T_i, T_j, Z_ij);

    // Jacobians.
    Mat6 J_i{}, J_j{};
    compute_jacobians(T_i, T_j, Z_ij, J_i, J_j);

    // Information matrix.
    Mat6 omega{};
    expand_information(edge.information, omega);

    // Error: r^T Omega r.
    const Vec6 omega_r = mat6_vec(omega, r.v);
    for (int d = 0; d < 6; ++d) {
      total_error += r[d] * omega_r[d];
    }

    // Precompute Omega * J_i and Omega * J_j.
    const Mat6 omega_Ji = mat6_multiply(omega, J_i);
    const Mat6 omega_Jj = mat6_multiply(omega, J_j);

    // H blocks: J_i^T * Omega * J_i, J_i^T * Omega * J_j, etc.
    int k_ii = bsr.find_block(idx_i, idx_i);
    int k_ij = bsr.find_block(idx_i, idx_j);
    int k_ji = bsr.find_block(idx_j, idx_i);
    int k_jj = bsr.find_block(idx_j, idx_j);

    assert(k_ii >= 0 && k_ij >= 0 && k_ji >= 0 && k_jj >= 0);

    add_block_bsr(bsr, k_ii, mat6_atb(J_i, omega_Ji));
    add_block_bsr(bsr, k_ij, mat6_atb(J_i, omega_Jj));
    add_block_bsr(bsr, k_ji, mat6_atb(J_j, omega_Ji));
    add_block_bsr(bsr, k_jj, mat6_atb(J_j, omega_Jj));

    // Gradient: b_i += J_i^T * Omega * r, b_j += J_j^T * Omega * r.
    for (int d = 0; d < 6; ++d) {
      double gi = 0.0, gj = 0.0;
      for (int k = 0; k < 6; ++k) {
        gi += J_i[k * 6 + d] * omega_r[k];  // J_i^T[d][k] * (Omega*r)[k]
        gj += J_j[k * 6 + d] * omega_r[k];
      }
      gradient[6 * idx_i + d] += gi;
      gradient[6 * idx_j + d] += gj;
    }
  }

  return total_error;
}

void bsr_spmv(const BSRMatrix& bsr, const double* x, double* y) {
  const int N = bsr.num_block_rows;

  for (int i = 0; i < N; ++i) {
    // Zero the output for this block-row.
    for (int d = 0; d < 6; ++d) {
      y[6 * i + d] = 0.0;
    }

    // Iterate over non-zero blocks in this row.
    for (int k = bsr.row_ptr[i]; k < bsr.row_ptr[i + 1]; ++k) {
      int j = bsr.col_idx[k];
      const double* blk = bsr.block(k);
      const double* xj = x + 6 * j;

      // y[6i:6i+6] += block * x[6j:6j+6]
      for (int r = 0; r < 6; ++r) {
        double s = 0.0;
        for (int c = 0; c < 6; ++c) {
          s += blk[r * 6 + c] * xj[c];
        }
        y[6 * i + r] += s;
      }
    }
  }
}

std::vector<double> bsr_extract_diagonal_inv(const BSRMatrix& bsr) {
  const int N = bsr.num_block_rows;
  std::vector<double> diag_inv(N * 36);

  for (int i = 0; i < N; ++i) {
    int k = bsr.find_block(i, i);
    assert(k >= 0);
    const double* blk = bsr.block(k);

    // Copy diagonal block into a local 6x6 and invert via Cholesky.
    double L[36] = {};
    for (int a = 0; a < 36; ++a) L[a] = blk[a];

    // In-place Cholesky: L = lower triangle of decomposition.
    for (int j = 0; j < 6; ++j) {
      double sum = 0.0;
      for (int p = 0; p < j; ++p) sum += L[j * 6 + p] * L[j * 6 + p];
      double diag = L[j * 6 + j] - sum;
      if (diag <= 1e-30) {
        // Fallback: use identity if block is near-singular.
        for (int a = 0; a < 36; ++a) diag_inv[i * 36 + a] = 0.0;
        for (int a = 0; a < 6; ++a) diag_inv[i * 36 + a * 6 + a] = 1.0;
        goto next_block;
      }
      L[j * 6 + j] = std::sqrt(diag);
      for (int row = j + 1; row < 6; ++row) {
        double s = 0.0;
        for (int p = 0; p < j; ++p) s += L[row * 6 + p] * L[j * 6 + p];
        L[row * 6 + j] = (L[row * 6 + j] - s) / L[j * 6 + j];
      }
    }

    // Invert via forward/back substitution with identity RHS columns.
    {
      double inv[36] = {};
      for (int col = 0; col < 6; ++col) {
        double z[6] = {};
        z[col] = 1.0;

        // Forward: L * y = e_col
        for (int row = 0; row < 6; ++row) {
          double s = 0.0;
          for (int p = 0; p < row; ++p) s += L[row * 6 + p] * z[p];
          z[row] = (z[row] - s) / L[row * 6 + row];
        }

        // Back: L^T * x = y
        for (int row = 5; row >= 0; --row) {
          double s = 0.0;
          for (int p = row + 1; p < 6; ++p) s += L[p * 6 + row] * z[p];
          z[row] = (z[row] - s) / L[row * 6 + row];
        }

        for (int row = 0; row < 6; ++row) {
          inv[row * 6 + col] = z[row];
        }
      }

      for (int a = 0; a < 36; ++a) diag_inv[i * 36 + a] = inv[a];
    }

    next_block:;
  }

  return diag_inv;
}

int pcg_solve(const BSRMatrix& H, const std::vector<double>& rhs,
              std::vector<double>& dx, int max_iter, double tol) {
  const int dim = 6 * H.num_block_rows;
  dx.assign(dim, 0.0);

  // Preconditioner: inverted diagonal blocks.
  std::vector<double> M_inv = bsr_extract_diagonal_inv(H);

  // r = rhs - H * dx = rhs (since dx = 0).
  std::vector<double> r(rhs.begin(), rhs.end());

  // Apply preconditioner: z = M^{-1} r.
  auto apply_precond = [&](const std::vector<double>& src,
                           std::vector<double>& dst) {
    const int N = H.num_block_rows;
    for (int i = 0; i < N; ++i) {
      const double* blk = M_inv.data() + i * 36;
      for (int row = 0; row < 6; ++row) {
        double s = 0.0;
        for (int c = 0; c < 6; ++c) {
          s += blk[row * 6 + c] * src[6 * i + c];
        }
        dst[6 * i + row] = s;
      }
    }
  };

  std::vector<double> z(dim);
  apply_precond(r, z);

  // p = z.
  std::vector<double> p(z);

  // rz = r^T z.
  double rz = 0.0;
  for (int i = 0; i < dim; ++i) rz += r[i] * z[i];

  double rhs_norm = 0.0;
  for (int i = 0; i < dim; ++i) rhs_norm += rhs[i] * rhs[i];
  rhs_norm = std::sqrt(rhs_norm);
  if (rhs_norm < 1e-30) return 0;

  std::vector<double> Ap(dim);

  for (int iter = 0; iter < max_iter; ++iter) {
    // Ap = H * p.
    bsr_spmv(H, p.data(), Ap.data());

    // alpha = rz / (p^T Ap).
    double pAp = 0.0;
    for (int i = 0; i < dim; ++i) pAp += p[i] * Ap[i];
    if (std::abs(pAp) < 1e-30) break;
    double alpha = rz / pAp;

    // dx += alpha * p.
    // r -= alpha * Ap.
    for (int i = 0; i < dim; ++i) {
      dx[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    // Check convergence: ||r|| / ||rhs|| < tol.
    double r_norm = 0.0;
    for (int i = 0; i < dim; ++i) r_norm += r[i] * r[i];
    r_norm = std::sqrt(r_norm);
    if (r_norm / rhs_norm < tol) return iter + 1;

    // z = M^{-1} r.
    apply_precond(r, z);

    double rz_new = 0.0;
    for (int i = 0; i < dim; ++i) rz_new += r[i] * z[i];

    double beta = rz_new / rz;
    rz = rz_new;

    // p = z + beta * p.
    for (int i = 0; i < dim; ++i) {
      p[i] = z[i] + beta * p[i];
    }
  }

  return max_iter;
}

}  // namespace cugraphopt
