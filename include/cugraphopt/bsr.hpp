#pragma once

#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/se3.hpp"

#include <vector>

namespace cugraphopt {

/// Block Sparse Row (BSR) format for 6x6 blocks.
///
/// For a graph with N poses, the Hessian H is a 6N x 6N sparse matrix where
/// non-zero entries appear in 6x6 blocks. The sparsity pattern is determined
/// by the graph topology: block (i,j) is non-zero iff poses i and j are
/// connected by an edge (or i == j for diagonal blocks).
///
/// Storage:
///   row_ptr[i]   = index into col_idx/values where block-row i starts
///   row_ptr[N]   = total number of non-zero blocks (nnz_blocks)
///   col_idx[k]   = block-column index of the k-th non-zero block
///   values[k*36] = 36 doubles (6x6 row-major) for the k-th non-zero block
///
/// Block-rows are sorted by row index; within each row, blocks are sorted by
/// column index.
struct BSRMatrix {
  int num_block_rows = 0;       // N (number of poses)
  int nnz_blocks = 0;           // total non-zero 6x6 blocks
  std::vector<int> row_ptr;     // length N+1
  std::vector<int> col_idx;     // length nnz_blocks
  std::vector<double> values;   // length nnz_blocks * 36

  /// Return pointer to the 6x6 block at position k in values array.
  double* block(int k) { return values.data() + k * 36; }
  const double* block(int k) const { return values.data() + k * 36; }

  /// Find the index of block (block_row, block_col), or -1 if not present.
  int find_block(int block_row, int block_col) const;
};

/// Build the symbolic sparsity pattern of the BSR Hessian from the pose graph
/// topology. Allocates row_ptr, col_idx, and zeroed values. Each edge (i,j)
/// contributes blocks at (i,i), (i,j), (j,i), (j,j). Diagonal blocks are
/// always present.
BSRMatrix bsr_symbolic(const PoseGraph& graph);

/// Numerically assemble the BSR Hessian and gradient from a pose graph.
/// Computes residuals, Jacobians, and accumulates J^T Omega J blocks and
/// J^T Omega r gradient entries. Also returns total squared error.
///
/// Requires bsr to have been symbolically assembled (bsr_symbolic).
/// The values array is zeroed and re-filled.
/// gradient must be pre-allocated to length 6 * num_block_rows.
double bsr_assemble(BSRMatrix& bsr, std::vector<double>& gradient,
                    const PoseGraph& graph);

/// BSR sparse matrix-vector multiply: y = H * x.
/// x and y are vectors of length 6 * num_block_rows.
void bsr_spmv(const BSRMatrix& bsr, const double* x, double* y);

/// Extract diagonal 6x6 blocks from BSR matrix and compute their inverses.
/// Returns a flat array of N * 36 doubles (N inverted 6x6 blocks).
std::vector<double> bsr_extract_diagonal_inv(const BSRMatrix& bsr);

/// Solve H * dx = rhs using Preconditioned Conjugate Gradient with
/// Block-Jacobi preconditioner. Returns the number of iterations used.
/// dx is the output solution vector (length 6N).
/// max_iter: maximum CG iterations. tol: relative residual tolerance.
int pcg_solve(const BSRMatrix& H, const std::vector<double>& rhs,
              std::vector<double>& dx, int max_iter, double tol);

}  // namespace cugraphopt
