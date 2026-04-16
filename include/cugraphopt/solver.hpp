#pragma once

#include "cugraphopt/bsr.hpp"
#include "cugraphopt/linearization.hpp"
#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/se3.hpp"

#include <vector>

namespace cugraphopt {

/// Solve a dense symmetric positive-definite system H*x = rhs using Cholesky
/// decomposition (in-place).  H is dim x dim row-major; rhs/x are length dim.
/// Returns false if H is not positive definite (decomposition fails).
bool dense_cholesky_solve(std::vector<double>& H, std::vector<double>& rhs,
                          int dim);

/// Apply a gauge fix by pinning the first pose: zero out the first 6 rows and
/// columns of H, set the 6x6 diagonal block to identity, and zero b[0..5].
/// This removes the rank deficiency (gauge freedom) in the Hessian.
void apply_gauge_fix(std::vector<double>& H, std::vector<double>& b, int dim);

/// Retract all poses on the SE(3) manifold:
///   T_new[i] = T_old[i] * Exp(dx[6*i : 6*i+6])
/// Modifies graph.nodes in-place.
void retract_poses(PoseGraph& graph, const std::vector<double>& dx);

/// Per-iteration statistics from the Gauss-Newton solver.
struct GNIterationStats {
  int iteration;
  double error;
  double linearize_ms;
  double solve_ms;
  double retract_ms;
  double total_ms;
};

/// Configuration for the Gauss-Newton solver.
struct GNConfig {
  int max_iterations = 30;
  double error_tolerance = 1e-6;      // stop when |error_change| / error < tol
  double gradient_tolerance = 1e-8;   // stop when ||b||_inf < tol
  bool verbose = false;
};

/// Result of a Gauss-Newton optimization run.
struct GNResult {
  double initial_error;
  double final_error;
  int iterations;
  std::vector<GNIterationStats> stats;
};

/// Run the full Gauss-Newton optimization on a pose graph (dense Cholesky).
/// Modifies graph.nodes in-place to the optimized poses.
GNResult solve_gauss_newton(PoseGraph& graph, const GNConfig& config);

/// Apply gauge fix to a BSR Hessian: zero first pose's rows/cols, set diagonal
/// to identity, and zero the first 6 gradient entries.
void apply_gauge_fix_bsr(BSRMatrix& bsr, std::vector<double>& b);

/// Run sparse Gauss-Newton optimization using BSR assembly + PCG solver.
/// Modifies graph.nodes in-place to the optimized poses.
GNResult solve_gauss_newton_sparse(PoseGraph& graph, const GNConfig& config);

}  // namespace cugraphopt
