#pragma once

#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/solver.hpp"

namespace cugraphopt {

/// Full GPU Gauss-Newton solver. Pure C++ interface (CUDA internals hidden).
/// Modifies graph.nodes in-place to the optimized poses.
GNResult solve_gauss_newton_gpu(PoseGraph& graph, const GNConfig& config);

/// GPU Levenberg-Marquardt solver with adaptive damping.
/// Adds lambda * diag(H) to the Hessian for trust-region behavior.
GNResult solve_lm_gpu(PoseGraph& graph, const GNConfig& config);

/// GPU Gauss-Newton solver using cuDSS (NVIDIA sparse direct solver)
/// instead of PCG. Exact solve each iteration -> fewer GN iterations needed.
GNResult solve_gauss_newton_cudss(PoseGraph& graph, const GNConfig& config);

}  // namespace cugraphopt
