#pragma once

#include "cugraphopt/pose_graph.hpp"
#include "cugraphopt/solver.hpp"

namespace cugraphopt {

/// Full GPU Gauss-Newton solver. Pure C++ interface (CUDA internals hidden).
/// Modifies graph.nodes in-place to the optimized poses.
GNResult solve_gauss_newton_gpu(PoseGraph& graph, const GNConfig& config);

}  // namespace cugraphopt
