#include "cugraphopt/solver.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <unordered_map>

namespace cugraphopt {

bool dense_cholesky_solve(std::vector<double>& H, std::vector<double>& rhs,
                          int dim) {
  // In-place Cholesky decomposition: H = L * L^T (lower-triangular L stored
  // over the lower triangle of H).
  for (int j = 0; j < dim; ++j) {
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
      sum += H[j * dim + k] * H[j * dim + k];
    }
    double diag = H[j * dim + j] - sum;
    if (diag <= 0.0) {
      return false;  // not positive definite
    }
    H[j * dim + j] = std::sqrt(diag);

    for (int i = j + 1; i < dim; ++i) {
      double s = 0.0;
      for (int k = 0; k < j; ++k) {
        s += H[i * dim + k] * H[j * dim + k];
      }
      H[i * dim + j] = (H[i * dim + j] - s) / H[j * dim + j];
    }
  }

  // Forward substitution: L * y = rhs
  for (int i = 0; i < dim; ++i) {
    double s = 0.0;
    for (int k = 0; k < i; ++k) {
      s += H[i * dim + k] * rhs[k];
    }
    rhs[i] = (rhs[i] - s) / H[i * dim + i];
  }

  // Back substitution: L^T * x = y
  for (int i = dim - 1; i >= 0; --i) {
    double s = 0.0;
    for (int k = i + 1; k < dim; ++k) {
      s += H[k * dim + i] * rhs[k];
    }
    rhs[i] = (rhs[i] - s) / H[i * dim + i];
  }

  return true;
}

void apply_gauge_fix(std::vector<double>& H, std::vector<double>& b, int dim) {
  // Zero out first 6 rows and columns, set diagonal to 1.
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < dim; ++j) {
      H[i * dim + j] = 0.0;
      H[j * dim + i] = 0.0;
    }
    H[i * dim + i] = 1.0;
    b[i] = 0.0;
  }
}

void retract_poses(PoseGraph& graph, const std::vector<double>& dx) {
  for (int i = 0; i < static_cast<int>(graph.nodes.size()); ++i) {
    SE3 T_old = to_SE3(graph.nodes[i]);

    // Extract the 6-vector for this pose.
    se3 delta{};
    for (int k = 0; k < 6; ++k) {
      delta[k] = dx[6 * i + k];
    }

    // Retract: T_new = T_old * Exp(delta)
    SE3 T_new = compose(T_old, exp(delta));

    // Write back to the node (convert SE3 -> quaternion + translation).
    auto& n = graph.nodes[i];
    n.x = T_new.t[0];
    n.y = T_new.t[1];
    n.z = T_new.t[2];

    // Convert rotation matrix to quaternion.
    const auto& R = T_new.R.m;
    double trace = R[0] + R[4] + R[8];
    double qw, qx, qy, qz;

    if (trace > 0.0) {
      double s = 0.5 / std::sqrt(trace + 1.0);
      qw = 0.25 / s;
      qx = (R[7] - R[5]) * s;  // (R(2,1) - R(1,2))
      qy = (R[2] - R[6]) * s;  // (R(0,2) - R(2,0))
      qz = (R[3] - R[1]) * s;  // (R(1,0) - R(0,1))
    } else if (R[0] > R[4] && R[0] > R[8]) {
      double s = 2.0 * std::sqrt(1.0 + R[0] - R[4] - R[8]);
      qw = (R[7] - R[5]) / s;
      qx = 0.25 * s;
      qy = (R[1] + R[3]) / s;
      qz = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
      double s = 2.0 * std::sqrt(1.0 + R[4] - R[0] - R[8]);
      qw = (R[2] - R[6]) / s;
      qx = (R[1] + R[3]) / s;
      qy = 0.25 * s;
      qz = (R[5] + R[7]) / s;
    } else {
      double s = 2.0 * std::sqrt(1.0 + R[8] - R[0] - R[4]);
      qw = (R[3] - R[1]) / s;
      qx = (R[2] + R[6]) / s;
      qy = (R[5] + R[7]) / s;
      qz = 0.25 * s;
    }

    // Normalize quaternion.
    double norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    n.qx = qx / norm;
    n.qy = qy / norm;
    n.qz = qz / norm;
    n.qw = qw / norm;
  }
}

GNResult solve_gauss_newton(PoseGraph& graph, const GNConfig& config) {
  using Clock = std::chrono::high_resolution_clock;

  GNResult result{};

  for (int iter = 0; iter < config.max_iterations; ++iter) {
    GNIterationStats stats{};
    stats.iteration = iter;

    auto t_total_start = Clock::now();

    // --- Linearize ---
    auto t0 = Clock::now();
    LinearizationResult lin = linearize(graph);
    auto t1 = Clock::now();
    stats.linearize_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (iter == 0) {
      result.initial_error = lin.total_error;
    }
    stats.error = lin.total_error;

    // Check gradient convergence.
    double grad_max = 0.0;
    for (int i = 0; i < lin.dim; ++i) {
      grad_max = std::max(grad_max, std::abs(lin.b[i]));
    }

    if (config.verbose) {
      std::printf("iter %3d  error=%.6e  |grad|_inf=%.6e  linearize=%.1fms",
                  iter, lin.total_error, grad_max, stats.linearize_ms);
    }

    if (grad_max < config.gradient_tolerance) {
      if (config.verbose) {
        std::printf("  -> converged (gradient)\n");
      }
      result.final_error = lin.total_error;
      result.iterations = iter;
      result.stats.push_back(stats);
      break;
    }

    // --- Solve H * dx = -b ---
    auto t2 = Clock::now();
    apply_gauge_fix(lin.H, lin.b, lin.dim);

    // Negate b to solve H * dx = -b.
    for (int i = 0; i < lin.dim; ++i) {
      lin.b[i] = -lin.b[i];
    }

    bool ok = dense_cholesky_solve(lin.H, lin.b, lin.dim);
    auto t3 = Clock::now();
    stats.solve_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (!ok) {
      if (config.verbose) {
        std::printf("  -> Cholesky failed\n");
      }
      result.final_error = lin.total_error;
      result.iterations = iter;
      result.stats.push_back(stats);
      break;
    }

    // --- Retract ---
    auto t4 = Clock::now();
    retract_poses(graph, lin.b);  // lin.b now holds dx
    auto t5 = Clock::now();
    stats.retract_ms =
        std::chrono::duration<double, std::milli>(t5 - t4).count();

    auto t_total_end = Clock::now();
    stats.total_ms =
        std::chrono::duration<double, std::milli>(t_total_end - t_total_start)
            .count();

    if (config.verbose) {
      std::printf("  solve=%.1fms  retract=%.1fms  total=%.1fms\n",
                  stats.solve_ms, stats.retract_ms, stats.total_ms);
    }

    result.stats.push_back(stats);
    result.final_error = lin.total_error;
    result.iterations = iter + 1;

    // Check error convergence.
    if (iter > 0) {
      double prev_error = result.stats[iter - 1].error;
      double rel_change = std::abs(lin.total_error - prev_error) /
                          (std::abs(prev_error) + 1e-30);
      if (rel_change < config.error_tolerance) {
        if (config.verbose) {
          std::printf("  -> converged (error stagnation, rel=%.2e)\n",
                      rel_change);
        }
        break;
      }
    }
  }

  return result;
}

}  // namespace cugraphopt
