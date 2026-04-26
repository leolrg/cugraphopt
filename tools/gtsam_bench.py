#!/usr/bin/env python3
"""Benchmark GTSAM on g2o pose graph datasets for comparison with CuGraphOpt."""

import argparse
import time
import sys

import gtsam
import numpy as np
from gtsam import symbol_shorthand

def load_g2o_and_solve(filepath, max_iterations=30, verbose=True):
    """Load a g2o file into GTSAM, optimize, and return timing."""

    # Use GTSAM's built-in g2o reader
    graph, initial = gtsam.readG2o(filepath, is3D=True)

    if verbose:
        print(f"Loaded: {initial.size()} nodes, {graph.size()} factors")

    # Add a prior on the first pose to fix gauge
    keys = gtsam.KeyVector()
    for key in initial.keys():
        keys.append(key)

    first_key = keys[0]
    prior_model = gtsam.noiseModel.Diagonal.Variances(
        np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    )
    graph.addPriorPose3(first_key, initial.atPose3(first_key), prior_model)

    # Compute initial error
    initial_error = graph.error(initial)
    if verbose:
        print(f"Initial error: {initial_error:.6e}")

    # ---- Gauss-Newton ----
    params_gn = gtsam.GaussNewtonParams()
    params_gn.setMaxIterations(max_iterations)
    params_gn.setRelativeErrorTol(1e-6)
    if verbose:
        params_gn.setVerbosity("ERROR")

    t0 = time.perf_counter()
    optimizer_gn = gtsam.GaussNewtonOptimizer(graph, initial, params_gn)
    result_gn = optimizer_gn.optimize()
    t1 = time.perf_counter()
    gn_time = (t1 - t0) * 1000  # ms
    gn_error = graph.error(result_gn)
    gn_iters = optimizer_gn.iterations()

    if verbose:
        print(f"GN: {gn_iters} iters, error {initial_error:.4e} -> {gn_error:.4e}, "
              f"time {gn_time:.1f} ms")

    # ---- Levenberg-Marquardt ----
    params_lm = gtsam.LevenbergMarquardtParams()
    params_lm.setMaxIterations(max_iterations)
    params_lm.setRelativeErrorTol(1e-6)
    if verbose:
        params_lm.setVerbosity("ERROR")

    t2 = time.perf_counter()
    optimizer_lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, params_lm)
    result_lm = optimizer_lm.optimize()
    t3 = time.perf_counter()
    lm_time = (t3 - t2) * 1000
    lm_error = graph.error(result_lm)
    lm_iters = optimizer_lm.iterations()

    if verbose:
        print(f"LM: {lm_iters} iters, error {initial_error:.4e} -> {lm_error:.4e}, "
              f"time {lm_time:.1f} ms")

    return {
        "nodes": initial.size(),
        "factors": graph.size(),
        "initial_error": initial_error,
        "gn_error": gn_error,
        "gn_iters": gn_iters,
        "gn_time_ms": gn_time,
        "lm_error": lm_error,
        "lm_iters": lm_iters,
        "lm_time_ms": lm_time,
    }


def main():
    parser = argparse.ArgumentParser(description="GTSAM benchmark on g2o files")
    parser.add_argument("files", nargs="+", help="g2o file paths")
    parser.add_argument("--max-iter", type=int, default=30)
    args = parser.parse_args()

    results = []
    for filepath in args.files:
        print(f"\n{'='*60}")
        print(f"Dataset: {filepath}")
        print(f"{'='*60}")
        r = load_g2o_and_solve(filepath, max_iterations=args.max_iter)
        results.append((filepath, r))

    # Summary table
    print(f"\n{'='*80}")
    print("GTSAM BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<30} {'Nodes':>6} {'GN ms':>8} {'GN err':>12} {'LM ms':>8} {'LM err':>12}")
    print("-" * 80)
    for filepath, r in results:
        name = filepath.split("/")[-1]
        print(f"{name:<30} {r['nodes']:>6} {r['gn_time_ms']:>8.1f} "
              f"{r['gn_error']:>12.4e} {r['lm_time_ms']:>8.1f} {r['lm_error']:>12.4e}")


if __name__ == "__main__":
    main()
