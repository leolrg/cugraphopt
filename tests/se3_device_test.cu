#include "cugraphopt/se3_device.cuh"
#include "cugraphopt/se3.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>

using namespace cugraphopt;
using namespace cugraphopt::device;

// ---- GPU test kernel that writes results to device memory -----------------

struct DeviceTestResults {
  // exp/log roundtrip
  double exp_log_err;
  // compose test
  double compose_pos_err;
  double compose_rot_err;
  // inverse test
  double inv_pos_err;
  // residual test
  double residual_err;
  // jacobian test
  double jac_i_err;
  double jac_j_err;
};

__global__ void test_device_math_kernel(DeviceTestResults* results) {
  // Test 1: exp/log roundtrip
  {
    DVec6 xi = {{0.1, -0.2, 0.3, 0.4, -0.1, 0.2}};
    DSE3 T = dse3_exp(xi);
    DVec6 xi_back = dse3_log(T);
    double err = 0.0;
    for (int i = 0; i < 6; ++i) err += (xi[i] - xi_back[i]) * (xi[i] - xi_back[i]);
    results->exp_log_err = sqrt(err);
  }

  // Test 2: compose
  {
    DVec6 xi_a = {{1, 0, 0, 0, 0, 0.1}};
    DVec6 xi_b = {{0, 1, 0, 0.1, 0, 0}};
    DSE3 Ta = dse3_exp(xi_a);
    DSE3 Tb = dse3_exp(xi_b);
    DSE3 Tc = dse3_compose(Ta, Tb);

    // Check position is roughly Ta.t + Ta.R * Tb.t
    DVec3 expected_t = dvec3_add(Ta.t, dmat3_vec(Ta.R.R, Tb.t));
    double pos_err = 0.0;
    for (int i = 0; i < 3; ++i) pos_err += (Tc.t[i] - expected_t[i]) * (Tc.t[i] - expected_t[i]);
    results->compose_pos_err = sqrt(pos_err);

    // Check rotation: Tc.R should be Ta.R * Tb.R
    DMat3 expected_R = dmat3_multiply(Ta.R.R, Tb.R.R);
    double rot_err = 0.0;
    for (int i = 0; i < 9; ++i) rot_err += (Tc.R.R.m[i] - expected_R.m[i]) * (Tc.R.R.m[i] - expected_R.m[i]);
    results->compose_rot_err = sqrt(rot_err);
  }

  // Test 3: inverse
  {
    DVec6 xi = {{0.5, -0.3, 0.7, 0.2, -0.4, 0.1}};
    DSE3 T = dse3_exp(xi);
    DSE3 T_inv = dse3_inverse(T);
    DSE3 I = dse3_compose(T, T_inv);

    // Should be identity: t = 0
    double pos_err = 0.0;
    for (int i = 0; i < 3; ++i) pos_err += I.t[i] * I.t[i];
    results->inv_pos_err = sqrt(pos_err);
  }

  // Test 4: residual for consistent poses (should be ~0)
  {
    DSO3 R_id = {dmat3_identity()};
    DSE3 T_i = {R_id, {{0, 0, 0}}};
    DSE3 T_j = {R_id, {{1, 0, 0}}};
    DSE3 Z_ij = {R_id, {{1, 0, 0}}};

    DVec6 r = dcompute_residual(T_i, T_j, Z_ij);
    double err = 0.0;
    for (int i = 0; i < 6; ++i) err += r[i] * r[i];
    results->residual_err = sqrt(err);
  }

  // Test 5: Jacobians
  {
    DSO3 R_id = {dmat3_identity()};
    DSE3 T_i = {R_id, {{0, 0, 0}}};
    DSE3 T_j = {R_id, {{1, 0.5, 0}}};
    DSE3 Z_ij = {R_id, {{1, 0, 0}}};

    DMat6 J_i, J_j;
    dcompute_jacobians(T_i, T_j, Z_ij, J_i, J_j);

    // J_j should be close to identity for small perturbation from consistent
    // Just check it's non-zero and reasonable
    double ji_norm = 0.0, jj_norm = 0.0;
    for (int a = 0; a < 36; ++a) {
      ji_norm += J_i.m[a] * J_i.m[a];
      jj_norm += J_j.m[a] * J_j.m[a];
    }
    results->jac_i_err = (ji_norm > 0.01) ? 0.0 : 1.0;  // 0 = ok
    results->jac_j_err = (jj_norm > 0.01) ? 0.0 : 1.0;
  }
}

// ---- CPU-side comparison --------------------------------------------------

static void test_cpu_gpu_agreement() {
  // Run the same computations on CPU using the device functions (they're
  // __host__ __device__) and compare with host-only functions.

  // exp test
  {
    se3 xi_host{};
    xi_host[0] = 0.1; xi_host[1] = -0.2; xi_host[2] = 0.3;
    xi_host[3] = 0.4; xi_host[4] = -0.1; xi_host[5] = 0.2;
    SE3 T_host = exp(xi_host);

    DVec6 xi_dev = {{0.1, -0.2, 0.3, 0.4, -0.1, 0.2}};
    DSE3 T_dev = dse3_exp(xi_dev);

    // Compare translation
    double err = 0.0;
    err += (T_host.t[0] - T_dev.t[0]) * (T_host.t[0] - T_dev.t[0]);
    err += (T_host.t[1] - T_dev.t[1]) * (T_host.t[1] - T_dev.t[1]);
    err += (T_host.t[2] - T_dev.t[2]) * (T_host.t[2] - T_dev.t[2]);
    assert(std::sqrt(err) < 1e-12);

    // Compare rotation
    double rot_err = 0.0;
    for (int i = 0; i < 9; ++i) {
      double diff = T_host.R.m[i] - T_dev.R.R.m[i];
      rot_err += diff * diff;
    }
    assert(std::sqrt(rot_err) < 1e-12);
  }

  // Log test
  {
    se3 xi_host{};
    xi_host[0] = 0.3; xi_host[1] = -0.1; xi_host[2] = 0.5;
    xi_host[3] = 0.2; xi_host[4] = -0.3; xi_host[5] = 0.1;
    SE3 T_host = exp(xi_host);
    se3 xi_back_host = log(T_host);

    DVec6 xi_dev = {{0.3, -0.1, 0.5, 0.2, -0.3, 0.1}};
    DSE3 T_dev = dse3_exp(xi_dev);
    DVec6 xi_back_dev = dse3_log(T_dev);

    double err = 0.0;
    for (int i = 0; i < 6; ++i) {
      double diff = xi_back_host[i] - xi_back_dev[i];
      err += diff * diff;
    }
    assert(std::sqrt(err) < 1e-10);
  }

  std::printf("PASS: test_cpu_gpu_agreement\n");
}

int main() {
  test_cpu_gpu_agreement();

  // Run GPU kernel
  DeviceTestResults* d_results;
  cudaMalloc(&d_results, sizeof(DeviceTestResults));

  test_device_math_kernel<<<1, 1>>>(d_results);
  cudaDeviceSynchronize();

  DeviceTestResults results;
  cudaMemcpy(&results, d_results, sizeof(DeviceTestResults), cudaMemcpyDeviceToHost);
  cudaFree(d_results);

  std::printf("GPU exp/log roundtrip error: %.2e\n", results.exp_log_err);
  std::printf("GPU compose position error:  %.2e\n", results.compose_pos_err);
  std::printf("GPU compose rotation error:  %.2e\n", results.compose_rot_err);
  std::printf("GPU inverse error:           %.2e\n", results.inv_pos_err);
  std::printf("GPU residual error:          %.2e\n", results.residual_err);

  assert(results.exp_log_err < 1e-10);
  assert(results.compose_pos_err < 1e-12);
  assert(results.compose_rot_err < 1e-12);
  assert(results.inv_pos_err < 1e-12);
  assert(results.residual_err < 1e-12);
  assert(results.jac_i_err < 0.5);
  assert(results.jac_j_err < 0.5);

  std::printf("PASS: GPU device math tests\n");
  std::printf("\nAll SE(3) device tests passed.\n");
  return 0;
}
