#include "cugraphopt/se3.hpp"

#include <cassert>
#include <cmath>

namespace {

constexpr double kTol = 1e-9;

bool near(double a, double b, double tol = kTol) {
  return std::abs(a - b) < tol;
}

bool mat3_near(const cugraphopt::Mat3& a, const cugraphopt::Mat3& b,
               double tol = kTol) {
  for (int i = 0; i < 9; ++i) {
    if (!near(a[i], b[i], tol)) return false;
  }
  return true;
}

bool vec3_near(const cugraphopt::Vec3& a, const cugraphopt::Vec3& b,
               double tol = kTol) {
  return near(a[0], b[0], tol) && near(a[1], b[1], tol) &&
         near(a[2], b[2], tol);
}

bool vec6_near(const cugraphopt::Vec6& a, const cugraphopt::Vec6& b,
               double tol = kTol) {
  for (int i = 0; i < 6; ++i) {
    if (!near(a[i], b[i], tol)) return false;
  }
  return true;
}

void test_hat3() {
  using namespace cugraphopt;
  const Vec3 v = {1, 2, 3};
  const Mat3 S = hat3(v);
  assert(near(S[0], 0));
  assert(near(S[1], -3));
  assert(near(S[2], 2));
  assert(near(S[3], 3));
  assert(near(S[4], 0));
  assert(near(S[5], -1));
  assert(near(S[6], -2));
  assert(near(S[7], 1));
  assert(near(S[8], 0));
}

void test_quat_to_SO3_identity() {
  using namespace cugraphopt;
  const SO3 R = quat_to_SO3(0, 0, 0, 1);
  assert(mat3_near(R.m, kIdentity3));
}

void test_quat_to_SO3_90_z() {
  using namespace cugraphopt;
  const double s = std::sin(M_PI / 4);
  const double c = std::cos(M_PI / 4);
  const SO3 R = quat_to_SO3(0, 0, s, c);
  const Mat3 expected = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  assert(mat3_near(R.m, expected, 1e-9));
}

void test_so3_exp_identity() {
  using namespace cugraphopt;
  const so3 zero{};
  const SO3 R = exp(zero);
  assert(mat3_near(R.m, kIdentity3));
}

void test_so3_exp_90_z() {
  using namespace cugraphopt;
  const so3 phi{{0, 0, M_PI / 2}};
  const SO3 R = exp(phi);
  const Mat3 expected = {0, -1, 0, 1, 0, 0, 0, 0, 1};
  assert(mat3_near(R.m, expected, 1e-9));
}

void test_so3_log_identity() {
  using namespace cugraphopt;
  const SO3 I;
  const so3 phi = log(I);
  assert(vec3_near(phi.v, {0, 0, 0}));
}

void test_so3_exp_log_roundtrip() {
  using namespace cugraphopt;
  const so3 phi{{0.1, -0.2, 0.3}};
  const SO3 R = exp(phi);
  const so3 phi_back = log(R);
  assert(vec3_near(phi.v, phi_back.v, 1e-9));
}

void test_mat3_multiply_identity() {
  using namespace cugraphopt;
  const Mat3 A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const Mat3 result = mat3_multiply(A, kIdentity3);
  assert(mat3_near(result, A));
}

void test_compose_identity() {
  using namespace cugraphopt;
  const SE3 T = {SO3{}, {1, 2, 3}};
  const SE3 I;
  const SE3 result = compose(T, I);
  assert(mat3_near(result.R.m, T.R.m));
  assert(vec3_near(result.t, T.t));
}

void test_inverse() {
  using namespace cugraphopt;
  const se3 xi{{0.1, 0.2, 0.3, 0.04, 0.05, 0.06}};
  const SE3 T = exp(xi);
  const SE3 T_inv = inverse(T);
  const SE3 I_result = compose(T, T_inv);
  assert(mat3_near(I_result.R.m, kIdentity3, 1e-9));
  assert(vec3_near(I_result.t, {0, 0, 0}, 1e-9));
}

void test_se3_exp_identity() {
  using namespace cugraphopt;
  const se3 zero{};
  const SE3 T = exp(zero);
  assert(mat3_near(T.R.m, kIdentity3));
  assert(vec3_near(T.t, {0, 0, 0}));
}

void test_se3_exp_pure_translation() {
  using namespace cugraphopt;
  const se3 xi{{1, 2, 3, 0, 0, 0}};
  const SE3 T = exp(xi);
  assert(mat3_near(T.R.m, kIdentity3, 1e-9));
  assert(vec3_near(T.t, {1, 2, 3}, 1e-9));
}

void test_se3_exp_log_roundtrip() {
  using namespace cugraphopt;
  const se3 xi{{0.1, -0.2, 0.3, 0.04, -0.05, 0.06}};
  const SE3 T = exp(xi);
  const se3 xi_back = log(T);
  assert(vec6_near(xi.v, xi_back.v, 1e-9));
}

}  // namespace

int main() {
  test_hat3();
  test_quat_to_SO3_identity();
  test_quat_to_SO3_90_z();
  test_so3_exp_identity();
  test_so3_exp_90_z();
  test_so3_log_identity();
  test_so3_exp_log_roundtrip();
  test_mat3_multiply_identity();
  test_compose_identity();
  test_inverse();
  test_se3_exp_identity();
  test_se3_exp_pure_translation();
  test_se3_exp_log_roundtrip();
  return 0;
}
