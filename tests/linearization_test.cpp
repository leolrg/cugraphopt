#include "cugraphopt/linearization.hpp"

#include <cassert>
#include <cmath>

namespace {

constexpr double kTol = 1e-6;

bool near(double a, double b, double tol = kTol) {
  return std::abs(a - b) < tol;
}

void test_expand_information_diagonal() {
  using namespace cugraphopt;
  std::array<double, 21> upper{};
  upper[0] = 1;
  upper[6] = 2;
  upper[11] = 3;
  upper[15] = 4;
  upper[18] = 5;
  upper[20] = 6;

  Mat6 omega{};
  expand_information(upper, omega);

  assert(near(omega[0 * 6 + 0], 1));
  assert(near(omega[1 * 6 + 1], 2));
  assert(near(omega[2 * 6 + 2], 3));
  assert(near(omega[3 * 6 + 3], 4));
  assert(near(omega[4 * 6 + 4], 5));
  assert(near(omega[5 * 6 + 5], 6));

  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      assert(near(omega[i * 6 + j], omega[j * 6 + i]));
    }
  }
}

void test_expand_information_full() {
  using namespace cugraphopt;
  std::array<double, 21> upper;
  for (int i = 0; i < 21; ++i) {
    upper[i] = i + 1.0;
  }

  Mat6 omega{};
  expand_information(upper, omega);

  assert(near(omega[0 * 6 + 0], 1));
  assert(near(omega[0 * 6 + 1], 2));
  assert(near(omega[0 * 6 + 5], 6));
  assert(near(omega[1 * 6 + 0], 2));
  assert(near(omega[1 * 6 + 1], 7));
  assert(near(omega[5 * 6 + 5], 21));
}

void test_residual_identity_consistent() {
  using namespace cugraphopt;
  const SE3 T_i;
  const SE3 T_j = {SO3{}, {1, 0, 0}};
  const SE3 Z_ij = T_j;

  const se3 r = compute_residual(T_i, T_j, Z_ij);
  for (int i = 0; i < 6; ++i) {
    assert(near(r[i], 0.0, 1e-10));
  }
}

void test_residual_with_error() {
  using namespace cugraphopt;
  const SE3 T_i;
  const SE3 T_j = {SO3{}, {2, 0, 0}};
  const SE3 Z_ij = {SO3{}, {1, 0, 0}};

  const se3 r = compute_residual(T_i, T_j, Z_ij);

  for (int i = 0; i < 6; ++i) {
    assert(std::isfinite(r[i]));
  }
  assert(near(r[0], 1.0, 1e-9));
  assert(near(r[1], 0.0, 1e-9));
  assert(near(r[2], 0.0, 1e-9));
}

}  // namespace

int main() {
  test_expand_information_diagonal();
  test_expand_information_full();
  test_residual_identity_consistent();
  test_residual_with_error();
  return 0;
}
