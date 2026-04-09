#pragma once

#include <array>

namespace cugraphopt {

// --- Raw fixed-size aliases (for internal math, Jacobians, etc.) ----------
using Vec3 = std::array<double, 3>;
using Vec6 = std::array<double, 6>;
using Mat3 = std::array<double, 9>;   // row-major
using Mat6 = std::array<double, 36>;  // row-major

extern const Mat3 kIdentity3;
extern const Mat6 kIdentity6;

// --- Lie algebra elements ------------------------------------------------

struct so3 {
  Vec3 v{};
  double& operator[](int i) { return v[i]; }
  const double& operator[](int i) const { return v[i]; }
};

struct se3 {
  Vec6 v{};
  double& operator[](int i) { return v[i]; }
  const double& operator[](int i) const { return v[i]; }
};

// --- Lie group elements --------------------------------------------------

struct SO3 {
  Mat3 m = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  double& operator()(int r, int c) { return m[r * 3 + c]; }
  const double& operator()(int r, int c) const { return m[r * 3 + c]; }
};

struct SE3 {
  SO3 R;
  Vec3 t{};
};

// --- Vec3 helpers --------------------------------------------------------
double vec3_dot(const Vec3& a, const Vec3& b);
Vec3 vec3_add(const Vec3& a, const Vec3& b);
Vec3 vec3_sub(const Vec3& a, const Vec3& b);
Vec3 vec3_scale(double s, const Vec3& a);
double vec3_norm(const Vec3& v);

// --- Mat3 helpers (row-major) --------------------------------------------
Mat3 mat3_multiply(const Mat3& A, const Mat3& B);
Mat3 mat3_transpose(const Mat3& A);
Vec3 mat3_vec(const Mat3& A, const Vec3& v);
Mat3 mat3_add(const Mat3& A, const Mat3& B);
Mat3 mat3_scale(double s, const Mat3& A);

// --- Mat6 helpers (row-major) --------------------------------------------
Mat6 mat6_multiply(const Mat6& A, const Mat6& B);
Mat6 mat6_transpose(const Mat6& A);
Vec6 mat6_vec(const Mat6& A, const Vec6& v);
Mat6 mat6_add(const Mat6& A, const Mat6& B);

// --- Skew-symmetric (works on any Vec3, not just so3) --------------------
Mat3 hat3(const Vec3& v);

// --- SO(3) exponential / logarithm maps ----------------------------------
SO3 exp(const so3& phi);            // so(3) -> SO(3)
so3 log(const SO3& R);              // SO(3) -> so(3)
Mat3 left_jacobian(const so3& phi);
Mat3 left_jacobian_inv(const so3& phi);

// --- SE(3) group operations and maps -------------------------------------
SE3 compose(const SE3& a, const SE3& b);
SE3 inverse(const SE3& a);
SE3 exp(const se3& xi);             // se(3) -> SE(3)
se3 log(const SE3& T);              // SE(3) -> se(3)

// --- Quaternion conversion -----------------------------------------------
SO3 quat_to_SO3(double qx, double qy, double qz, double qw);

}  // namespace cugraphopt
