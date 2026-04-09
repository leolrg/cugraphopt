#include "cugraphopt/se3.hpp"

#include <algorithm>
#include <cmath>

namespace cugraphopt {

namespace {
constexpr double kEps = 1e-10;
}  // namespace

const Mat3 kIdentity3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
const Mat6 kIdentity6 = {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                         0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                         0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1};

// --- Vec3 ----------------------------------------------------------------

double vec3_dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec3 vec3_add(const Vec3& a, const Vec3& b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

Vec3 vec3_sub(const Vec3& a, const Vec3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 vec3_scale(double s, const Vec3& a) {
  return {s * a[0], s * a[1], s * a[2]};
}

double vec3_norm(const Vec3& v) {
  return std::sqrt(vec3_dot(v, v));
}

// --- Mat3 ----------------------------------------------------------------

Mat3 mat3_multiply(const Mat3& A, const Mat3& B) {
  Mat3 C{};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 3; ++k) {
        C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
      }
    }
  }
  return C;
}

Mat3 mat3_transpose(const Mat3& A) {
  return {A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]};
}

Vec3 mat3_vec(const Mat3& A, const Vec3& v) {
  return {A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
          A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
          A[6] * v[0] + A[7] * v[1] + A[8] * v[2]};
}

Mat3 mat3_add(const Mat3& A, const Mat3& B) {
  Mat3 C;
  for (int i = 0; i < 9; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

Mat3 mat3_scale(double s, const Mat3& A) {
  Mat3 B;
  for (int i = 0; i < 9; ++i) {
    B[i] = s * A[i];
  }
  return B;
}

// --- Mat6 ----------------------------------------------------------------

Mat6 mat6_multiply(const Mat6& A, const Mat6& B) {
  Mat6 C{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      for (int k = 0; k < 6; ++k) {
        C[i * 6 + j] += A[i * 6 + k] * B[k * 6 + j];
      }
    }
  }
  return C;
}

Mat6 mat6_transpose(const Mat6& A) {
  Mat6 B;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      B[i * 6 + j] = A[j * 6 + i];
    }
  }
  return B;
}

Vec6 mat6_vec(const Mat6& A, const Vec6& v) {
  Vec6 r{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      r[i] += A[i * 6 + j] * v[j];
    }
  }
  return r;
}

Mat6 mat6_add(const Mat6& A, const Mat6& B) {
  Mat6 C;
  for (int i = 0; i < 36; ++i) {
    C[i] = A[i] + B[i];
  }
  return C;
}

// --- hat -----------------------------------------------------------------

Mat3 hat3(const Vec3& v) {
  return {0,     -v[2],  v[1],
          v[2],   0,    -v[0],
         -v[1],  v[0],   0};
}

// --- SO(3) ---------------------------------------------------------------

SO3 exp(const so3& phi) {
  const double theta = vec3_norm(phi.v);
  if (theta < kEps) {
    return SO3{mat3_add(kIdentity3, hat3(phi.v))};
  }
  const Mat3 K = hat3(vec3_scale(1.0 / theta, phi.v));
  const Mat3 K2 = mat3_multiply(K, K);
  return SO3{mat3_add(mat3_add(kIdentity3, mat3_scale(std::sin(theta), K)),
                      mat3_scale(1.0 - std::cos(theta), K2))};
}

so3 log(const SO3& R) {
  const double cos_theta =
      std::clamp((R.m[0] + R.m[4] + R.m[8] - 1.0) / 2.0, -1.0, 1.0);
  const double theta = std::acos(cos_theta);

  if (theta < kEps) {
    return so3{{(R.m[7] - R.m[5]) / 2.0, (R.m[2] - R.m[6]) / 2.0,
                (R.m[3] - R.m[1]) / 2.0}};
  }

  if (M_PI - theta < kEps) {
    int k = 0;
    if (R.m[4] > R.m[0]) k = 1;
    if (R.m[8] > R.m[k * 3 + k]) k = 2;
    Vec3 n{};
    n[k] = std::sqrt((R.m[k * 3 + k] + 1.0) / 2.0);
    for (int j = 0; j < 3; ++j) {
      if (j != k) n[j] = R.m[j * 3 + k] / (2.0 * n[k]);
    }
    return so3{vec3_scale(theta, n)};
  }

  const double coeff = theta / (2.0 * std::sin(theta));
  return so3{{coeff * (R.m[7] - R.m[5]), coeff * (R.m[2] - R.m[6]),
              coeff * (R.m[3] - R.m[1])}};
}

Mat3 left_jacobian(const so3& phi) {
  const double theta = vec3_norm(phi.v);
  const Mat3 S = hat3(phi.v);
  const Mat3 S2 = mat3_multiply(S, S);

  if (theta < kEps) {
    return mat3_add(mat3_add(kIdentity3, mat3_scale(0.5, S)),
                    mat3_scale(1.0 / 6.0, S2));
  }

  const double t2 = theta * theta;
  const double a = (1.0 - std::cos(theta)) / t2;
  const double b = (theta - std::sin(theta)) / (t2 * theta);
  return mat3_add(mat3_add(kIdentity3, mat3_scale(a, S)),
                  mat3_scale(b, S2));
}

Mat3 left_jacobian_inv(const so3& phi) {
  const double theta = vec3_norm(phi.v);
  const Mat3 S = hat3(phi.v);
  const Mat3 S2 = mat3_multiply(S, S);

  if (theta < kEps) {
    return mat3_add(mat3_add(kIdentity3, mat3_scale(-0.5, S)),
                    mat3_scale(1.0 / 12.0, S2));
  }

  const double t2 = theta * theta;
  const double c = 1.0 / t2 -
                   (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
  return mat3_add(mat3_add(kIdentity3, mat3_scale(-0.5, S)),
                  mat3_scale(c, S2));
}

// --- SE(3) ---------------------------------------------------------------

SE3 compose(const SE3& a, const SE3& b) {
  return {SO3{mat3_multiply(a.R.m, b.R.m)},
          vec3_add(mat3_vec(a.R.m, b.t), a.t)};
}

SE3 inverse(const SE3& a) {
  const Mat3 Rt = mat3_transpose(a.R.m);
  return {SO3{Rt}, vec3_scale(-1.0, mat3_vec(Rt, a.t))};
}

SE3 exp(const se3& xi) {
  const so3 phi{{xi[3], xi[4], xi[5]}};
  const Vec3 rho = {xi[0], xi[1], xi[2]};
  const SO3 R = exp(phi);
  const Mat3 V = left_jacobian(phi);
  return {R, mat3_vec(V, rho)};
}

se3 log(const SE3& T) {
  const so3 phi = log(T.R);
  const Mat3 V_inv = left_jacobian_inv(phi);
  const Vec3 rho = mat3_vec(V_inv, T.t);
  return se3{{rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]}};
}

// --- Quaternion ----------------------------------------------------------

SO3 quat_to_SO3(double qx, double qy, double qz, double qw) {
  const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const double wx = qw * qx, wy = qw * qy, wz = qw * qz;
  return SO3{{1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy),
              2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx),
              2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)}};
}

}  // namespace cugraphopt
