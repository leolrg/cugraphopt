#pragma once

#include <cmath>

/// Device-compatible SE(3) Lie group math.
/// All functions are __host__ __device__ so they can run on both CPU and GPU.
/// Uses plain double arrays instead of std::array for CUDA compatibility.

namespace cugraphopt {
namespace device {

// ---- Device-compatible vector/matrix types --------------------------------

struct DVec3 {
  double v[3];
  __host__ __device__ double& operator[](int i) { return v[i]; }
  __host__ __device__ const double& operator[](int i) const { return v[i]; }
};

struct DVec6 {
  double v[6];
  __host__ __device__ double& operator[](int i) { return v[i]; }
  __host__ __device__ const double& operator[](int i) const { return v[i]; }
};

struct DMat3 {
  double m[9];  // row-major
  __host__ __device__ double& operator()(int r, int c) { return m[r * 3 + c]; }
  __host__ __device__ const double& operator()(int r, int c) const { return m[r * 3 + c]; }
};

struct DMat6 {
  double m[36];  // row-major
  __host__ __device__ double& operator()(int r, int c) { return m[r * 6 + c]; }
  __host__ __device__ const double& operator()(int r, int c) const { return m[r * 6 + c]; }
};

struct DSO3 {
  DMat3 R;
};

struct DSE3 {
  DSO3 R;
  DVec3 t;
};

// ---- Vec3 operations ------------------------------------------------------

__host__ __device__ inline double dvec3_dot(const DVec3& a, const DVec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__host__ __device__ inline DVec3 dvec3_add(const DVec3& a, const DVec3& b) {
  return {{a[0] + b[0], a[1] + b[1], a[2] + b[2]}};
}

__host__ __device__ inline DVec3 dvec3_sub(const DVec3& a, const DVec3& b) {
  return {{a[0] - b[0], a[1] - b[1], a[2] - b[2]}};
}

__host__ __device__ inline DVec3 dvec3_scale(double s, const DVec3& a) {
  return {{s * a[0], s * a[1], s * a[2]}};
}

__host__ __device__ inline double dvec3_norm(const DVec3& v) {
  return sqrt(dvec3_dot(v, v));
}

// ---- Mat3 operations ------------------------------------------------------

__host__ __device__ inline DMat3 dmat3_multiply(const DMat3& A, const DMat3& B) {
  DMat3 C;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        s += A.m[i * 3 + k] * B.m[k * 3 + j];
      }
      C.m[i * 3 + j] = s;
    }
  }
  return C;
}

__host__ __device__ inline DMat3 dmat3_transpose(const DMat3& A) {
  return {{A.m[0], A.m[3], A.m[6], A.m[1], A.m[4], A.m[7], A.m[2], A.m[5], A.m[8]}};
}

__host__ __device__ inline DVec3 dmat3_vec(const DMat3& A, const DVec3& v) {
  return {{A.m[0] * v[0] + A.m[1] * v[1] + A.m[2] * v[2],
           A.m[3] * v[0] + A.m[4] * v[1] + A.m[5] * v[2],
           A.m[6] * v[0] + A.m[7] * v[1] + A.m[8] * v[2]}};
}

__host__ __device__ inline DMat3 dmat3_add(const DMat3& A, const DMat3& B) {
  DMat3 C;
  for (int i = 0; i < 9; ++i) C.m[i] = A.m[i] + B.m[i];
  return C;
}

__host__ __device__ inline DMat3 dmat3_scale(double s, const DMat3& A) {
  DMat3 B;
  for (int i = 0; i < 9; ++i) B.m[i] = s * A.m[i];
  return B;
}

// ---- Mat6 operations ------------------------------------------------------

__host__ __device__ inline DMat6 dmat6_multiply(const DMat6& A, const DMat6& B) {
  DMat6 C;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 6; ++j) {
      double s = 0.0;
      for (int k = 0; k < 6; ++k) {
        s += A.m[i * 6 + k] * B.m[k * 6 + j];
      }
      C.m[i * 6 + j] = s;
    }
  }
  return C;
}

__host__ __device__ inline DMat6 dmat6_transpose(const DMat6& A) {
  DMat6 B;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      B.m[i * 6 + j] = A.m[j * 6 + i];
  return B;
}

__host__ __device__ inline DVec6 dmat6_vec(const DMat6& A, const DVec6& v) {
  DVec6 r;
  for (int i = 0; i < 6; ++i) {
    double s = 0.0;
    for (int j = 0; j < 6; ++j) s += A.m[i * 6 + j] * v[j];
    r[i] = s;
  }
  return r;
}

__host__ __device__ inline DMat6 dmat6_add(const DMat6& A, const DMat6& B) {
  DMat6 C;
  for (int i = 0; i < 36; ++i) C.m[i] = A.m[i] + B.m[i];
  return C;
}

// ---- Skew-symmetric -------------------------------------------------------

__host__ __device__ inline DMat3 dhat3(const DVec3& v) {
  return {{0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0}};
}

// ---- SO(3) exp/log --------------------------------------------------------

constexpr double kDeviceEps = 1e-10;

__host__ __device__ inline DMat3 dmat3_identity() {
  return {{1, 0, 0, 0, 1, 0, 0, 0, 1}};
}

__host__ __device__ inline DSO3 dso3_exp(const DVec3& phi) {
  const double theta = dvec3_norm(phi);
  DMat3 I = dmat3_identity();

  if (theta < kDeviceEps) {
    return DSO3{dmat3_add(I, dhat3(phi))};
  }

  DVec3 axis = dvec3_scale(1.0 / theta, phi);
  DMat3 K = dhat3(axis);
  DMat3 K2 = dmat3_multiply(K, K);

  return DSO3{dmat3_add(dmat3_add(I, dmat3_scale(sin(theta), K)),
                        dmat3_scale(1.0 - cos(theta), K2))};
}

__host__ __device__ inline DVec3 dso3_log(const DSO3& R) {
  double cos_theta = (R.R.m[0] + R.R.m[4] + R.R.m[8] - 1.0) / 2.0;
  if (cos_theta > 1.0) cos_theta = 1.0;
  if (cos_theta < -1.0) cos_theta = -1.0;
  double theta = acos(cos_theta);

  if (theta < kDeviceEps) {
    return {{(R.R.m[7] - R.R.m[5]) / 2.0,
             (R.R.m[2] - R.R.m[6]) / 2.0,
             (R.R.m[3] - R.R.m[1]) / 2.0}};
  }

  if (M_PI - theta < kDeviceEps) {
    int k = 0;
    if (R.R.m[4] > R.R.m[0]) k = 1;
    if (R.R.m[8] > R.R.m[k * 3 + k]) k = 2;
    DVec3 n = {{0, 0, 0}};
    n[k] = sqrt((R.R.m[k * 3 + k] + 1.0) / 2.0);
    for (int j = 0; j < 3; ++j) {
      if (j != k) n[j] = R.R.m[j * 3 + k] / (2.0 * n[k]);
    }
    return dvec3_scale(theta, n);
  }

  double coeff = theta / (2.0 * sin(theta));
  return {{coeff * (R.R.m[7] - R.R.m[5]),
           coeff * (R.R.m[2] - R.R.m[6]),
           coeff * (R.R.m[3] - R.R.m[1])}};
}

__host__ __device__ inline DMat3 dleft_jacobian(const DVec3& phi) {
  double theta = dvec3_norm(phi);
  DMat3 S = dhat3(phi);
  DMat3 S2 = dmat3_multiply(S, S);
  DMat3 I = dmat3_identity();

  if (theta < kDeviceEps) {
    return dmat3_add(dmat3_add(I, dmat3_scale(0.5, S)),
                     dmat3_scale(1.0 / 6.0, S2));
  }

  double t2 = theta * theta;
  double a = (1.0 - cos(theta)) / t2;
  double b = (theta - sin(theta)) / (t2 * theta);
  return dmat3_add(dmat3_add(I, dmat3_scale(a, S)), dmat3_scale(b, S2));
}

__host__ __device__ inline DMat3 dleft_jacobian_inv(const DVec3& phi) {
  double theta = dvec3_norm(phi);
  DMat3 S = dhat3(phi);
  DMat3 S2 = dmat3_multiply(S, S);
  DMat3 I = dmat3_identity();

  if (theta < kDeviceEps) {
    return dmat3_add(dmat3_add(I, dmat3_scale(-0.5, S)),
                     dmat3_scale(1.0 / 12.0, S2));
  }

  double t2 = theta * theta;
  double c = 1.0 / t2 - (1.0 + cos(theta)) / (2.0 * theta * sin(theta));
  return dmat3_add(dmat3_add(I, dmat3_scale(-0.5, S)), dmat3_scale(c, S2));
}

// ---- SE(3) operations -----------------------------------------------------

__host__ __device__ inline DSE3 dse3_compose(const DSE3& a, const DSE3& b) {
  return {DSO3{dmat3_multiply(a.R.R, b.R.R)},
          dvec3_add(dmat3_vec(a.R.R, b.t), a.t)};
}

__host__ __device__ inline DSE3 dse3_inverse(const DSE3& a) {
  DMat3 Rt = dmat3_transpose(a.R.R);
  return {DSO3{Rt}, dvec3_scale(-1.0, dmat3_vec(Rt, a.t))};
}

__host__ __device__ inline DSE3 dse3_exp(const DVec6& xi) {
  DVec3 phi = {{xi[3], xi[4], xi[5]}};
  DVec3 rho = {{xi[0], xi[1], xi[2]}};
  DSO3 R = dso3_exp(phi);
  DMat3 V = dleft_jacobian(phi);
  return {R, dmat3_vec(V, rho)};
}

__host__ __device__ inline DVec6 dse3_log(const DSE3& T) {
  DVec3 phi = dso3_log(T.R);
  DMat3 V_inv = dleft_jacobian_inv(phi);
  DVec3 rho = dmat3_vec(V_inv, T.t);
  return {{rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]}};
}

// ---- Quaternion conversion ------------------------------------------------

__host__ __device__ inline DSO3 dquat_to_SO3(double qx, double qy, double qz,
                                              double qw) {
  double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  double wx = qw * qx, wy = qw * qy, wz = qw * qz;
  return DSO3{{{1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
                2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
                2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)}}};
}

// ---- Residual + Jacobian (used by linearization kernels) ------------------

__host__ __device__ inline DVec6 dcompute_residual(const DSE3& T_i,
                                                    const DSE3& T_j,
                                                    const DSE3& Z_ij) {
  DSE3 T_ij_pred = dse3_compose(dse3_inverse(T_i), T_j);
  DSE3 E = dse3_compose(dse3_inverse(Z_ij), T_ij_pred);
  return dse3_log(E);
}

constexpr double kFiniteDiffEps = 1e-8;

__host__ __device__ inline void dcompute_jacobians(const DSE3& T_i,
                                                    const DSE3& T_j,
                                                    const DSE3& Z_ij,
                                                    DMat6& J_i, DMat6& J_j) {
  for (int a = 0; a < 36; ++a) { J_i.m[a] = 0.0; J_j.m[a] = 0.0; }

  for (int k = 0; k < 6; ++k) {
    DVec6 delta = {{0, 0, 0, 0, 0, 0}};

    // J_i column k
    delta[k] = kFiniteDiffEps;
    DSE3 T_i_plus = dse3_compose(T_i, dse3_exp(delta));
    delta[k] = -kFiniteDiffEps;
    DSE3 T_i_minus = dse3_compose(T_i, dse3_exp(delta));
    delta[k] = 0.0;

    DVec6 ri_plus = dcompute_residual(T_i_plus, T_j, Z_ij);
    DVec6 ri_minus = dcompute_residual(T_i_minus, T_j, Z_ij);

    for (int row = 0; row < 6; ++row) {
      J_i.m[row * 6 + k] = (ri_plus[row] - ri_minus[row]) / (2.0 * kFiniteDiffEps);
    }

    // J_j column k
    delta[k] = kFiniteDiffEps;
    DSE3 T_j_plus = dse3_compose(T_j, dse3_exp(delta));
    delta[k] = -kFiniteDiffEps;
    DSE3 T_j_minus = dse3_compose(T_j, dse3_exp(delta));

    DVec6 rj_plus = dcompute_residual(T_i, T_j_plus, Z_ij);
    DVec6 rj_minus = dcompute_residual(T_i, T_j_minus, Z_ij);

    for (int row = 0; row < 6; ++row) {
      J_j.m[row * 6 + k] = (rj_plus[row] - rj_minus[row]) / (2.0 * kFiniteDiffEps);
    }
  }
}

// ---- Information matrix expansion -----------------------------------------

__host__ __device__ inline void dexpand_information(const double upper[21],
                                                     DMat6& omega) {
  for (int a = 0; a < 36; ++a) omega.m[a] = 0.0;
  int k = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = i; j < 6; ++j) {
      omega.m[i * 6 + j] = upper[k];
      omega.m[j * 6 + i] = upper[k];
      ++k;
    }
  }
}

}  // namespace device
}  // namespace cugraphopt
