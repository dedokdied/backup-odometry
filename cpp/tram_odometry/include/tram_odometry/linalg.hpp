// Dependency-free fixed-size linear algebra for the error-state Kalman filter.
// Everything is stack allocated: no heap traffic inside the control loop.
#pragma once

#include <cmath>
#include <cstddef>

namespace tram {

template <int N>
struct Vec {
  double v[N];

  double& operator[](int i) { return v[i]; }
  double operator[](int i) const { return v[i]; }
  void setZero() {
    for (int i = 0; i < N; ++i) v[i] = 0.0;
  }
};

template <int N>
inline Vec<N> operator+(const Vec<N>& a, const Vec<N>& b) {
  Vec<N> r;
  for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
  return r;
}

template <int N>
inline Vec<N> operator-(const Vec<N>& a, const Vec<N>& b) {
  Vec<N> r;
  for (int i = 0; i < N; ++i) r[i] = a[i] - b[i];
  return r;
}

template <int N>
inline Vec<N> operator*(const Vec<N>& a, double s) {
  Vec<N> r;
  for (int i = 0; i < N; ++i) r[i] = a[i] * s;
  return r;
}

template <int N>
inline double dot(const Vec<N>& a, const Vec<N>& b) {
  double s = 0.0;
  for (int i = 0; i < N; ++i) s += a[i] * b[i];
  return s;
}

template <int N>
inline double norm(const Vec<N>& a) {
  return std::sqrt(dot(a, a));
}

template <int N>
struct Mat {
  double a[N][N];

  void setZero() {
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) a[i][j] = 0.0;
  }
  void setIdentity() {
    setZero();
    for (int i = 0; i < N; ++i) a[i][i] = 1.0;
  }
  double& operator()(int i, int j) { return a[i][j]; }
  double operator()(int i, int j) const { return a[i][j]; }
};

template <int N>
inline Mat<N> operator*(const Mat<N>& m, const Mat<N>& n) {
  Mat<N> r;
  r.setZero();
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) {
      const double mik = m(i, k);
      if (mik == 0.0) continue;
      for (int j = 0; j < N; ++j) r(i, j) += mik * n(k, j);
    }
  }
  return r;
}

template <int N>
inline Mat<N> operator+(const Mat<N>& m, const Mat<N>& n) {
  Mat<N> r;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) r(i, j) = m(i, j) + n(i, j);
  return r;
}

template <int N>
inline Mat<N> operator-(const Mat<N>& m, const Mat<N>& n) {
  Mat<N> r;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) r(i, j) = m(i, j) - n(i, j);
  return r;
}

template <int N>
inline Mat<N> operator*(const Mat<N>& m, double s) {
  Mat<N> r;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) r(i, j) = m(i, j) * s;
  return r;
}

template <int N>
inline Mat<N> transpose(const Mat<N>& m) {
  Mat<N> r;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) r(i, j) = m(j, i);
  return r;
}

template <int N>
inline Vec<N> operator*(const Mat<N>& m, const Vec<N>& x) {
  Vec<N> r;
  for (int i = 0; i < N; ++i) {
    double s = 0.0;
    for (int j = 0; j < N; ++j) s += m(i, j) * x[j];
    r[i] = s;
  }
  return r;
}

/// Gauss-Jordan inverse with partial pivoting. Returns false if singular.
template <int N>
inline bool invert(const Mat<N>& in, Mat<N>& out) {
  double work[N][2 * N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) work[i][j] = in(i, j);
    for (int j = 0; j < N; ++j) work[i][N + j] = (i == j) ? 1.0 : 0.0;
  }
  for (int col = 0; col < N; ++col) {
    int piv = col;
    double best = std::fabs(work[col][col]);
    for (int r = col + 1; r < N; ++r) {
      const double v = std::fabs(work[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-14) return false;
    if (piv != col) {
      for (int j = 0; j < 2 * N; ++j) {
        const double tmp = work[col][j];
        work[col][j] = work[piv][j];
        work[piv][j] = tmp;
      }
    }
    const double d = work[col][col];
    for (int j = 0; j < 2 * N; ++j) work[col][j] /= d;
    for (int r = 0; r < N; ++r) {
      if (r == col) continue;
      const double f = work[r][col];
      if (f == 0.0) continue;
      for (int j = 0; j < 2 * N; ++j) work[r][j] -= f * work[col][j];
    }
  }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) out(i, j) = work[i][N + j];
  return true;
}

/// Symmetrise, guarding against numerical drift in long runs.
template <int N>
inline void symmetrize(Mat<N>& m) {
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      const double avg = 0.5 * (m(i, j) + m(j, i));
      m(i, j) = avg;
      m(j, i) = avg;
    }
  }
  for (int i = 0; i < N; ++i) {
    if (m(i, i) < 1e-9) m(i, i) = 1e-9;
  }
}

}  // namespace tram
