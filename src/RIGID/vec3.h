/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing authors: Shern Tee (GU), Stephen Sanderson (UQ)
   ------------------------------------------------------------------------- */

#ifndef LMP_VEC3_H
#define LMP_VEC3_H

namespace RigsMath {

struct Vec3 {
  double x, y, z;

  Vec3() = default;
  Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
  Vec3(const double *p) : x(p[0]), y(p[1]), z(p[2]) {}

  Vec3 operator+(const Vec3 &b) const {
    return {x + b.x, y + b.y, z + b.z};
  }
  Vec3 &operator+=(const Vec3 &b) {
    x += b.x; y += b.y; z += b.z;
    return *this;
  }
  Vec3 operator-(const Vec3 &b) const {
    return {x - b.x, y - b.y, z - b.z};
  }
  Vec3 &operator-=(const Vec3 &b) {
    x -= b.x; y -= b.y; z -= b.z;
    return *this;
  }
  Vec3 operator*(double s) const {
    return {x * s, y * s, z * s};
  }
  Vec3 &operator*=(double s) {
    x *= s; y *= s; z *= s;
    return *this;
  }
  Vec3 operator/(double s) const {
    return {x / s, y / s, z / s};
  }
  Vec3 &operator/=(double s) {
    x /= s; y /= s; z /= s;
    return *this;
  }
  double &operator[](int i) { return (&x)[i]; }
  double operator[](int i) const { return (&x)[i]; }
  double *data() { return &x; }
  const double *data() const { return &x; }
};

inline double dot(const Vec3 &a, const Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3 &a, const Vec3 &b) {
  return {a.y * b.z - a.z * b.y,
          a.z * b.x - a.x * b.z,
          a.x * b.y - a.y * b.x};
}

inline double normsq(const Vec3 &v) {
  return v.x * v.x + v.y * v.y + v.z * v.z;
}

inline Vec3 operator*(double s, const Vec3 &v) {
  return v * s;
}

}

#endif