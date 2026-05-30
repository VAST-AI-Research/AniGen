#pragma once

#include <cuda.h>
#include <cuda_runtime.h>


namespace cumesh {


/**
 * A 3D vector class with overloaded operators and methods.
 */
struct __align__(16) Vec3f {
    float x, y, z;

    __device__ __forceinline__ Vec3f();
    __device__ __forceinline__ Vec3f(float x, float y, float z);
    __device__ __forceinline__ Vec3f(float3 v);
    __device__ __forceinline__ Vec3f operator+(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator+=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator-(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f& operator-=(const Vec3f& o);
    __device__ __forceinline__ Vec3f operator*(float s) const;
    __device__ __forceinline__ Vec3f& operator*=(float s);
    __device__ __forceinline__ Vec3f operator/(float s) const;
    __device__ __forceinline__ Vec3f& operator/=(float s);
    __device__ __forceinline__ float dot(const Vec3f& o) const;
    __device__ __forceinline__ float norm() const;
    __device__ __forceinline__ float norm2() const;
    __device__ __forceinline__ Vec3f normalized() const;
    __device__ __forceinline__ void normalize();
    __device__ __forceinline__ Vec3f cross(const Vec3f& o) const;
    __device__ __forceinline__ Vec3f slerp(const Vec3f& o, float t) const;
};


/**
 * QEM (Quadric Error Metric) class for mesh simplification.
 */
struct __align__(16) QEM
{
    // store upper triangle of symmetric 4x4 matrix:
    // e = [ 00, 01, 02, 03, 11, 12, 13, 22, 23, 33 ]
    float e[10];

    __device__ __forceinline__ QEM();
    __device__ __forceinline__ QEM operator+(const QEM& o) const;
    __device__ __forceinline__ QEM& operator+=(const QEM& o);
    __device__ __forceinline__ QEM operator-(const QEM& o) const;
    __device__ __forceinline__ QEM& operator-=(const QEM& o);
    __device__ __forceinline__ void zero();
    __device__ __forceinline__ void add_plane(float4 p);
    __device__ __forceinline__ float evaluate(const Vec3f& p) const;
    __device__ __forceinline__ bool solve_optimal(float3 &out, float &err) const;
};


__device__ __forceinline__ Vec3f::Vec3f() {
    x = 0.0f;
    y = 0.0f;
    z = 0.0f;
}

__device__ __forceinline__ Vec3f::Vec3f(float x, float y, float z) {
    this->x = x;
    this->y = y;
    this->z = z;
}

__device__ __forceinline__ Vec3f::Vec3f(float3 v) {
    x = v.x;
    y = v.y;
    z = v.z;
}


__device__ __forceinline__ Vec3f Vec3f::operator+(const Vec3f& o) const {
    return Vec3f(x + o.x, y + o.y, z + o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator+=(const Vec3f& o) {
    x += o.x;
    y += o.y;
    z += o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator-(const Vec3f& o) const {
    return Vec3f(x - o.x, y - o.y, z - o.z);
}


__device__ __forceinline__ Vec3f& Vec3f::operator-=(const Vec3f& o) {
    x -= o.x;
    y -= o.y;
    z -= o.z;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator*(float s) const {
    return Vec3f(x * s, y * s, z * s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator*=(float s) {
    x *= s;
    y *= s;
    z *= s;
    return *this;
}


__device__ __forceinline__ Vec3f Vec3f::operator/(float s) const {
    return Vec3f(x / s, y / s, z / s);
}


__device__ __forceinline__ Vec3f& Vec3f::operator/=(float s) {
    x /= s;
    y /= s;
    z /= s;
    return *this;
}


__device__ __forceinline__ float Vec3f::dot(const Vec3f& o) const {
    return x * o.x + y * o.y + z * o.z;
}


__device__ __forceinline__ float Vec3f::norm() const {
    return sqrtf(x * x + y * y + z * z);
}


__device__ __forceinline__ float Vec3f::norm2() const {
    return x * x + y * y + z * z;
}


__device__ __forceinline__ Vec3f Vec3f::normalized() const {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    return Vec3f(x * inv_norm, y * inv_norm, z * inv_norm);
}


__device__ __forceinline__ void Vec3f::normalize() {
    float inv_norm = rsqrtf(x * x + y * y + z * z);
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
}


__device__ __forceinline__ Vec3f Vec3f::cross(const Vec3f& o) const {
    return Vec3f(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
}


__device__ __forceinline__ Vec3f Vec3f::slerp(const Vec3f& o, float t) const {
    float dot_prod = this->dot(o);
    dot_prod = fmaxf(fminf(dot_prod, 1.0f), -1.0f); // Clamp to [-1, 1]
    float theta = acosf(dot_prod) * t;
    Vec3f relative_vec = (o - (*this) * dot_prod).normalized();
    return (*this) * cosf(theta) + relative_vec * sinf(theta);
}


__device__ __forceinline__ QEM::QEM() {
    zero();
}


__device__ __forceinline__ QEM QEM::operator+(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] + o.e[i];
    return res;
}


__device__ __forceinline__ QEM& QEM::operator+=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] += o.e[i];
    return *this;
}


__device__ __forceinline__ QEM QEM::operator-(const QEM& o) const {
    QEM res;
    #pragma unroll
    for (int i = 0; i < 10; ++i) res.e[i] = e[i] - o.e[i];
    return res;
}


__device__ __forceinline__ QEM& QEM::operator-=(const QEM& o) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] -= o.e[i];
    return *this;
}

__device__ __forceinline__ void QEM::zero() {
    #pragma unroll
    for (int i = 0; i < 10; ++i) e[i] = 0.0f;
}


// Add plane p = (a,b,c,d) as outer product p * p^T
__device__ __forceinline__ void QEM::add_plane(float4 p) {
    // upper triangle indices mapping:
    // (0,0)->e[0]
    // (0,1)->e[1]
    // (0,2)->e[2]
    // (0,3)->e[3]
    // (1,1)->e[4]
    // (1,2)->e[5]
    // (1,3)->e[6]
    // (2,2)->e[7]
    // (2,3)->e[8]
    // (3,3)->e[9]
    float a = p.x, b = p.y, c = p.z, d = p.w;
    e[0] += a * a;
    e[1] += a * b;
    e[2] += a * c;
    e[3] += a * d;
    e[4] += b * b;
    e[5] += b * c;
    e[6] += b * d;
    e[7] += c * c;
    e[8] += c * d;
    e[9] += d * d;
}


// Evaluate v^T * Q * v for v = (x,y,z,1)
__device__ __forceinline__ float QEM::evaluate(const Vec3f& p) const {
    // compute v = [x,y,z,1]
    float x = p.x, y = p.y, z = p.z, w = 1.0f;
    // expand symmetric multiplication using stored upper triangular
    // result = sum_{i<=j} M_ij * v_i * v_j * (1 if i==j else 2)
    float res = 0.0f;
    // (0,0)
    res += e[0] * x * x;
    // (0,1) and (1,0)
    res += 2.0f * e[1] * x * y;
    // (0,2)
    res += 2.0f * e[2] * x * z;
    // (0,3)
    res += 2.0f * e[3] * x * w;
    // (1,1)
    res += e[4] * y * y;
    // (1,2)
    res += 2.0f * e[5] * y * z;
    // (1,3)
    res += 2.0f * e[6] * y * w;
    // (2,2)
    res += e[7] * z * z;
    // (2,3)
    res += 2.0f * e[8] * z * w;
    // (3,3)
    res += e[9] * w * w;
    return res;
}


// Try to solve for optimal point minimizing v^T Q v with constraint v = (x,y,z,1)
// Solve the linear system: A * [x y z]^T = -b, where
// A = top-left 3x3 of Q, b = [e03, e13, e23] (note signs)
// Return true if solved (matrix invertible), false otherwise. err returns the error at the solution.
__device__ __forceinline__ bool QEM::solve_optimal(float3 &out, float &err) const {
    // Build A (symmetric)
    float A00 = e[0];
    float A01 = e[1];
    float A02 = e[2];
    float A11 = e[4];
    float A12 = e[5];
    float A22 = e[7];
    // b = (e03, e13, e23) where e03=e[3], e13=e[6], e23=e[8]
    float b0 = e[3];
    float b1 = e[6];
    float b2 = e[8];

    // Solve A * x = -b
    // Use analytic inverse for 3x3 symmetric matrix (compute determinant)
    // Compute determinant
    float det =
        A00 * (A11 * A22 - A12 * A12) -
        A01 * (A01 * A22 - A12 * A02) +
        A02 * (A01 * A12 - A11 * A02);

    if (fabsf(det) < 1e-12f) {
        // singular - fall back: pick minimal among corners (or average 0)
        // Here choose to put out as (0,0,0)
        out = make_float3(0.0f, 0.0f, 0.0f);
        err = evaluate(out);
        return false;
    }

    float invDet = 1.0f / det;

    // Compute inverse(A) via adjugate
    float inv00 =  (A11 * A22 - A12 * A12) * invDet;
    float inv01 = -(A01 * A22 - A12 * A02) * invDet;
    float inv02 =  (A01 * A12 - A11 * A02) * invDet;
    float inv11 =  (A00 * A22 - A02 * A02) * invDet;
    float inv12 = -(A00 * A12 - A01 * A02) * invDet;
    float inv22 =  (A00 * A11 - A01 * A01) * invDet;

    // x = -inv(A) * b
    float x = -(inv00 * b0 + inv01 * b1 + inv02 * b2);
    float y = -(inv01 * b0 + inv11 * b1 + inv12 * b2);
    float z = -(inv02 * b0 + inv12 * b1 + inv22 * b2);

    out = make_float3(x, y, z);
    err = evaluate(out);
    return true;
}


} // namespace cumesh
