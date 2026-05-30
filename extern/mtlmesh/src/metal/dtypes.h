// Data types shared between host (C++) and device (.metal)
#pragma once

#include <cmath>
#include <simd/simd.h>

namespace mtlmesh {

// 12-byte packed int3 — matches PyTorch [F,3] int32 layout (no padding).
struct packed_int3 {
    int x, y, z;
};

struct Vec3f {
    float x, y, z;

    Vec3f() : x(0), y(0), z(0) {}
    Vec3f(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3f(simd_float3 v) : x(v.x), y(v.y), z(v.z) {}

    Vec3f operator+(const Vec3f& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3f& operator+=(const Vec3f& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3f operator-(const Vec3f& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3f& operator-=(const Vec3f& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3f operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3f& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
    Vec3f operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3f& operator/=(float s) { x /= s; y /= s; z /= s; return *this; }

    float dot(const Vec3f& o) const { return x * o.x + y * o.y + z * o.z; }
    float norm() const { return sqrtf(x * x + y * y + z * z); }
    float norm2() const { return x * x + y * y + z * z; }
    Vec3f normalized() const {
        float inv = 1.0f / sqrtf(x * x + y * y + z * z);
        return {x * inv, y * inv, z * inv};
    }
    void normalize() {
        float inv = 1.0f / sqrtf(x * x + y * y + z * z);
        x *= inv; y *= inv; z *= inv;
    }
    Vec3f cross(const Vec3f& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    Vec3f slerp(const Vec3f& o, float t) const {
        float d = dot(o);
        d = fmaxf(fminf(d, 1.0f), -1.0f);
        float theta = acosf(d) * t;
        Vec3f rel = (o - (*this) * d).normalized();
        return (*this) * cosf(theta) + rel * sinf(theta);
    }

    simd_float3 to_simd() const { return simd_make_float3(x, y, z); }
};

struct QEM {
    float e[10];

    QEM() { zero(); }
    QEM operator+(const QEM& o) const {
        QEM res;
        for (int i = 0; i < 10; i++) res.e[i] = e[i] + o.e[i];
        return res;
    }
    QEM& operator+=(const QEM& o) {
        for (int i = 0; i < 10; i++) e[i] += o.e[i];
        return *this;
    }
    void zero() {
        for (int i = 0; i < 10; i++) e[i] = 0.0f;
    }
    void add_plane(float a, float b, float c, float d) {
        e[0] += a * a; e[1] += a * b; e[2] += a * c; e[3] += a * d;
        e[4] += b * b; e[5] += b * c; e[6] += b * d;
        e[7] += c * c; e[8] += c * d;
        e[9] += d * d;
    }
    float evaluate(const Vec3f& p) const {
        float x = p.x, y = p.y, z = p.z;
        return e[0]*x*x + 2*e[1]*x*y + 2*e[2]*x*z + 2*e[3]*x
             + e[4]*y*y + 2*e[5]*y*z + 2*e[6]*y
             + e[7]*z*z + 2*e[8]*z
             + e[9];
    }
    bool solve_optimal(simd_float3& out, float& err) const {
        float det = e[0]*(e[4]*e[7]-e[5]*e[5]) - e[1]*(e[1]*e[7]-e[5]*e[2]) + e[2]*(e[1]*e[5]-e[4]*e[2]);
        if (fabsf(det) < 1e-12f) {
            out = simd_make_float3(0, 0, 0);
            err = evaluate({0, 0, 0});
            return false;
        }
        float id = 1.0f / det;
        float i00 =  (e[4]*e[7]-e[5]*e[5])*id;
        float i01 = -(e[1]*e[7]-e[5]*e[2])*id;
        float i02 =  (e[1]*e[5]-e[4]*e[2])*id;
        float i11 =  (e[0]*e[7]-e[2]*e[2])*id;
        float i12 = -(e[0]*e[5]-e[1]*e[2])*id;
        float i22 =  (e[0]*e[4]-e[1]*e[1])*id;
        float ox = -(i00*e[3]+i01*e[6]+i02*e[8]);
        float oy = -(i01*e[3]+i11*e[6]+i12*e[8]);
        float oz = -(i02*e[3]+i12*e[6]+i22*e[8]);
        out = simd_make_float3(ox, oy, oz);
        err = evaluate({ox, oy, oz});
        return true;
    }
};

} // namespace mtlmesh
