// Metal shader data types
#ifndef MTLMESH_DTYPES_METAL
#define MTLMESH_DTYPES_METAL
#include <metal_stdlib>
using namespace metal;

// Metal's built-in packed_int3 is 12 bytes — matches PyTorch [F,3] int32 layout.
// Conversions: int3(packed_int3) and packed_int3(int3) are built-in.
// Helper aliases for readability in kernel code:
inline int3 to_int3(packed_int3 p) { return int3(p); }
inline packed_int3 from_int3(int3 v) { return packed_int3(v); }

struct Vec3f {
    float x, y, z;

    Vec3f() thread : x(0), y(0), z(0) {}
    Vec3f(float x, float y, float z) thread : x(x), y(y), z(z) {}
    Vec3f(float3 v) thread : x(v.x), y(v.y), z(v.z) {}

    Vec3f operator+(const thread Vec3f& o) const thread { return Vec3f(x+o.x, y+o.y, z+o.z); }
    thread Vec3f& operator+=(const thread Vec3f& o) thread { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3f operator-(const thread Vec3f& o) const thread { return Vec3f(x-o.x, y-o.y, z-o.z); }
    thread Vec3f& operator-=(const thread Vec3f& o) thread { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3f operator*(float s) const thread { return Vec3f(x*s, y*s, z*s); }
    thread Vec3f& operator*=(float s) thread { x*=s; y*=s; z*=s; return *this; }
    Vec3f operator/(float s) const thread { return Vec3f(x/s, y/s, z/s); }
    thread Vec3f& operator/=(float s) thread { x/=s; y/=s; z/=s; return *this; }

    float dot(const thread Vec3f& o) const thread { return x*o.x + y*o.y + z*o.z; }
    float norm() const thread { return sqrt(x*x + y*y + z*z); }
    float norm2() const thread { return x*x + y*y + z*z; }
    Vec3f normalized() const thread {
        float inv = rsqrt(x*x + y*y + z*z);
        return Vec3f(x*inv, y*inv, z*inv);
    }
    void normalize() thread {
        float inv = rsqrt(x*x + y*y + z*z);
        x *= inv; y *= inv; z *= inv;
    }
    Vec3f cross(const thread Vec3f& o) const thread {
        return Vec3f(y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x);
    }
    Vec3f slerp(const thread Vec3f& o, float t) const thread {
        float d = dot(o);
        d = clamp(d, -1.0f, 1.0f);
        float theta = acos(d) * t;
        Vec3f rel = (o - (*this) * d).normalized();
        return (*this) * cos(theta) + rel * sin(theta);
    }
    float3 to_float3() const thread { return float3(x, y, z); }
};

struct QEM {
    float e[10];

    void zero() thread {
        for (int i = 0; i < 10; i++) e[i] = 0.0f;
    }
    void add_plane(float4 p) thread {
        float a = p.x, b = p.y, c = p.z, d = p.w;
        e[0] += a*a; e[1] += a*b; e[2] += a*c; e[3] += a*d;
        e[4] += b*b; e[5] += b*c; e[6] += b*d;
        e[7] += c*c; e[8] += c*d;
        e[9] += d*d;
    }
    QEM operator+(const thread QEM& o) const thread {
        QEM res;
        for (int i = 0; i < 10; i++) res.e[i] = e[i] + o.e[i];
        return res;
    }
    thread QEM& operator+=(const thread QEM& o) thread {
        for (int i = 0; i < 10; i++) e[i] += o.e[i];
        return *this;
    }
    QEM operator-(const thread QEM& o) const thread {
        QEM res;
        for (int i = 0; i < 10; i++) res.e[i] = e[i] - o.e[i];
        return res;
    }
    thread QEM& operator-=(const thread QEM& o) thread {
        for (int i = 0; i < 10; i++) e[i] -= o.e[i];
        return *this;
    }
    float evaluate(const thread Vec3f& p) const thread {
        float x = p.x, y = p.y, z = p.z;
        return e[0]*x*x + 2*e[1]*x*y + 2*e[2]*x*z + 2*e[3]*x
             + e[4]*y*y + 2*e[5]*y*z + 2*e[6]*y
             + e[7]*z*z + 2*e[8]*z + e[9];
    }
    bool solve_optimal(thread float3& out, thread float& err) const thread {
        float det = e[0]*(e[4]*e[7]-e[5]*e[5]) - e[1]*(e[1]*e[7]-e[5]*e[2]) + e[2]*(e[1]*e[5]-e[4]*e[2]);
        if (abs(det) < 1e-12f) {
            out = float3(0); err = evaluate(Vec3f(0,0,0));
            return false;
        }
        float id = 1.0f / det;
        float i00 = (e[4]*e[7]-e[5]*e[5])*id, i01 = -(e[1]*e[7]-e[5]*e[2])*id, i02 = (e[1]*e[5]-e[4]*e[2])*id;
        float i11 = (e[0]*e[7]-e[2]*e[2])*id, i12 = -(e[0]*e[5]-e[1]*e[2])*id;
        float i22 = (e[0]*e[4]-e[1]*e[1])*id;
        float ox = -(i00*e[3]+i01*e[6]+i02*e[8]);
        float oy = -(i01*e[3]+i11*e[6]+i12*e[8]);
        float oz = -(i02*e[3]+i12*e[6]+i22*e[8]);
        out = float3(ox, oy, oz);
        err = evaluate(Vec3f(ox, oy, oz));
        return true;
    }
};

#endif // MTLMESH_DTYPES_METAL
