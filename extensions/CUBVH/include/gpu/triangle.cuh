#pragma once

#include <gpu/common.h>

namespace cubvh {

// Triangle data structure
struct Triangle {

    __host__ __device__ Eigen::Vector3f sample_uniform_position(const Eigen::Vector2f& sample) const {
        float sqrt_x = std::sqrt(sample.x());
        float factor0 = 1.0f - sqrt_x;
        float factor1 = sqrt_x * (1.0f - sample.y());
        float factor2 = sqrt_x * sample.y();

        return factor0 * a + factor1 * b + factor2 * c;
    }

    __host__ __device__ float surface_area() const {
        return 0.5f * Eigen::Vector3f((b - a).cross(c - a)).norm();
    }

    __host__ __device__ Eigen::Vector3f normal() const {
        Eigen::Vector3f n = (b - a).cross(c - a);
        float norm = n.norm();
        if (norm > 1e-12f) {
            return n / norm;
        }
        return Eigen::Vector3f::Zero();
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, Eigen::Vector3f& n) const { // based on https://www.iquilezles.org/www/articles/intersectors/intersectors.htm
        Eigen::Vector3f v1v0 = b - a;
        Eigen::Vector3f v2v0 = c - a;
        Eigen::Vector3f rov0 = ro - a;
        n = v1v0.cross( v2v0 );
        Eigen::Vector3f q = rov0.cross( rd );
        float d = safe_divide(1.0f, rd.dot(n));
        float u = d*-q.dot( v2v0 );
        float v = d* q.dot( v1v0 );
        float t = d*-n.dot( rov0 );
        if( u<0.0f || u>1.0f || v<0.0f || (u+v)>1.0f || t<0.0f) t = 1e6f;
        return t; // Eigen::Vector3f( t, u, v );
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd) const {
        Eigen::Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    __host__ __device__ float distance_sq(const Eigen::Vector3f& pos) const {
        // prepare data
        Eigen::Vector3f v21 = b - a; Eigen::Vector3f p1 = pos - a;
        Eigen::Vector3f v32 = c - b; Eigen::Vector3f p2 = pos - b;
        Eigen::Vector3f v13 = a - c; Eigen::Vector3f p3 = pos - c;
        Eigen::Vector3f nor = v21.cross(v13);

        return
            // inside/outside test
            (sign(v21.cross(nor).dot(p1)) + sign(v32.cross(nor).dot(p2)) + sign(v13.cross(nor).dot(p3)) < 2.0f)
            ?
            // 3 edges
            std::min(
                std::min(
                    (v21 * clamp(v21.dot(p1) / v21.squaredNorm(), 0.0f, 1.0f)-p1).squaredNorm(),
                    (v32 * clamp(v32.dot(p2) / v32.squaredNorm(), 0.0f, 1.0f)-p2).squaredNorm()
                ),
                (v13 * clamp(v13.dot(p3) / v13.squaredNorm(), 0.0f, 1.0f)-p3).squaredNorm()
            )
            :
            // 1 face
            nor.dot(p1)*nor.dot(p1)/nor.squaredNorm();
    }

    __host__ __device__ float distance(const Eigen::Vector3f& pos) const {
        return std::sqrt(distance_sq(pos));
    }

    __host__ __device__ bool point_in_triangle(const Eigen::Vector3f& p) const {
        // Move the triangle so that the point becomes the
        // triangles origin
        Eigen::Vector3f local_a = a - p;
        Eigen::Vector3f local_b = b - p;
        Eigen::Vector3f local_c = c - p;

        // The point should be moved too, so they are both
        // relative, but because we don't use p in the
        // equation anymore, we don't need it!
        // p -= p;

        // Compute the normal vectors for triangles:
        // u = normal of PBC
        // v = normal of PCA
        // w = normal of PAB

        Eigen::Vector3f u = local_b.cross(local_c);
        Eigen::Vector3f v = local_c.cross(local_a);
        Eigen::Vector3f w = local_a.cross(local_b);

        // Test to see if the normals are facing
        // the same direction, return false if not
        if (u.dot(v) < 0.0f) {
            return false;
        }
        if (u.dot(w) < 0.0f) {
            return false;
        }

        // All normals facing the same way, return true
        return true;
    }

    __host__ __device__ Eigen::Vector3f closest_point_to_line(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c) const {
        Eigen::Vector3f ab = b - a;
        float denom = ab.squaredNorm();
        if (denom <= 1e-12f) {
            return a;
        }
        float t = (c - a).dot(ab) / denom;
        t = clamp(t, 0.0f, 1.0f);
        return a + t * ab;
    }

    __host__ __device__ Eigen::Vector3f closest_point(Eigen::Vector3f point) const {
        Eigen::Vector3f ab = b - a;
        Eigen::Vector3f ac = c - a;
        Eigen::Vector3f n = ab.cross(ac);
        float norm_sq = n.squaredNorm();
        if (norm_sq > 1e-12f) {
            float dist = n.dot(point - a) / norm_sq;
            point -= dist * n;
        }

        if (point_in_triangle(point)) {
            return point;
        }

        Eigen::Vector3f c1 = closest_point_to_line(a, b, point);
        Eigen::Vector3f c2 = closest_point_to_line(b, c, point);
        Eigen::Vector3f c3 = closest_point_to_line(c, a, point);

        float mag1 = (point - c1).squaredNorm();
        float mag2 = (point - c2).squaredNorm();
        float mag3 = (point - c3).squaredNorm();

        float min = std::min(mag1, mag2);
        min = std::min(min, mag3);

        if (min == mag1) {
            return c1;
        }
        else if (min == mag2) {
            return c2;
        }
        return c3;
    }

    __host__ __device__ Eigen::Vector3f barycentric(const Eigen::Vector3f& p) const {
        Eigen::Vector3f v0 = b - a;
        Eigen::Vector3f v1 = c - a;
        Eigen::Vector3f v2 = p - a;

        float d00 = v0.dot(v0);
        float d01 = v0.dot(v1);
        float d11 = v1.dot(v1);
        float d20 = v2.dot(v0);
        float d21 = v2.dot(v1);

        float denom = d00 * d11 - d01 * d01;
        if (fabsf(denom) < 1e-12f) {
            float ab = (b - a).squaredNorm();
            float bc = (c - b).squaredNorm();
            float ca = (a - c).squaredNorm();
            const float eps = 1e-12f;

            if (ab >= bc && ab >= ca && ab > eps) {
                float t = clamp((p - a).dot(b - a) / ab, 0.0f, 1.0f);
                return Eigen::Vector3f(1.0f - t, t, 0.0f);
            }
            if (bc >= ca && bc > eps) {
                float t = clamp((p - b).dot(c - b) / bc, 0.0f, 1.0f);
                return Eigen::Vector3f(0.0f, 1.0f - t, t);
            }
            if (ca > eps) {
                float t = clamp((p - c).dot(a - c) / ca, 0.0f, 1.0f);
                return Eigen::Vector3f(t, 0.0f, 1.0f - t);
            }
            return Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        }

        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;

        return Eigen::Vector3f(u, v, w);
    }

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3;
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    Eigen::Vector3f a, b, c;
    int64_t id;
};


inline std::ostream& operator<<(std::ostream& os, const Triangle& triangle) {
    os << "[";
    os << "a=[" << triangle.a.x() << "," << triangle.a.y() << "," << triangle.a.z() << "], ";
    os << "b=[" << triangle.b.x() << "," << triangle.b.y() << "," << triangle.b.z() << "], ";
    os << "c=[" << triangle.c.x() << "," << triangle.c.y() << "," << triangle.c.z() << "]";
    os << "]";
    return os;
}


}