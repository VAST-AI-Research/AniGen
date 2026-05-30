// Geometry compute kernels - face areas, normals
#include <metal_stdlib>
using namespace metal;
#include "dtypes.metal"

kernel void compute_face_areas_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    constant int& F [[buffer(2)]],
    device float* face_areas [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 face = to_int3(faces[tid]);
    Vec3f v0 = Vec3f(float3(vertices[face.x]));
    Vec3f v1 = Vec3f(float3(vertices[face.y]));
    Vec3f v2 = Vec3f(float3(vertices[face.z]));
    face_areas[tid] = 0.5f * (v1 - v0).cross(v2 - v0).norm();
}

kernel void compute_face_normals_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    constant int& F [[buffer(2)]],
    device packed_float3* face_normals [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)F) return;
    int3 face = to_int3(faces[tid]);
    Vec3f v0 = Vec3f(float3(vertices[face.x]));
    Vec3f v1 = Vec3f(float3(vertices[face.y]));
    Vec3f v2 = Vec3f(float3(vertices[face.z]));
    Vec3f normal = (v1 - v0).cross(v2 - v0);
    normal.normalize();
    face_normals[tid] = packed_float3(normal.to_float3());
}

kernel void compute_vertex_normals_kernel(
    device const packed_float3* vertices [[buffer(0)]],
    device const packed_int3* faces [[buffer(1)]],
    device const int* vert2face [[buffer(2)]],
    device const int* vert2face_offset [[buffer(3)]],
    constant int& V [[buffer(4)]],
    device packed_float3* vertex_normals [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)V) return;
    int start = vert2face_offset[tid];
    int end = vert2face_offset[tid + 1];

    Vec3f normal(0.0f, 0.0f, 0.0f);
    Vec3f first_face_normal;
    for (int i = start; i < end; i++) {
        int fid = vert2face[i];
        int3 face = to_int3(faces[fid]);
        Vec3f v0 = Vec3f(float3(vertices[face.x]));
        Vec3f v1 = Vec3f(float3(vertices[face.y]));
        Vec3f v2 = Vec3f(float3(vertices[face.z]));
        Vec3f fn = (v1 - v0).cross(v2 - v0);
        normal = normal + fn;
        if (i == start) first_face_normal = fn;
    }
    normal.normalize();
    if (isnan(normal.x)) normal = first_face_normal;
    vertex_normals[tid] = packed_float3(normal.to_float3());
}
