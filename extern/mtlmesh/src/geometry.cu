#include "cumesh.h"
#include "dtypes.cuh"
#include "shared.h"
#include <cub/cub.cuh>


namespace cumesh {


static __global__ void compute_face_areas_kernel(
    const float3* vertices,
    const int3* faces,
    const size_t F,
    float* face_areas
) {
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= F) return;
    int3 face = faces[fid];
    Vec3f v0 = Vec3f(vertices[face.x]);
    Vec3f v1 = Vec3f(vertices[face.y]);
    Vec3f v2 = Vec3f(vertices[face.z]);
    face_areas[fid] = 0.5 * (v1 - v0).cross(v2 - v0).norm();
}


void CuMesh::compute_face_areas() {
    size_t F = this->faces.size;
    this->face_areas.resize(F);
    compute_face_areas_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->faces.ptr,
        F,
        this->face_areas.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


static __global__ void compute_face_normals_kernel(
    const float3* vertices,
    const int3* faces,
    const size_t F,
    float3* face_normals
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 face = faces[tid];
    Vec3f v0 = Vec3f(vertices[face.x]);
    Vec3f v1 = Vec3f(vertices[face.y]);
    Vec3f v2 = Vec3f(vertices[face.z]);

    Vec3f normal = (v1 - v0).cross(v2 - v0);
    normal.normalize();
    face_normals[tid] = make_float3(normal.x, normal.y, normal.z);
}


void CuMesh::compute_face_normals() {
    size_t F = this->faces.size;
    this->face_normals.resize(F);
    compute_face_normals_kernel<<<(F + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->faces.ptr,
        F,
        this->face_normals.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


static __global__ void compute_vertex_normals_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const size_t V,
    float3* vertex_normals
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    int start = vert2face_offset[tid];
    int end = vert2face_offset[tid + 1];

    Vec3f normal(0.0f, 0.0f, 0.0f);
    Vec3f first_face_normal;
    for (int i = start; i < end; i++) {
        int fid = vert2face[i];
        int3 face = faces[fid];
        Vec3f v0 = Vec3f(vertices[face.x]);
        Vec3f v1 = Vec3f(vertices[face.y]);
        Vec3f v2 = Vec3f(vertices[face.z]);

        Vec3f face_normal = (v1 - v0).cross(v2 - v0);
        normal += face_normal;
        if (i == start) {
            first_face_normal = face_normal;
        }
    }

    normal.normalize();
    // if NAN, fallback to first face normal
    if (isnan(normal.x)) {
        normal = first_face_normal;
    }
    vertex_normals[tid] = make_float3(normal.x, normal.y, normal.z);
}


void CuMesh::compute_vertex_normals() {
    if (this->vert2face.is_empty() || this->vert2face_offset.is_empty()) {
        this->get_vertex_face_adjacency();
    }

    size_t V = this->vertices.size;
    this->vertex_normals.resize(V);
    compute_vertex_normals_kernel<<<(V + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vertices.ptr,
        this->faces.ptr,
        this->vert2face.ptr,
        this->vert2face_offset.ptr,
        V,
        this->vertex_normals.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


} // namespace cumesh