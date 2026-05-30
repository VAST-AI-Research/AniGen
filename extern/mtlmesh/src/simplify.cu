#include "cumesh.h"
#include "dtypes.cuh"
#include <cub/cub.cuh>


namespace cumesh {


__device__ inline uint64_t pack_key_value_positive(int key, float value) {
    unsigned int v = __float_as_uint(value);
    return (static_cast<uint64_t>(v) << 32) |
           static_cast<unsigned int>(key);
}


__device__ inline void unpack_key_value_positive(uint64_t key_value, int& key, float& value) {
    key = static_cast<int>(key_value & 0xffffffffu);
    value = __uint_as_float(static_cast<unsigned int>(key_value >> 32));
}


/**
 * Get the QEM for each vertex
 * 
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param vert2face: the buffer for neighbor face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighbor face ids offset, shape (V+1)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param qems: the buffer for QEMs, shape (V)
 */
static __global__ void get_qem_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const int V,
    const int F,
    QEM* qems
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // compute QEM
    QEM v_qem;
    for (int f = vert2face_offset[tid]; f < vert2face_offset[tid+1]; f++) {
        int3 f_vids = faces[vert2face[f]];
        Vec3f f_v0(vertices[f_vids.x]);
        Vec3f e1(vertices[f_vids.y]);
        Vec3f e2(vertices[f_vids.z]);
        e1 -= f_v0;
        e2 -= f_v0;
        Vec3f n = e1.cross(e2);
        n.normalize();
        float d = -(n.dot(f_v0));
        v_qem.add_plane({ n.x, n.y, n.z, d });
    }
    qems[tid] = v_qem;
}


/**
 * Get the QEM for each vertex
 */
void get_qem(
    CuMesh& ctx
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    ctx.temp_storage.resize(V * sizeof(QEM));
    get_qem_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        V, F,
        reinterpret_cast<QEM*>(ctx.temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
}


inline __device__ bool process_incident_tri(
    int tri_idx,
    int collapse_keep_vert, // the vertex we keep (e0 or e1)
    int collapse_other_vert, // the other one (the one removed)
    const float3* vertices,
    const int3* faces,
    const Vec3f& v_new, // midpoint
    float& skinny_cost,
    int& num_tri
) {
    const float EPS = 1e-12f;
    int3 f_vids = faces[tri_idx];

    // If this triangle contains the other vertex (the edge), it will be removed, skip it
    if (f_vids.x == collapse_other_vert || f_vids.y == collapse_other_vert || f_vids.z == collapse_other_vert)
        return true; // skip, not an error

    // get old positions
    Vec3f a(vertices[f_vids.x]);
    Vec3f b(vertices[f_vids.y]);
    Vec3f c(vertices[f_vids.z]);

    // build new positions: replace occurrences of collapse_keep_vert with v_new
    Vec3f na = (f_vids.x == collapse_keep_vert) ? v_new : a;
    Vec3f nb = (f_vids.y == collapse_keep_vert) ? v_new : b;
    Vec3f nc = (f_vids.z == collapse_keep_vert) ? v_new : c;

    // compute old edge vectors (for old normal)
    Vec3f old_e1 = b - a;
    Vec3f old_e2 = c - a;
    Vec3f old_normal = old_e1.cross(old_e2);
    float old_area = 0.5f * old_normal.norm();

    // compute new edge vectors consistently: e1 = nb - na, e2 = nc - na
    Vec3f new_e1 = nb - na;
    Vec3f new_e2 = nc - na;
    Vec3f new_normal = new_e1.cross(new_e2);
    float new_area = 0.5f * new_normal.norm();

    // check flipping
    if (old_normal.dot(new_normal) < 0.0f) {
        return false; // invalid (flipped)
    }

    // compute side lengths squared for shape metric
    Vec3f new_e0 = nc - nb;
    float denom = new_e0.norm2() + new_e1.norm2() + new_e2.norm2();
    if (denom < EPS) denom = EPS;
    float shapeMetric = 4.0f * sqrtf(3.0f) * new_area / denom;
    float term = 1.0f - fminf(fmaxf(shapeMetric, 0.0f), 1.0f);
    skinny_cost += term;
    num_tri += 1;
    return true;
}


/**
 * Get the cost for each edge collapse
 * 
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param vert2face: the buffer for neighbor face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighbor face ids offset, shape (V+1)
 * @param edges: the buffer for edges, shape (E)
 * @param vert_is_boundary: the buffer for boundary vertex indicator, shape (V)
 * @param qems: the buffer for QEMs, shape (V)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 */
static __global__ void get_edge_collapse_cost_kernel(
    const float3* vertices,
    const int3* faces,
    const int* vert2face,
    const int* vert2face_offset,
    const uint64_t* edges,
    const uint8_t * vert_is_boundary,
    const QEM* qems,
    const int V,
    const int F,
    const int E,
    const float lambda_edge_length,
    const float lambda_skinny,
    float* edge_collapse_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // get edge vertices
    Vec3f v0(vertices[e0]);
    Vec3f v1(vertices[e1]);
    uint8_t v0_is_bound = vert_is_boundary[e0];
    uint8_t v1_is_bound = vert_is_boundary[e1];
    float w0 = 0.5;
    if (v0_is_bound && !v1_is_bound) w0 = 1.0;
    else if (!v0_is_bound &&  v1_is_bound) w0 = 0.0;
    Vec3f v = v0 * w0 + v1 * (1.0f - w0);

    float cost = 0.0f;

    // QEM cost
    QEM edge_qem = qems[e0] + qems[e1];
    float qem_cost = edge_qem.evaluate(v);
    cost += qem_cost;

    // edge length cost
    float edge_length2 = (v1 - v0).norm2();
    cost += lambda_edge_length * edge_length2;

    // skinny cost
    float skinny_cost = 0.0f;
    int num_tri = 0;
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int tri_idx = vert2face[f];
        if (!process_incident_tri(tri_idx, e0, e1, vertices, faces, v, skinny_cost, num_tri)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int tri_idx = vert2face[f];
        if (!process_incident_tri(tri_idx, e1, e0, vertices, faces, v, skinny_cost, num_tri)) {
            edge_collapse_costs[tid] = INFINITY;
            return;
        }
    }
    if (num_tri > 0) {
        skinny_cost /= num_tri;
    }
    cost += lambda_skinny * skinny_cost * edge_length2;

    edge_collapse_costs[tid] = cost;
}


/**
 * Get the cost for each edge collapse
 */
void get_edge_collapse_cost(
    CuMesh& ctx,
    float lambda_edge_length,
    float lambda_skinny
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.edge_collapse_costs.resize(E);
    get_edge_collapse_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edges.ptr,
        ctx.vert_is_boundary.ptr,
        reinterpret_cast<const QEM*>(ctx.temp_storage.ptr),
        V, F, E,
        lambda_edge_length, lambda_skinny,
        ctx.edge_collapse_costs.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Propagate cost to neighboring faces
 * 
 * @param edges: the buffer for edges, shape (E)
 * @param vert2face: the buffer for neighboring face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighboring face ids offset, shape (V+1)
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param propagated_costs: the buffer for edge collapse costs propagated, shape (F)
 */
static __global__ void propagate_cost_kernel(
    const uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const int V,
    const int F,
    const int E,
    uint64_t* propagated_costs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    uint64_t cost = pack_key_value_positive(tid, edge_collapse_costs[tid]);

    // propagate cost to neighboring faces
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]), static_cast<unsigned long long>(cost));
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        atomicMin(reinterpret_cast<unsigned long long*>(&propagated_costs[vert2face[f]]), static_cast<unsigned long long>(cost));
    }
}


/**
 * Propagate cost to neighboring faces
 */
void propagate_cost(
    CuMesh& ctx
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.propagated_costs.resize(F);
    ctx.propagated_costs.fill(std::numeric_limits<uint64_t>::max());
    propagate_cost_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.edges.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edge_collapse_costs.ptr,
        V, F, E,
        ctx.propagated_costs.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Collapse edges parallelly
 * 
 * @param vertices: the vertices of the mesh, shape (V)
 * @param faces: the faces of the mesh, shape (F)
 * @param edges: the buffer for edges, shape (E)
 * @param vert2face: the buffer for neighboring face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighboring face ids offset, shape (V+1)
 * @param edge_collapse_costs: the buffer for edge collapse costs, shape (E)
 * @param propagated_costs: the buffer for edge collapse costs propagated, shape (F)
 * @param vert_is_boundary: the buffer for boundary vertex indicator, shape (V)
 * @param V: the number of vertices
 * @param F: the number of faces
 * @param E: the number of edges
 * @param collapse_thresh: the threshold for cost collapse
 * @param vertices_kept: the flag for vertices kept, shape (V)
 * @param faces_kept: the flag for faces kept, shape (F)
 */
static __global__ void collapse_edges_kernel(
    float3* vertices,
    int3* faces,
    uint64_t* edges,
    const int* vert2face,
    const int* vert2face_offset,
    const float* edge_collapse_costs,
    const uint64_t* propagated_costs,
    const uint8_t * vert_is_boundary,
    const int V,
    const int F,
    const int E,
    const float collapse_thresh,
    int* vertices_kept,
    int* faces_kept
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    float cost = edge_collapse_costs[tid];
    if (cost > collapse_thresh) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);
    uint64_t pack = pack_key_value_positive(tid, cost);

    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        if (propagated_costs[vert2face[f]] != pack) return;
    }

    // collapse edge
    Vec3f v0(vertices[e0]);
    Vec3f v1(vertices[e1]);
    uint8_t v0_is_bound = vert_is_boundary[e0];
    uint8_t v1_is_bound = vert_is_boundary[e1];
    float w0 = 0.5;
    if (v0_is_bound && !v1_is_bound) w0 = 1.0;
    else if (!v0_is_bound &&  v1_is_bound) w0 = 0.0;
    Vec3f v_new = v0 * w0 + v1 * (1.0f - w0);
    vertices[e0] = { v_new.x, v_new.y, v_new.z };
    vertices_kept[e1] = 0;
    // delete shared faces
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1) {
            faces_kept[fid] = 0;
        }
    }
    // update faces
    for (int f = vert2face_offset[e1]; f < vert2face_offset[e1+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1) {
            f_vids.x = e0;
        } else if (f_vids.y == e1) {
            f_vids.y = e0;
        } else if (f_vids.z == e1) {
            f_vids.z = e0;
        }
        faces[fid] = f_vids;
    }
}


static __global__ void compress_vertices_kernel(
    const int* vertices_map,
    const float3* old_vertices,
    const int V,
    float3* new_vertices
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;
    int new_id = vertices_map[tid];
    int is_kept = vertices_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_vertices[new_id] = old_vertices[tid];
    }
}


static __global__ void compress_faces_kernel(
    const int* faces_map,
    const int* vertices_map,
    const int3* old_faces,
    const int F,
    int3* new_faces
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;
    int new_id = faces_map[tid];
    int is_kept = faces_map[tid + 1] == new_id + 1;
    if (is_kept) {
        new_faces[new_id].x = vertices_map[old_faces[tid].x];
        new_faces[new_id].y = vertices_map[old_faces[tid].y];
        new_faces[new_id].z = vertices_map[old_faces[tid].z];        
    }
}


/**
 * Collapse edges parallelly
 */
void collapse_edges(
    CuMesh& ctx,
    float collapse_thresh
) {
    size_t V = ctx.vertices.size;
    size_t F = ctx.faces.size;
    size_t E = ctx.edges.size;
    ctx.vertices_map.resize(V + 1);
    ctx.faces_map.resize(F + 1);
    ctx.vertices_map.fill(1);
    ctx.faces_map.fill(1);
    collapse_edges_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices.ptr,
        ctx.faces.ptr,
        ctx.edges.ptr,
        ctx.vert2face.ptr,
        ctx.vert2face_offset.ptr,
        ctx.edge_collapse_costs.ptr,
        ctx.propagated_costs.ptr,
        ctx.vert_is_boundary.ptr,
        V, F, E,
        collapse_thresh,
        ctx.vertices_map.ptr,
        ctx.faces_map.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    
    // update vertices buffer
    // get vertices map
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        ctx.vertices_map.ptr, V+1
    ));
    ctx.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.vertices_map.ptr, V+1
    ));
    int new_num_vertices;
    CUDA_CHECK(cudaMemcpy(&new_num_vertices, ctx.vertices_map.ptr + V, sizeof(int), cudaMemcpyDeviceToHost));
    // compress vertices
    ctx.temp_storage.resize(new_num_vertices * sizeof(float3));
    compress_vertices_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.vertices_map.ptr,
        ctx.vertices.ptr,
        V,
        reinterpret_cast<float3*>(ctx.temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    swap_buffers(ctx.temp_storage, ctx.vertices);

    // update faces buffer
    // get faces map
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        ctx.faces_map.ptr, F+1
    ));
    ctx.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        ctx.cub_temp_storage.ptr, temp_storage_bytes,
        ctx.faces_map.ptr, F+1
    ));
    int new_num_faces;
    CUDA_CHECK(cudaMemcpy(&new_num_faces, ctx.faces_map.ptr + F, sizeof(int), cudaMemcpyDeviceToHost));
    // compress faces
    ctx.temp_storage.resize(new_num_faces * sizeof(int3));
    compress_faces_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        ctx.faces_map.ptr,
        ctx.vertices_map.ptr,
        ctx.faces.ptr,
        F,
        reinterpret_cast<int3*>(ctx.temp_storage.ptr)
    );
    CUDA_CHECK(cudaGetLastError());
    swap_buffers(ctx.temp_storage, ctx.faces);
}


std::tuple<int, int> CuMesh::simplify_step(float lambda_edge_length, float lambda_skinny, float threshold, bool timing) {
    std::chrono::high_resolution_clock::time_point start, end;

    if (timing) start = std::chrono::high_resolution_clock::now();
    this->get_vertex_face_adjacency();
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_vertex_face_adjacency: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    this->get_edges();
    this->get_boundary_info();
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_edges: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    get_qem(*this);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_qem: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    get_edge_collapse_cost(*this, lambda_edge_length, lambda_skinny);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "get_edge_collapse_cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    propagate_cost(*this);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "propagate_cost: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    if (timing) start = std::chrono::high_resolution_clock::now();
    collapse_edges(*this, threshold);
    if (timing) {
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        std::cout << "collapse_edges: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us" << std::endl;
    }

    // Delete all cached info since mesh has changed
    this->clear_cache();

    return std::make_tuple(this->vertices.size, this->faces.size);
}


} // namespace cumesh
