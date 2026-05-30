#include "cumesh.h"
#include "shared.h"

#include <cub/cub.cuh>


namespace cumesh {

/**
 * Get count of neighboring faces for each vertex
 * 
 * @param faces: the faces of the mesh, shape (F)
 * @param F: the number of faces
 * @param neighbor_face_cnt: the buffer for neighbor face count, shape (V+1)
 */
static __global__ void get_neighbor_face_cnt_kernel(
    const int3* faces,
    const int F,
    int* neighbor_face_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 f = faces[tid];

    atomicAdd(&neighbor_face_cnt[f.x], 1);
    atomicAdd(&neighbor_face_cnt[f.y], 1);
    atomicAdd(&neighbor_face_cnt[f.z], 1);
}


/**
 * Fill the neighboring face ids for each vertex
 * 
 * @param faces: the faces of the mesh, shape (F)
 * @param F: the number of faces
 * @param neighbor_face_ids: the buffer for neighbor face ids, shape (total_neighbor_face_cnt)
 * @param neighbor_face_ids_offset: the buffer for neighbor face ids offset, shape (V+1)
 * @param neighbor_face_cnt: the buffer for neighbor face count, shape (V+1)
 */
static __global__ void fill_neighbor_face_ids_kernel(
    const int3* faces,
    const int F,
    int* neighbor_face_ids,
    int* neighbor_face_ids_offset,
    int* neighbor_face_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int3 f = faces[tid];

    neighbor_face_ids[neighbor_face_ids_offset[f.x] + atomicAdd(&neighbor_face_cnt[f.x], 1)] = tid;
    neighbor_face_ids[neighbor_face_ids_offset[f.y] + atomicAdd(&neighbor_face_cnt[f.y], 1)] = tid;
    neighbor_face_ids[neighbor_face_ids_offset[f.z] + atomicAdd(&neighbor_face_cnt[f.z], 1)] = tid;
}


void CuMesh::get_vertex_face_adjacency() {
    size_t F = this->faces.size;
    size_t V = this->vertices.size;

    // get neighboring face count for each vertex
    this->vert2face_cnt.resize(V + 1);
    this->vert2face_cnt.zero();
    get_neighbor_face_cnt_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(this->faces.ptr, F, this->vert2face_cnt.ptr);
    CUDA_CHECK(cudaGetLastError());

    // allocate memory for neighboring face ids
    this->vert2face_offset.resize(V + 1);
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        this->vert2face_cnt.ptr, this->vert2face_offset.ptr,
        V + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->vert2face_cnt.ptr, this->vert2face_offset.ptr,
        V + 1
    ));
    this->vert2face.resize(F*3);

    // fill neighboring face ids for each vertex
    this->vert2face_cnt.zero();
    fill_neighbor_face_ids_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(this->faces.ptr, F, 
        this->vert2face.ptr,
        this->vert2face_offset.ptr,
        this->vert2face_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Expand edges for each triangle face
 * 
 * @param faces: the faces of the mesh, shape (F)
 * @param F: the number of faces
 * @param edges: the buffer for edges, shape (F*3)
 */
static __global__ void expand_edges_kernel(
    const int3* faces,
    const int F,
    uint64_t *edges
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int base = tid * 3;
    int3 f = faces[tid];
    
    // expand edges
    edges[base + 0] = ((uint64_t)min(f.x, f.y) << 32) | max(f.x, f.y);
    edges[base + 1] = ((uint64_t)min(f.y, f.z) << 32) | max(f.y, f.z);
    edges[base + 2] = ((uint64_t)min(f.z, f.x) << 32) | max(f.z, f.x);
}


void CuMesh::get_edges() {
    size_t F = this->faces.size;
    this->edges.resize(F * 3);
    expand_edges_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(this->faces.ptr, F, this->edges.ptr);
    CUDA_CHECK(cudaGetLastError());

    // sort edges
    this->temp_storage.resize(F * 3 * sizeof(uint64_t));
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes,
        this->edges.ptr,
        reinterpret_cast<uint64_t*>(this->temp_storage.ptr),
        F * 3
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->edges.ptr,
        reinterpret_cast<uint64_t*>(this->temp_storage.ptr),
        F * 3
    ));

    // unique edges
    int* num_edges;
    CUDA_CHECK(cudaMalloc(&num_edges, sizeof(int)));
    this->edge2face_cnt.resize(F * 3);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(this->temp_storage.ptr), this->edges.ptr, this->edge2face_cnt.ptr, num_edges,
        F * 3
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(this->temp_storage.ptr), this->edges.ptr, this->edge2face_cnt.ptr, num_edges,
        F * 3
    ));
    CUDA_CHECK(cudaMemcpy(&this->edges.size, num_edges, sizeof(int), cudaMemcpyDeviceToHost));
    this->edge2face_cnt.size = this->edges.size;
    CUDA_CHECK(cudaFree(num_edges));
}


/**
 * Get edge-face adjacency
 * 
 * @param faces: the faces of the mesh, shape (F)
 * @param edges: the buffer for edges, shape (E)
 * @param edge2face_cnt: the buffer for edge duplication number, shape (E)
 * @param vert2face: the buffer for neighboring face ids, shape (total_neighbor_face_cnt)
 * @param vert2face_offset: the buffer for neighboring face ids offset, shape (V+1)
 * @param edge2face_offset: the buffer for edge to face adjacency offset, shape (E+1)
 * @param E: the number of edges
 * @param edge2face: the buffer for edge to face adjacency, shape (total_edge_face_cnt)
 * @param face2edge: the buffer for face to edge adjacency, shape (F*3)
 */
static __global__ void get_edge_face_adjacency_kernel(
    const int3* faces,
    const uint64_t* edges,
    const int* edge2face_cnt,
    const int* vert2face,
    const int* vert2face_offset,
    const int* edge2face_offset,
    const int E,
    int* edge2face,
    int3* face2edge
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // assign connectivity
    int ptr = edge2face_offset[tid];
    for (int f = vert2face_offset[e0]; f < vert2face_offset[e0+1]; f++) {
        int fid = vert2face[f];
        int3 f_vids = faces[fid];
        if (f_vids.x == e1 || f_vids.y == e1 || f_vids.z == e1) {
            // this face contains the edge
            edge2face[ptr] = fid;
            ptr++;
            // fill face2edge
            if (f_vids.x == e0 && f_vids.y == e1 || f_vids.x == e1 && f_vids.y == e0) {
                face2edge[fid].x = tid;
            } else if (f_vids.y == e0 && f_vids.z == e1 || f_vids.y == e1 && f_vids.z == e0) {
                face2edge[fid].y = tid;
            } else if (f_vids.z == e0 && f_vids.x == e1 || f_vids.z == e1 && f_vids.x == e0) {
                face2edge[fid].z = tid;
            }
        }
    }
}


void CuMesh::get_edge_face_adjacency() {
    if (this->edges.is_empty() || this->edge2face_cnt.is_empty()) {
        this->get_edges();
    }
    if (this->vert2face.is_empty() || this->vert2face_offset.is_empty()) {
        this->get_vertex_face_adjacency();    
    }
    size_t F = this->faces.size;
    size_t E = this->edges.size;

    // allocate memory for edge2face_offset
    this->edge2face_offset.resize(E + 1);
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        this->edge2face_cnt.ptr, this->edge2face_offset.ptr,
        E + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->edge2face_cnt.ptr, this->edge2face_offset.ptr,
        E + 1
    ));

    // allocate memory for edge2face
    int total_edge_face_cnt;
    CUDA_CHECK(cudaMemcpy(&total_edge_face_cnt, &this->edge2face_offset.ptr[E], sizeof(int), cudaMemcpyDeviceToHost));
    this->edge2face.resize(total_edge_face_cnt);

    // allocate memory for face2edge
    this->face2edge.resize(F);

    // get edge-face adjacency
    get_edge_face_adjacency_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->faces.ptr,
        this->edges.ptr,
        this->edge2face_cnt.ptr,
        this->vert2face.ptr,
        this->vert2face_offset.ptr,
        this->edge2face_offset.ptr,
        E,
        this->edge2face.ptr,
        this->face2edge.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Get vertex adjacent edge number
 * 
 * @param edges: the buffer for edges, shape (E)
 * @param E: the number of edges
 * @param vert2edge_cnt: the buffer for vertex adjacent edge number, shape (V)
 */
static __global__ void get_vertex_edge_cnt_kernel(
    const uint64_t* edges,
    const int E,
    int* vert2edge_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // count vertex adjacent edge number
    atomicAdd(&vert2edge_cnt[e0], 1);
    atomicAdd(&vert2edge_cnt[e1], 1);
}


/**
 * Get vertex-edge adjacency
 * 
 * @param edges: the buffer for edges, shape (E)
 * @param E: the number of edges
 * @param vert2edge: the buffer for vertex to edge adjacency, shape (total_vertex_edge_cnt)
 * @param vert2edge_offset: the buffer for vertex to edge adjacency offset, shape (V+1)
 * @param vert2edge_cnt: the buffer for vertex adjacent edge number, shape (V)
 */
static __global__ void get_vertex_edge_adjacency_kernel(
    const uint64_t* edges,
    const int E,
    int* vert2edge,
    int* vert2edge_offset,
    int* vert2edge_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= E) return;

    // get edge
    uint64_t e = edges[tid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // assign connectivity
    vert2edge[vert2edge_offset[e0] + atomicAdd(&vert2edge_cnt[e0], 1)] = tid;
    vert2edge[vert2edge_offset[e1] + atomicAdd(&vert2edge_cnt[e1], 1)] = tid;
}


void CuMesh::get_vertex_edge_adjacency() {
    if (this->edges.is_empty()) {
        this->get_edges();
    }
    size_t E = this->edges.size;
    size_t V = this->vertices.size;

    // get vertex adjacent edge number
    this->vert2edge_cnt.resize(V + 1);
    this->vert2edge_cnt.zero();
    get_vertex_edge_cnt_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr, E, this->vert2edge_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // allocate memory for vert2edge_offset
    this->vert2edge_offset.resize(V + 1);
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        this->vert2edge_cnt.ptr, this->vert2edge_offset.ptr,
        V + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->vert2edge_cnt.ptr, this->vert2edge_offset.ptr,
        V + 1
    ));

    // get vertex-edge adjacency
    this->vert2edge.resize(2 * E);
    this->vert2edge_cnt.zero();
    get_vertex_edge_adjacency_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr, E,
        this->vert2edge.ptr,
        this->vert2edge_offset.ptr,
        this->vert2edge_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Set vertex boundary indicator
 * 
 * @param edges: the buffer for edges, shape (E)
 * @param boundaries: the buffer for boundary edges, shape (B)
 * @param edge2face_cnt: the buffer for edge duplication number, shape (E)
 * @param B: the number of boundary edges
 * @param vert_is_boundary: the buffer for boundary vertex indicator, shape (V)
 */
static __global__ void set_boundary_vertex_kernel(
    const uint64_t* edges,
    const int* boundaries,
    const int* edge2face_cnt,
    const int B,
    uint8_t* vert_is_boundary
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    int eid = boundaries[tid];

    if (edge2face_cnt[eid] == 1) {
        // get edge
        uint64_t e = edges[eid];
        int e0 = int(e >> 32);
        int e1 = int(e & 0xFFFFFFFF);

        // set boundary vertex
        vert_is_boundary[e0] = 1;
        vert_is_boundary[e1] = 1;
    }
}


struct is_boundary_edge {
    const int* edge2face_cnt;
    __host__ __device__
    bool operator()(const int& idx) const {
        return edge2face_cnt[idx] == 1;
    }
};


void CuMesh::get_boundary_info() {
    if (this->edges.is_empty() || this->edge2face_cnt.is_empty()) {
        this->get_edges();
    }
    size_t E = this->edges.size;

    // Select boundary edges
    size_t temp_storage_bytes = 0;
    int *cu_num_boundary, *cu_edge_idx;
    CUDA_CHECK(cudaMalloc(&cu_num_boundary, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_edge_idx, E * sizeof(int)));
    this->boundaries.resize(E);
    arange_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_edge_idx, E);
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, temp_storage_bytes,
        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,
        E,
        is_boundary_edge{this->edge2face_cnt.ptr}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::If(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_edge_idx, this->boundaries.ptr, cu_num_boundary,
        E,
        is_boundary_edge{this->edge2face_cnt.ptr}
    ));
    CUDA_CHECK(cudaMemcpy(&this->boundaries.size, cu_num_boundary, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_num_boundary));
    CUDA_CHECK(cudaFree(cu_edge_idx));

    // Set vertex boundary indicator
    this->vert_is_boundary.resize(this->vertices.size);
    this->vert_is_boundary.zero();
    if (this->boundaries.size > 0) {
        set_boundary_vertex_kernel<<<(this->boundaries.size+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->edges.ptr, this->boundaries.ptr, this->edge2face_cnt.ptr,
            this->boundaries.size, this->vert_is_boundary.ptr
        );
        CUDA_CHECK(cudaGetLastError());
    }
}


static __global__ void get_vertex_boundary_cnt_kernel(
    const uint64_t* edges,
    const int* boundaries,
    const int B,
    int* vert2bound_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    int eid = boundaries[tid];

    // get edge
    uint64_t e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // count vertex adjacent boundary number
    atomicAdd(&vert2bound_cnt[e0], 1);
    atomicAdd(&vert2bound_cnt[e1], 1);
}


/**
 * Get vertex-boundary adjacency
 * 
 * @param edges: the buffer for edges, shape (E)
 * @param boundaries: the buffer for boundary edges, shape (B)
 * @param B: the number of boundary edges
 * @param vert2bound: the buffer for vertex to boundary adjacency, shape (total_vertex_boundary_cnt)
 * @param vert2bound_offset: the buffer for vertex to boundary adjacency offset, shape (V+1)
 * @param vert2bound_cnt: the buffer for vertex adjacent boundary number, shape (V)
 */
static __global__ void get_vertex_boundary_adjacency_kernel(
    const uint64_t* edges,
    const int* boundaries,
    const int B,
    int* vert2bound,
    int* vert2bound_offset,
    int* vert2bound_cnt
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    int eid = boundaries[tid];

    // get edge
    uint64_t e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    // assign connectivity
    vert2bound[vert2bound_offset[e0] + atomicAdd(&vert2bound_cnt[e0], 1)] = tid;
    vert2bound[vert2bound_offset[e1] + atomicAdd(&vert2bound_cnt[e1], 1)] = tid;
}


void CuMesh::get_vertex_boundary_adjacency() {
    if (this->edges.is_empty()) {
        this->get_edges();
    } 
    if (this->boundaries.is_empty()) {
        this->get_boundary_info();
    }
    size_t V = this->vertices.size;
    size_t B = this->boundaries.size;

    // Early return if no boundaries
    if (B == 0) {
        this->vert2bound_cnt.resize(V + 1);
        this->vert2bound_cnt.zero();
        this->vert2bound_offset.resize(V + 1);
        this->vert2bound_offset.zero();
        return;
    }

    // get vertex adjacent boundary number
    this->vert2bound_cnt.resize(V + 1);
    this->vert2bound_cnt.zero();
    get_vertex_boundary_cnt_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr, this->boundaries.ptr, B, this->vert2bound_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());

    // allocate memory for vert2bound_offset
    this->vert2bound_offset.resize(V + 1);
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        this->vert2bound_cnt.ptr, this->vert2bound_offset.ptr,
        V + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->vert2bound_cnt.ptr, this->vert2bound_offset.ptr,
        V + 1
    ));

    // get vertex-boundary adjacency
    this->vert2bound.resize(2 * B);
    this->vert2bound_cnt.zero();
    get_vertex_boundary_adjacency_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr, this->boundaries.ptr, B,
        this->vert2bound.ptr,
        this->vert2bound_offset.ptr,
        this->vert2bound_cnt.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


static __global__ void get_vertex_is_manifold_kernel(
    const int* vert2edge,
    const int* vert2edge_offset,
    const int* edge2face_cnt,
    const int V,
    uint8_t* vert_is_manifold
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= V) return;

    // traverse all edges of the vertex
    int num_boundaries = 0;
    bool is_manifold = true;
    for (int i = vert2edge_offset[tid]; i < vert2edge_offset[tid+1]; i++) {
        int eid = vert2edge[i];
        // boundary edge
        if (edge2face_cnt[eid] == 1) {
            num_boundaries++;
            if (num_boundaries > 2) {
                is_manifold = false;
                break;
            }
        }
        // non-manifold edge
        else if (edge2face_cnt[eid] > 2) {
            is_manifold = false;
            break;
        }
    }

    vert_is_manifold[tid] = is_manifold ? 1 : 0;
}       


void CuMesh::get_vertex_is_manifold() {
    if (this->vert2edge.is_empty() || this->vert2edge_offset.is_empty()) {
        this->get_vertex_edge_adjacency();
    }
    if (this->edge2face_cnt.is_empty()) {
        this->get_edges();
    }
    size_t V = this->vertices.size;

    // get vertex is manifold
    this->vert_is_manifold.resize(V);
    get_vertex_is_manifold_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->vert2edge.ptr,
        this->vert2edge_offset.ptr,
        this->edge2face_cnt.ptr,
        V,
        this->vert_is_manifold.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


/**
 * Set manifold face adjacency
 * 
 * @param manifold_edge_idx: the buffer for manifold edge index, shape (M)
 * @param edge2face: the buffer for edge to face adjacency, shape (total_edge_face_cnt)
 * @param edge2face_offset: the buffer for edge to face adjacency offset, shape (E+1)
 * @param M: the number of manifold edges
 * @param manifold_face_adj: the buffer for manifold face adjacency, shape (M)
 */
static __global__ void set_manifold_face_adj_kernel(
    const int* manifold_edge_idx,
    const int* edge2face,
    const int* edge2face_offset,
    const int M,
    int2* manifold_face_adj
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get edge
    int edge_idx = manifold_edge_idx[tid];

    // get adjacent faces
    int start = edge2face_offset[edge_idx];
    int end = edge2face_offset[edge_idx+1];
    if (end - start != 2) return;   // if not a manifold edge
    int f0 = edge2face[start];
    int f1 = edge2face[start + 1];

    manifold_face_adj[tid] = {f0, f1};
}


struct is_manifold_edge {
    const int* edge2face_cnt;
    __host__ __device__
    bool operator()(const int& idx) const {
        return edge2face_cnt[idx] == 2;
    }
};


void CuMesh::get_manifold_face_adjacency() {
    if (this->edge2face.is_empty() || this->edge2face_offset.is_empty()) {
        this->get_edge_face_adjacency();
    }
    size_t E = this->edges.size;

    // Select manifold edges
    size_t temp_storage_bytes = 0;
    int *cu_num_manifold_edges, *cu_edge_idx, *cu_manifold_edge_idx;
    CUDA_CHECK(cudaMalloc(&cu_num_manifold_edges, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_edge_idx, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_manifold_edge_idx, E * sizeof(int)));
    arange_kernel<<<(E+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_edge_idx, E);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, temp_storage_bytes,
        cu_edge_idx, cu_manifold_edge_idx, cu_num_manifold_edges,
        E,
        is_manifold_edge{this->edge2face_cnt.ptr}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::If(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_edge_idx, cu_manifold_edge_idx, cu_num_manifold_edges,
        E,
        is_manifold_edge{this->edge2face_cnt.ptr}
    ));
    int manifold_edge_count;
    CUDA_CHECK(cudaMemcpy(&manifold_edge_count, cu_num_manifold_edges, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_num_manifold_edges));
    CUDA_CHECK(cudaFree(cu_edge_idx));

    // set manifold_face_adj
    this->manifold_face_adj.resize(manifold_edge_count);
    set_manifold_face_adj_kernel<<<(manifold_edge_count+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_manifold_edge_idx,
        this->edge2face.ptr,
        this->edge2face_offset.ptr,
        manifold_edge_count,
        this->manifold_face_adj.ptr
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_manifold_edge_idx));
}


static __global__ void set_manifold_bound_adj_kernel(
    const int* manifold_boundary_verts_idx,
    const int* vert2bound,
    const int* vert2bound_offset,
    const size_t MBV,
    int2* manifold_bound_adj
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= MBV) return;

    // get vertex
    int vert_idx = manifold_boundary_verts_idx[tid];

    // get adjacent boundaries
    int b0 = vert2bound[vert2bound_offset[vert_idx]];
    int b1 = vert2bound[vert2bound_offset[vert_idx] + 1];

    manifold_bound_adj[tid] = {b0, b1};
}


struct is_manifold_boundary_vertex {
    const uint8_t* vert_is_manifold;
    const uint8_t* vert_is_boundary;
    __host__ __device__
    bool operator()(const int& idx) const {
        return vert_is_manifold[idx] && vert_is_boundary[idx];
    }
};


void CuMesh::get_manifold_boundary_adjacency() {
    if (this->vert2bound.is_empty() || this->vert2bound_offset.is_empty()) {
        this->get_vertex_boundary_adjacency();
    }
    if (this->vert_is_manifold.is_empty()) {
        this->get_vertex_is_manifold();
    }
    size_t V = this->vertices.size;

    // Select manifold boundary vertices
    size_t temp_storage_bytes = 0;
    int *cu_num_manifold_boundary_verts, *cu_vert_idx, *cu_manifold_vert_idx;
    CUDA_CHECK(cudaMalloc(&cu_num_manifold_boundary_verts, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_vert_idx, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_manifold_vert_idx, V * sizeof(int)));
    arange_kernel<<<(V+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cu_vert_idx, V);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, temp_storage_bytes,
        cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,
        V,
        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr}
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::If(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_vert_idx, cu_manifold_vert_idx, cu_num_manifold_boundary_verts,
        V,
        is_manifold_boundary_vertex{this->vert_is_manifold.ptr, this->vert_is_boundary.ptr}
    ));
    int manifold_boundary_vert_count;
    CUDA_CHECK(cudaMemcpy(&manifold_boundary_vert_count, cu_num_manifold_boundary_verts, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_num_manifold_boundary_verts));
    CUDA_CHECK(cudaFree(cu_vert_idx));

    // Early return if no manifold boundary vertices
    if (manifold_boundary_vert_count == 0) {
        CUDA_CHECK(cudaFree(cu_manifold_vert_idx));
        return;
    }

    // set manifold_bound_adj
    this->manifold_bound_adj.resize(manifold_boundary_vert_count);
    set_manifold_bound_adj_kernel<<<(manifold_boundary_vert_count+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_manifold_vert_idx,
        this->vert2bound.ptr,
        this->vert2bound_offset.ptr,
        manifold_boundary_vert_count,
        this->manifold_bound_adj.ptr
    );
    CUDA_CHECK(cudaGetLastError());
}


void CuMesh::get_connected_components() {
    if (this->manifold_face_adj.is_empty()) {
        this->get_manifold_face_adjacency();
    }

    size_t M = this->manifold_face_adj.size;
    size_t F = this->faces.size;

    // Iterative Hook and Compress
    this->conn_comp_ids.resize(F);
    arange_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(this->conn_comp_ids.ptr, F);
    CUDA_CHECK(cudaGetLastError());
    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->manifold_face_adj.ptr,
            M,
            this->conn_comp_ids.ptr,
            cu_end_flag
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_kernel<<<(F+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->conn_comp_ids.ptr,
            F
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);
    CUDA_CHECK(cudaFree(cu_end_flag));

    // Compresses boundary components
    this->num_conn_comps = compress_ids(this->conn_comp_ids.ptr, F, this->cub_temp_storage);
}


void CuMesh::get_boundary_connected_components() {
    if (this->manifold_bound_adj.is_empty()) {
        this->get_manifold_boundary_adjacency();
    }
    size_t M = this->manifold_bound_adj.size;
    size_t B = this->boundaries.size;

    // Early return if no boundaries
    if (B == 0) {
        this->num_bound_conn_comps = 0;
        return;
    }

    // Iterative Hook and Compress
    this->bound_conn_comp_ids.resize(B);
    arange_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(this->bound_conn_comp_ids.ptr, B);
    CUDA_CHECK(cudaGetLastError());
    int* cu_end_flag; int h_end_flag;
    CUDA_CHECK(cudaMalloc(&cu_end_flag, sizeof(int)));
    do {
        h_end_flag = 1;
        CUDA_CHECK(cudaMemcpy(cu_end_flag, &h_end_flag, sizeof(int), cudaMemcpyHostToDevice));

        // Hook
        hook_edges_kernel<<<(M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->manifold_bound_adj.ptr,
            M,
            this->bound_conn_comp_ids.ptr,
            cu_end_flag
        );
        CUDA_CHECK(cudaGetLastError());

        // Compress
        compress_components_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
            this->bound_conn_comp_ids.ptr,
            B
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&h_end_flag, cu_end_flag, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_end_flag == 0);
    CUDA_CHECK(cudaFree(cu_end_flag));

    // Compresses boundary components
    this->num_bound_conn_comps = compress_ids(this->bound_conn_comp_ids.ptr, B, this->cub_temp_storage);
}


static __global__ void is_bound_conn_comp_loop_kernel(
    const uint64_t* edges,
    const int* boundaries,
    const int* bound_conn_comp_ids,
    const int* vert2bound,
    const int* vert2bound_offset,
    const int B,
    int* is_bound_conn_comp_loop
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B) return;

    // get edge
    int eid = boundaries[tid];
    uint64_t e = edges[eid];
    int e0 = int(e >> 32);
    int e1 = int(e & 0xFFFFFFFF);

    int self_comp_id = bound_conn_comp_ids[tid];

    // check if both vertices are connected to another boundary with the same connected component id
    int cnt = 0;
    for (int i = vert2bound_offset[e0]; i < vert2bound_offset[e0+1]; i++) {
        int b = vert2bound[i];
        if (b == tid) continue; // skip self
        int comp_id = bound_conn_comp_ids[b];
        if (comp_id == self_comp_id) cnt++;
    }
    if (cnt == 0) {
        is_bound_conn_comp_loop[self_comp_id] = 0;   // no loop
        return;
    }
    cnt = 0;
    for (int i = vert2bound_offset[e1]; i < vert2bound_offset[e1+1]; i++) {
        int b = vert2bound[i];
        if (b == tid) continue; // skip self
        int comp_id = bound_conn_comp_ids[b];
        if (comp_id == self_comp_id) cnt++;
    }
    if (cnt == 0) {
        is_bound_conn_comp_loop[self_comp_id] = 0;   // no loop
        return;
    }
}


void CuMesh::get_boundary_loops() {
    if (this->bound_conn_comp_ids.is_empty()) {
        this->get_boundary_connected_components();
    }

    size_t B = this->boundaries.size;

    // Early return if no boundaries or boundary components
    if (B == 0 || this->num_bound_conn_comps == 0) {
        this->num_bound_loops = 0;
        return;
    }

    // Check if boundary components are loops
    int* cu_is_bound_conn_comp_loop;
    CUDA_CHECK(cudaMalloc(&cu_is_bound_conn_comp_loop, this->num_bound_conn_comps * sizeof(int)));
    fill_kernel<<<(this->num_bound_conn_comps+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_is_bound_conn_comp_loop,
        this->num_bound_conn_comps,
        1
    );
    CUDA_CHECK(cudaGetLastError());
    is_bound_conn_comp_loop_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        this->edges.ptr,
        this->boundaries.ptr,
        this->bound_conn_comp_ids.ptr,
        this->vert2bound.ptr,
        this->vert2bound_offset.ptr,
        B,
        cu_is_bound_conn_comp_loop
    );
    CUDA_CHECK(cudaGetLastError());
    int* cu_num_bound_loops;
    CUDA_CHECK(cudaMalloc(&cu_num_bound_loops, sizeof(int)));
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(
        nullptr, temp_storage_bytes,
        cu_is_bound_conn_comp_loop,
        cu_num_bound_loops,
        this->num_bound_conn_comps
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_is_bound_conn_comp_loop,
        cu_num_bound_loops,
        this->num_bound_conn_comps
    ));
    CUDA_CHECK(cudaMemcpy(&this->num_bound_loops, cu_num_bound_loops, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_num_bound_loops));
    if (this->num_bound_loops == 0) {
        CUDA_CHECK(cudaFree(cu_is_bound_conn_comp_loop));
        return;
    }
    
    // Sort boundaries by connected component ids
    int *cu_bound_sorted, *cu_bound_conn_comp_ids_sorted;
    CUDA_CHECK(cudaMalloc(&cu_bound_sorted, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_bound_conn_comp_ids_sorted, B * sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        this->bound_conn_comp_ids.ptr, cu_bound_conn_comp_ids_sorted,
        this->boundaries.ptr, cu_bound_sorted,
        B
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->bound_conn_comp_ids.ptr, cu_bound_conn_comp_ids_sorted,
        this->boundaries.ptr, cu_bound_sorted,
        B
    ));

    // Select loops
    int* cu_bound_is_on_loop;
    CUDA_CHECK(cudaMalloc(&cu_bound_is_on_loop, B * sizeof(int)));
    index_kernel<<<(B+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        cu_is_bound_conn_comp_loop,
        cu_bound_conn_comp_ids_sorted,
        B,
        cu_bound_is_on_loop
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(cu_is_bound_conn_comp_loop));
    this->loop_boundaries.resize(B);
    int *cu_loop_bound_conn_comp_ids_sorted, *cu_num_bound_on_loop;
    CUDA_CHECK(cudaMalloc(&cu_loop_bound_conn_comp_ids_sorted, B * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_num_bound_on_loop, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_bound_sorted, cu_bound_is_on_loop, this->loop_boundaries.ptr, cu_num_bound_on_loop,
        B
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_bound_sorted, cu_bound_is_on_loop, this->loop_boundaries.ptr, cu_num_bound_on_loop,
        B
    ));
    int num_bound_on_loop;
    CUDA_CHECK(cudaMemcpy(&num_bound_on_loop, cu_num_bound_on_loop, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(cu_bound_sorted));
    this->loop_boundaries.resize(num_bound_on_loop);
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        nullptr, temp_storage_bytes,
        cu_bound_conn_comp_ids_sorted, cu_bound_is_on_loop, cu_loop_bound_conn_comp_ids_sorted, cu_num_bound_on_loop,
        B
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceSelect::Flagged(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_bound_conn_comp_ids_sorted, cu_bound_is_on_loop, cu_loop_bound_conn_comp_ids_sorted, cu_num_bound_on_loop,
        B
    ));
    CUDA_CHECK(cudaFree(cu_bound_conn_comp_ids_sorted));
    CUDA_CHECK(cudaFree(cu_bound_is_on_loop));
    CUDA_CHECK(cudaFree(cu_num_bound_on_loop));
    
    // RLE
    this->loop_boundaries_offset.resize(this->num_bound_loops + 1);
    this->loop_boundaries_offset.zero();
    int* cu_rle_unique_out, *cu_rle_num_runs;
    CUDA_CHECK(cudaMalloc(&cu_rle_unique_out, this->num_bound_loops * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cu_rle_num_runs, sizeof(int)));
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        nullptr, temp_storage_bytes,
        cu_loop_bound_conn_comp_ids_sorted,
        cu_rle_unique_out, this->loop_boundaries_offset.ptr, cu_rle_num_runs,
        num_bound_on_loop
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        cu_loop_bound_conn_comp_ids_sorted,
        cu_rle_unique_out, this->loop_boundaries_offset.ptr, cu_rle_num_runs,
        num_bound_on_loop
    ));
    CUDA_CHECK(cudaFree(cu_loop_bound_conn_comp_ids_sorted));
    CUDA_CHECK(cudaFree(cu_rle_unique_out));
    CUDA_CHECK(cudaFree(cu_rle_num_runs));

    // Scan loop boundaries offset
    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_bytes,
        this->loop_boundaries_offset.ptr,
        this->num_bound_loops + 1
    ));
    this->cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        this->cub_temp_storage.ptr, temp_storage_bytes,
        this->loop_boundaries_offset.ptr,
        this->num_bound_loops + 1
    ));
}


} // namespace cumesh
