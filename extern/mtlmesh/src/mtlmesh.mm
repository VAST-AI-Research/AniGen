
// MtlMesh implementation — pure Metal compute shaders + MetalPrimitives
// Zero PyTorch in the GPU path. All data lives in id<MTLBuffer> StorageModeShared.
#import "mtlmesh.h"
#import <torch/torch.h>
#include <unistd.h>

namespace mtlmesh {

// ========== Shorthand ==========
#define CTX   MetalContext::instance()
#define PRIMS MetalPrimitives::instance()

template<typename T> static inline T* PTR(id<MTLBuffer> buf) { return (T*)[buf contents]; }

// ========== Buffer helpers ==========

id<MTLBuffer> MtlMesh::alloc(size_t bytes) {
    return [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

id<MTLBuffer> MtlMesh::alloc_zero(size_t bytes) {
    auto buf = alloc(bytes);
    memset([buf contents], 0, bytes);
    return buf;
}

// faces: [F,3] int32 (12 bytes/row) → Metal packed_int3 (12 bytes/row)
// Zero-copy when page-aligned; backing tensor stored in init_faces_backing_.
id<MTLBuffer> MtlMesh::faces_from_tensor(const torch::Tensor& t) {
    int F = (int)t.size(0);
    size_t bytes = (size_t)F * 12;
    auto tc = t.cpu().contiguous().to(torch::kInt32);
    void* ptr = tc.data_ptr<int>();
    // Try zero-copy: requires page-aligned pointer and size
    size_t page = sysconf(_SC_PAGESIZE);
    if (((uintptr_t)ptr % page) == 0 && (bytes % page) == 0) {
        auto buf = [dev_ newBufferWithBytesNoCopy:ptr
                                           length:bytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        if (buf) {
            init_faces_backing_ = tc;  // prevent deallocation
            return buf;
        }
    }
    // Fallback: alloc + memcpy
    auto buf = alloc(bytes);
    memcpy([buf contents], ptr, bytes);
    return buf;
}

// Metal packed_int3 (12 bytes/row) → [F,3] int32 — direct memcpy
torch::Tensor MtlMesh::faces_to_tensor(id<MTLBuffer> buf, int F) {
    auto t = torch::empty({F, 3}, torch::kInt32);
    memcpy(t.data_ptr<int>(), [buf contents], F * 12);
    return t;
}

// [V,3] float32 → V*12 bytes (packed_float3 is 12 bytes, matches directly)
// Zero-copy when page-aligned; backing tensor stored in init_verts_backing_.
id<MTLBuffer> MtlMesh::verts_from_tensor(const torch::Tensor& t) {
    int N = (int)t.size(0);
    size_t bytes = (size_t)N * 12;
    auto tc = t.cpu().contiguous().to(torch::kFloat32);
    void* ptr = tc.data_ptr<float>();
    // Try zero-copy: requires page-aligned pointer and size
    size_t page = sysconf(_SC_PAGESIZE);
    if (((uintptr_t)ptr % page) == 0 && (bytes % page) == 0) {
        auto buf = [dev_ newBufferWithBytesNoCopy:ptr
                                           length:bytes
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        if (buf) {
            init_verts_backing_ = tc;  // prevent deallocation
            return buf;
        }
    }
    // Fallback: alloc + memcpy
    auto buf = alloc(bytes);
    memcpy([buf contents], ptr, bytes);
    return buf;
}

torch::Tensor MtlMesh::verts_to_tensor(id<MTLBuffer> buf, int N) {
    auto t = torch::empty({N, 3}, torch::kFloat32);
    memcpy(t.data_ptr<float>(), [buf contents], N * 12);
    return t;
}

torch::Tensor MtlMesh::int2_to_tensor(id<MTLBuffer> buf, int N) {
    auto t = torch::empty({N, 2}, torch::kInt32);
    memcpy(t.data_ptr<int>(), [buf contents], N * 8);
    return t;
}

// NOTE: Currently unused. If used with zero-copy, the caller must retain
// the backing tensor externally — this function does not store it.
id<MTLBuffer> MtlMesh::ints_from_tensor(const torch::Tensor& t) {
    auto tc = t.cpu().contiguous().to(torch::kInt32);
    int N = (int)tc.numel();
    auto buf = alloc(N * 4);
    memcpy([buf contents], tc.data_ptr<int>(), N * 4);
    return buf;
}

torch::Tensor MtlMesh::ints_to_tensor(id<MTLBuffer> buf, int N) {
    auto t = torch::empty({N}, torch::kInt32);
    memcpy(t.data_ptr<int>(), [buf contents], N * 4);
    return t;
}

// ========== Constructor / Destructor ==========

MtlMesh::MtlMesh() : dev_(CTX.device()) {}
MtlMesh::~MtlMesh() {}

int MtlMesh::num_vertices() const { return num_verts; }
int MtlMesh::num_faces() const { return num_fcs; }
int MtlMesh::num_edges() const { return num_edges_; }
int MtlMesh::num_boundaries() const { return num_boundaries_; }
int MtlMesh::num_conneted_components() const { return num_conn_comps; }
int MtlMesh::num_boundary_conneted_components() const { return num_bound_conn_comps; }
int MtlMesh::num_boundary_loops() const { return num_bound_loops; }

void MtlMesh::clear_cache() {
    face_areas = face_normals = vertex_normals = nil;
    edges = edge2face_cnt = edge2face_offset = edge2face = face2edge = nil;
    boundaries = vert_is_boundary = vert_is_manifold = nil;
    vert2face = vert2face_cnt = vert2face_offset = nil;
    vert2edge = vert2edge_cnt = vert2edge_offset = nil;
    vert2bound = vert2bound_cnt = vert2bound_offset = nil;
    manifold_face_adj = manifold_bound_adj = nil;
    conn_comp_ids = bound_conn_comp_ids = nil;
    loop_boundaries = loop_boundaries_offset = nil;
    atlas_chart_ids = atlas_chart_vertex_map = atlas_chart_faces = nil;
    atlas_chart_faces_offset = atlas_chart_vertex_offset = nil;
    num_edges_ = num_boundaries_ = 0;
    num_manifold_edges_ = num_manifold_bound_verts_ = 0;
    num_conn_comps = num_bound_conn_comps = num_bound_loops = 0;
    num_loop_boundaries_ = 0;
    atlas_num_charts = 0;
}

// ========== I/O ==========

void MtlMesh::init(const torch::Tensor& verts, const torch::Tensor& fcs) {
    num_verts = (int)verts.size(0);
    num_fcs = (int)fcs.size(0);
    vertices = verts_from_tensor(verts);
    faces = faces_from_tensor(fcs);
    clear_cache();
}

std::tuple<torch::Tensor, torch::Tensor> MtlMesh::read() {
    return {verts_to_tensor(vertices, num_verts), faces_to_tensor(faces, num_fcs)};
}

torch::Tensor MtlMesh::read_face_areas() {
    TORCH_CHECK(face_areas != nil, "Face areas not computed");
    auto t = torch::empty({num_fcs}, torch::kFloat32);
    memcpy(t.data_ptr<float>(), [face_areas contents], num_fcs * 4);
    return t;
}

torch::Tensor MtlMesh::read_face_normals() {
    TORCH_CHECK(face_normals != nil, "Face normals not computed");
    return verts_to_tensor(face_normals, num_fcs);
}

torch::Tensor MtlMesh::read_vertex_normals() {
    TORCH_CHECK(vertex_normals != nil, "Vertex normals not computed");
    return verts_to_tensor(vertex_normals, num_verts);
}

torch::Tensor MtlMesh::read_edges() {
    TORCH_CHECK(edges != nil, "Edges not computed");
    int E = num_edges_;
    auto t = torch::empty({E, 2}, torch::kInt32);
    auto* src = PTR<uint64_t>(edges);
    auto* dst = t.data_ptr<int>();
    for (int i = 0; i < E; i++) {
        dst[i*2+0] = (int)(src[i] >> 32);
        dst[i*2+1] = (int)(src[i] & 0xFFFFFFFF);
    }
    return t;
}

torch::Tensor MtlMesh::read_boundaries() {
    if (boundaries == nil) return torch::empty({0}, torch::kInt32);
    return ints_to_tensor(boundaries, num_boundaries_);
}

torch::Tensor MtlMesh::read_manifold_face_adjacency() {
    if (manifold_face_adj == nil || num_manifold_edges_ == 0)
        return torch::empty({0, 2}, torch::kInt32);
    return int2_to_tensor(manifold_face_adj, num_manifold_edges_);
}

torch::Tensor MtlMesh::read_manifold_boundary_adjacency() {
    if (manifold_bound_adj == nil || num_manifold_bound_verts_ == 0)
        return torch::empty({0, 2}, torch::kInt32);
    return int2_to_tensor(manifold_bound_adj, num_manifold_bound_verts_);
}

std::tuple<int, torch::Tensor> MtlMesh::read_connected_components() {
    if (conn_comp_ids == nil) return {0, torch::empty({0}, torch::kInt32)};
    return {num_conn_comps, ints_to_tensor(conn_comp_ids, num_fcs)};
}

std::tuple<int, torch::Tensor> MtlMesh::read_boundary_connected_components() {
    if (bound_conn_comp_ids == nil) return {0, torch::empty({0}, torch::kInt32)};
    return {num_bound_conn_comps, ints_to_tensor(bound_conn_comp_ids, num_boundaries_)};
}

std::tuple<int, torch::Tensor, torch::Tensor> MtlMesh::read_boundary_loops() {
    if (loop_boundaries == nil)
        return {0, torch::empty({0}, torch::kInt32), torch::empty({0}, torch::kInt32)};
    return {num_bound_loops,
            ints_to_tensor(loop_boundaries, num_loop_boundaries_),
            ints_to_tensor(loop_boundaries_offset, num_bound_loops + 1)};
}

std::unordered_map<std::string, torch::Tensor> MtlMesh::read_all_cache() {
    std::unordered_map<std::string, torch::Tensor> cache;
    if (vertices) cache["vertices"] = verts_to_tensor(vertices, num_verts);
    if (faces) cache["faces"] = faces_to_tensor(faces, num_fcs);
    return cache;
}

// ========== Geometry ==========

void MtlMesh::compute_face_areas() {
    int F = num_fcs;
    face_areas = alloc(F * 4);
    CTX.dispatch("compute_face_areas_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBytes:&F length:sizeof(int) atIndex:2];
        [enc setBuffer:face_areas offset:0 atIndex:3];
    }, F);
}

void MtlMesh::compute_face_normals() {
    int F = num_fcs;
    face_normals = alloc(F * 12);
    CTX.dispatch("compute_face_normals_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBytes:&F length:sizeof(int) atIndex:2];
        [enc setBuffer:face_normals offset:0 atIndex:3];
    }, F);
}

void MtlMesh::compute_vertex_normals() {
    if (!vert2face) get_vertex_face_adjacency();
    int V = num_verts;
    vertex_normals = alloc(V * 12);
    CTX.dispatch("compute_vertex_normals_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:vert2face offset:0 atIndex:2];
        [enc setBuffer:vert2face_offset offset:0 atIndex:3];
        [enc setBytes:&V length:sizeof(int) atIndex:4];
        [enc setBuffer:vertex_normals offset:0 atIndex:5];
    }, V);
}

// ========== Connectivity ==========

void MtlMesh::get_vertex_face_adjacency() {
    int V = num_verts, F = num_fcs;

    // Step 1: Count neighbor faces per vertex (atomics)
    vert2face_cnt = alloc_zero((V + 1) * 4);
    CTX.dispatch("get_neighbor_face_cnt_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:faces offset:0 atIndex:0];
        [enc setBytes:&F length:sizeof(int) atIndex:1];
        [enc setBuffer:vert2face_cnt offset:0 atIndex:2];
    }, F);

    // Step 2: Exclusive prefix sum → offsets
    vert2face_offset = alloc((V + 1) * 4);
    memcpy([vert2face_offset contents], [vert2face_cnt contents], (V + 1) * 4);
    PRIMS.exclusive_sum(vert2face_offset, V + 1);

    // Step 3: Read total
    int total = PTR<int>(vert2face_offset)[V];

    // Step 4: Fill adjacency (zero cnt for reuse as atomic counter)
    vert2face = alloc(total * 4);
    memset([vert2face_cnt contents], 0, (V + 1) * 4);
    CTX.dispatch("fill_neighbor_face_ids_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:faces offset:0 atIndex:0];
        [enc setBytes:&F length:sizeof(int) atIndex:1];
        [enc setBuffer:vert2face offset:0 atIndex:2];
        [enc setBuffer:vert2face_offset offset:0 atIndex:3];
        [enc setBuffer:vert2face_cnt offset:0 atIndex:4];
    }, F);
}

void MtlMesh::get_edges() {
    int F = num_fcs;

    // Step 1: Expand 3 edges per face
    auto all_edges = alloc(3 * F * 8);
    CTX.dispatch("expand_edges_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:faces offset:0 atIndex:0];
        [enc setBytes:&F length:sizeof(int) atIndex:1];
        [enc setBuffer:all_edges offset:0 atIndex:2];
    }, F);

    // Step 2: Sort edges (need key-value sort; values are dummy)
    auto dummy_vals = alloc(3 * F * 4);
    PRIMS.sort_pairs_uint64(all_edges, dummy_vals, 3 * F);

    // Step 3: Run-length encode for uint64 keys (CPU — RLE only supports int keys)
    auto unique_buf = alloc(3 * F * 8);
    auto counts_buf = alloc(3 * F * 4);
    auto* sk = PTR<uint64_t>(all_edges);
    auto* uk = PTR<uint64_t>(unique_buf);
    auto* cnt = PTR<int>(counts_buf);
    int E = 0;
    for (int i = 0; i < 3 * F; ) {
        uint64_t key = sk[i];
        int run = 1;
        while (i + run < 3 * F && sk[i + run] == key) run++;
        uk[E] = key;
        cnt[E] = run;
        E++;
        i += run;
    }

    edges = unique_buf;
    edge2face_cnt = counts_buf;
    num_edges_ = E;
}

void MtlMesh::get_edge_face_adjacency() {
    if (!edges) get_edges();
    if (!vert2face) get_vertex_face_adjacency();

    int E = num_edges_, F = num_fcs;

    // Build offset from counts via exclusive prefix sum
    edge2face_offset = alloc((E + 1) * 4);
    memcpy([edge2face_offset contents], [edge2face_cnt contents], E * 4);
    PTR<int>(edge2face_offset)[E] = 0;
    PRIMS.exclusive_sum(edge2face_offset, E + 1);

    int total = PTR<int>(edge2face_offset)[E];
    edge2face = alloc_zero(total * 4);
    face2edge = alloc_zero(F * 12);  // packed_int3 stride

    CTX.dispatch("get_edge_face_adjacency_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:faces offset:0 atIndex:0];
        [enc setBuffer:edges offset:0 atIndex:1];
        [enc setBuffer:edge2face_cnt offset:0 atIndex:2];
        [enc setBuffer:vert2face offset:0 atIndex:3];
        [enc setBuffer:vert2face_offset offset:0 atIndex:4];
        [enc setBuffer:edge2face_offset offset:0 atIndex:5];
        [enc setBytes:&E length:sizeof(int) atIndex:6];
        [enc setBuffer:edge2face offset:0 atIndex:7];
        [enc setBuffer:face2edge offset:0 atIndex:8];
    }, E);
}

void MtlMesh::get_vertex_edge_adjacency() {
    if (!edges) get_edges();
    int V = num_verts, E = num_edges_;

    // Step 1: Count edges per vertex
    vert2edge_cnt = alloc_zero((V + 1) * 4);
    CTX.dispatch("get_vertex_edge_cnt_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBytes:&E length:sizeof(int) atIndex:1];
        [enc setBuffer:vert2edge_cnt offset:0 atIndex:2];
    }, E);

    // Step 2: Exclusive prefix sum
    vert2edge_offset = alloc((V + 1) * 4);
    memcpy([vert2edge_offset contents], [vert2edge_cnt contents], (V + 1) * 4);
    PRIMS.exclusive_sum(vert2edge_offset, V + 1);
    int total = PTR<int>(vert2edge_offset)[V];

    // Step 3: Fill (zero cnt for reuse)
    vert2edge = alloc(total * 4);
    memset([vert2edge_cnt contents], 0, (V + 1) * 4);
    CTX.dispatch("get_vertex_edge_adjacency_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBytes:&E length:sizeof(int) atIndex:1];
        [enc setBuffer:vert2edge offset:0 atIndex:2];
        [enc setBuffer:vert2edge_offset offset:0 atIndex:3];
        [enc setBuffer:vert2edge_cnt offset:0 atIndex:4];
    }, E);
}

void MtlMesh::get_boundary_info() {
    if (!edges) get_edges();
    int V = num_verts, E = num_edges_;

    // Select edges where edge2face_cnt == 1 (CPU, zero-copy on Apple Silicon)
    auto* cnt = PTR<int>(edge2face_cnt);
    int B = 0;
    for (int i = 0; i < E; i++) if (cnt[i] == 1) B++;

    boundaries = alloc(B * 4);
    auto* bptr = PTR<int>(boundaries);
    int idx = 0;
    for (int i = 0; i < E; i++) if (cnt[i] == 1) bptr[idx++] = i;
    num_boundaries_ = B;

    // Set boundary vertices
    vert_is_boundary = alloc_zero(V);
    if (B > 0) {
        CTX.dispatch("set_boundary_vertex_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:edges offset:0 atIndex:0];
            [enc setBuffer:boundaries offset:0 atIndex:1];
            [enc setBuffer:edge2face_cnt offset:0 atIndex:2];
            [enc setBytes:&B length:sizeof(int) atIndex:3];
            [enc setBuffer:vert_is_boundary offset:0 atIndex:4];
        }, B);
    }
}

void MtlMesh::get_vertex_boundary_adjacency() {
    if (!edges) get_edges();
    if (!boundaries) get_boundary_info();
    int V = num_verts, B = num_boundaries_;

    if (B == 0) {
        vert2bound_cnt = alloc_zero((V + 1) * 4);
        vert2bound_offset = alloc_zero((V + 1) * 4);
        return;
    }

    // Count
    vert2bound_cnt = alloc_zero((V + 1) * 4);
    CTX.dispatch("get_vertex_boundary_cnt_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBuffer:boundaries offset:0 atIndex:1];
        [enc setBytes:&B length:sizeof(int) atIndex:2];
        [enc setBuffer:vert2bound_cnt offset:0 atIndex:3];
    }, B);

    // Exclusive prefix sum
    vert2bound_offset = alloc((V + 1) * 4);
    memcpy([vert2bound_offset contents], [vert2bound_cnt contents], (V + 1) * 4);
    PRIMS.exclusive_sum(vert2bound_offset, V + 1);
    int total = PTR<int>(vert2bound_offset)[V];

    // Fill
    vert2bound = alloc(total * 4);
    memset([vert2bound_cnt contents], 0, (V + 1) * 4);
    CTX.dispatch("get_vertex_boundary_adjacency_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBuffer:boundaries offset:0 atIndex:1];
        [enc setBytes:&B length:sizeof(int) atIndex:2];
        [enc setBuffer:vert2bound offset:0 atIndex:3];
        [enc setBuffer:vert2bound_offset offset:0 atIndex:4];
        [enc setBuffer:vert2bound_cnt offset:0 atIndex:5];
    }, B);
}

void MtlMesh::get_vertex_is_manifold() {
    if (!vert2edge) get_vertex_edge_adjacency();
    if (!edge2face_cnt) get_edges();
    int V = num_verts;

    vert_is_manifold = alloc(V);
    memset([vert_is_manifold contents], 1, V);

    CTX.dispatch("get_vertex_is_manifold_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vert2edge offset:0 atIndex:0];
        [enc setBuffer:vert2edge_offset offset:0 atIndex:1];
        [enc setBuffer:edge2face_cnt offset:0 atIndex:2];
        [enc setBytes:&V length:sizeof(int) atIndex:3];
        [enc setBuffer:vert_is_manifold offset:0 atIndex:4];
    }, V);
}

void MtlMesh::get_manifold_face_adjacency() {
    if (!edge2face) get_edge_face_adjacency();
    int E = num_edges_;

    // Select manifold edge indices (edge2face_cnt == 2) on CPU
    auto* cnt = PTR<int>(edge2face_cnt);
    int M = 0;
    for (int i = 0; i < E; i++) if (cnt[i] == 2) M++;

    auto manifold_edge_idx = alloc(M * 4);
    auto* mptr = PTR<int>(manifold_edge_idx);
    int idx = 0;
    for (int i = 0; i < E; i++) if (cnt[i] == 2) mptr[idx++] = i;
    num_manifold_edges_ = M;

    manifold_face_adj = alloc(M * 8);
    if (M > 0) {
        CTX.dispatch("set_manifold_face_adj_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:manifold_edge_idx offset:0 atIndex:0];
            [enc setBuffer:edge2face offset:0 atIndex:1];
            [enc setBuffer:edge2face_offset offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(int) atIndex:3];
            [enc setBuffer:manifold_face_adj offset:0 atIndex:4];
        }, M);
    }
}

void MtlMesh::get_manifold_boundary_adjacency() {
    if (!vert2bound) get_vertex_boundary_adjacency();
    if (!vert_is_manifold) get_vertex_is_manifold();
    if (!vert_is_boundary) get_boundary_info();
    int V = num_verts;

    // Select manifold boundary vertices on CPU
    auto* mani = PTR<uint8_t>(vert_is_manifold);
    auto* bnd  = PTR<uint8_t>(vert_is_boundary);
    int MBV = 0;
    for (int i = 0; i < V; i++) if (mani[i] && bnd[i]) MBV++;

    auto mb_verts = alloc(MBV * 4);
    auto* mbptr = PTR<int>(mb_verts);
    int idx = 0;
    for (int i = 0; i < V; i++) if (mani[i] && bnd[i]) mbptr[idx++] = i;
    num_manifold_bound_verts_ = MBV;

    manifold_bound_adj = alloc(MBV * 8);
    if (MBV > 0) {
        CTX.dispatch("set_manifold_bound_adj_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:mb_verts offset:0 atIndex:0];
            [enc setBuffer:vert2bound offset:0 atIndex:1];
            [enc setBuffer:vert2bound_offset offset:0 atIndex:2];
            [enc setBytes:&MBV length:sizeof(int) atIndex:3];
            [enc setBuffer:manifold_bound_adj offset:0 atIndex:4];
        }, MBV);
    }
}

// ========== Union-Find ==========

void MtlMesh::hook_and_compress(id<MTLBuffer> adj, int M, id<MTLBuffer> comp_ids, int N) {
    auto end_flag = alloc(4);

    while (true) {
        PTR<int>(end_flag)[0] = 1;

        CTX.dispatch("hook_edges_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:adj offset:0 atIndex:0];
            [enc setBytes:&M length:sizeof(int) atIndex:1];
            [enc setBuffer:comp_ids offset:0 atIndex:2];
            [enc setBuffer:end_flag offset:0 atIndex:3];
        }, M);

        CTX.dispatch("compress_components_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:comp_ids offset:0 atIndex:0];
            [enc setBytes:&N length:sizeof(int) atIndex:1];
        }, N);

        if (PTR<int>(end_flag)[0] == 1) break;
    }
}

int MtlMesh::compress_ids(id<MTLBuffer> ids, int N) {
    if (N == 0) return 0;

    // Sort with index tracking
    auto idx_buf = alloc(N * 4);
    auto* idx = PTR<int>(idx_buf);
    for (int i = 0; i < N; i++) idx[i] = i;

    auto ids_copy = alloc(N * 4);
    memcpy([ids_copy contents], [ids contents], N * 4);
    PRIMS.sort_pairs_int(ids_copy, idx_buf, N);

    // Mark transitions (CPU, zero-copy)
    auto* sorted = PTR<int>(ids_copy);
    auto trans_buf = alloc(N * 4);
    auto* trans = PTR<int>(trans_buf);
    trans[0] = 0;
    for (int i = 1; i < N; i++)
        trans[i] = (sorted[i] != sorted[i-1]) ? 1 : 0;

    // Inclusive prefix sum → compressed IDs in sorted order
    PRIMS.inclusive_sum(trans_buf, N);

    // Scatter back to original order (CPU)
    auto* result = PTR<int>(ids);
    for (int i = 0; i < N; i++)
        result[idx[i]] = trans[i];

    return trans[N-1] + 1;
}

// ========== Connected Components ==========

void MtlMesh::get_connected_components() {
    if (!manifold_face_adj) get_manifold_face_adjacency();
    int F = num_fcs, M = num_manifold_edges_;

    // Initialize comp_ids = [0, 1, 2, ..., F-1]
    conn_comp_ids = alloc(F * 4);
    auto* cids = PTR<int>(conn_comp_ids);
    for (int i = 0; i < F; i++) cids[i] = i;

    hook_and_compress(manifold_face_adj, M, conn_comp_ids, F);
    num_conn_comps = compress_ids(conn_comp_ids, F);
}

void MtlMesh::get_boundary_connected_components() {
    if (!manifold_bound_adj) get_manifold_boundary_adjacency();
    int B = num_boundaries_, MBV = num_manifold_bound_verts_;

    if (B == 0) { num_bound_conn_comps = 0; return; }

    bound_conn_comp_ids = alloc(B * 4);
    auto* cids = PTR<int>(bound_conn_comp_ids);
    for (int i = 0; i < B; i++) cids[i] = i;

    hook_and_compress(manifold_bound_adj, MBV, bound_conn_comp_ids, B);
    num_bound_conn_comps = compress_ids(bound_conn_comp_ids, B);
}

void MtlMesh::get_boundary_loops() {
    if (!bound_conn_comp_ids) get_boundary_connected_components();
    int B = num_boundaries_;

    if (B == 0 || num_bound_conn_comps == 0) { num_bound_loops = 0; return; }

    // Check which boundary components are loops
    auto is_loop = alloc(num_bound_conn_comps * 4);
    auto* lp = PTR<int>(is_loop);
    for (int i = 0; i < num_bound_conn_comps; i++) lp[i] = 1;

    CTX.dispatch("is_bound_conn_comp_loop_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBuffer:boundaries offset:0 atIndex:1];
        [enc setBuffer:bound_conn_comp_ids offset:0 atIndex:2];
        [enc setBuffer:vert2bound offset:0 atIndex:3];
        [enc setBuffer:vert2bound_offset offset:0 atIndex:4];
        [enc setBytes:&B length:sizeof(int) atIndex:5];
        [enc setBuffer:is_loop offset:0 atIndex:6];
    }, B);

    // Count loops and gather loop boundaries (CPU)
    num_bound_loops = 0;
    for (int i = 0; i < num_bound_conn_comps; i++) if (lp[i]) num_bound_loops++;
    if (num_bound_loops == 0) return;

    // Sort boundaries by component ID, then filter loops
    auto sort_keys = alloc(B * 4);
    auto sort_vals = alloc(B * 4);
    memcpy([sort_keys contents], [bound_conn_comp_ids contents], B * 4);
    auto* sv = PTR<int>(sort_vals);
    auto* bp = PTR<int>(boundaries);
    for (int i = 0; i < B; i++) sv[i] = bp[i];
    PRIMS.sort_pairs_int(sort_keys, sort_vals, B);

    // Filter: keep only boundaries whose component is a loop
    auto* sk = PTR<int>(sort_keys);
    int total_loop_bounds = 0;
    for (int i = 0; i < B; i++) if (lp[sk[i]]) total_loop_bounds++;

    loop_boundaries = alloc(total_loop_bounds * 4);
    auto loop_comp_ids = alloc(total_loop_bounds * 4);
    auto* lb = PTR<int>(loop_boundaries);
    auto* lc = PTR<int>(loop_comp_ids);
    int idx = 0;
    for (int i = 0; i < B; i++) {
        if (lp[sk[i]]) {
            lb[idx] = PTR<int>(sort_vals)[i];
            lc[idx] = sk[i];
            idx++;
        }
    }
    num_loop_boundaries_ = total_loop_bounds;

    // RLE to get offsets per loop
    auto unique_buf = alloc(total_loop_bounds * 4);
    auto counts_buf = alloc(total_loop_bounds * 4);
    int nloops = PRIMS.run_length_encode(loop_comp_ids, unique_buf, counts_buf, total_loop_bounds);

    // Build offset array
    loop_boundaries_offset = alloc((nloops + 1) * 4);
    memcpy([loop_boundaries_offset contents], [counts_buf contents], nloops * 4);
    PTR<int>(loop_boundaries_offset)[nloops] = 0;
    PRIMS.exclusive_sum(loop_boundaries_offset, nloops + 1);

    num_bound_loops = nloops;
}

// ========== Cleanup ==========

void MtlMesh::remove_faces(torch::Tensor& face_mask) {
    // face_mask is bool tensor; keep faces where mask is true
    auto mask_cpu = face_mask.cpu().to(torch::kBool).contiguous();
    TORCH_CHECK((int)mask_cpu.numel() == num_fcs,
                "remove_faces: face_mask has ", mask_cpu.numel(),
                " entries but mesh has ", num_fcs, " faces");
    auto* m = mask_cpu.data_ptr<bool>();
    int F = num_fcs;
    int V = num_verts;

    int new_F = 0;
    for (int i = 0; i < F; i++) if (m[i]) new_F++;

    auto new_faces = alloc((size_t)new_F * 12);
    auto* src = PTR<int>(faces);
    auto* dst = PTR<int>(new_faces);
    int idx = 0;
    for (int i = 0; i < F; i++) {
        if (m[i]) {
            // Validate face vertex indices before copy — upstream producers
            // have been observed to emit out-of-range indices on pathological
            // meshes, which previously segfaulted remove_unreferenced_vertices.
            int a = src[i*3+0], b = src[i*3+1], c = src[i*3+2];
            TORCH_CHECK(a >= 0 && a < V && b >= 0 && b < V && c >= 0 && c < V,
                        "remove_faces: face ", i, " has vertex index out of range (",
                        a, ", ", b, ", ", c, ") for V=", V);
            dst[idx*3+0] = a;
            dst[idx*3+1] = b;
            dst[idx*3+2] = c;
            idx++;
        }
    }

    faces = new_faces;
    num_fcs = new_F;
    remove_unreferenced_vertices();
}

void MtlMesh::remove_unreferenced_vertices() {
    int V = num_verts, F = num_fcs;

    // Mark referenced vertices (CPU)
    std::vector<uint8_t> ref(V, 0);
    auto* fp = PTR<int>(faces);
    for (int i = 0; i < F; i++) {
        int a = fp[i*3+0], b = fp[i*3+1], c = fp[i*3+2];
        TORCH_CHECK(a >= 0 && a < V && b >= 0 && b < V && c >= 0 && c < V,
                    "remove_unreferenced_vertices: face ", i,
                    " has vertex index out of range (", a, ", ", b, ", ", c,
                    ") for V=", V);
        ref[a] = 1;
        ref[b] = 1;
        ref[c] = 1;
    }

    // Build remap
    std::vector<int> remap(V, -1);
    int new_V = 0;
    for (int i = 0; i < V; i++) {
        if (ref[i]) { remap[i] = new_V++; }
    }

    // Compact vertices
    auto new_verts = alloc(new_V * 12);
    auto* vsrc = PTR<float>(vertices);
    auto* vdst = PTR<float>(new_verts);
    for (int i = 0; i < V; i++) {
        if (ref[i]) {
            int j = remap[i];
            vdst[j*3+0] = vsrc[i*3+0];
            vdst[j*3+1] = vsrc[i*3+1];
            vdst[j*3+2] = vsrc[i*3+2];
        }
    }

    // Remap face indices
    for (int i = 0; i < F; i++) {
        fp[i*3+0] = remap[fp[i*3+0]];
        fp[i*3+1] = remap[fp[i*3+1]];
        fp[i*3+2] = remap[fp[i*3+2]];
    }

    vertices = new_verts;
    // faces buffer is reused (indices remapped in-place), but if it was
    // zero-copy the underlying tensor data was mutated directly — safe.
    num_verts = new_V;
    clear_cache();
}

void MtlMesh::remove_duplicate_faces() {
    int F = num_fcs;
    auto* fp = PTR<int>(faces);

    // Sort vertex indices within each face (CPU)
    struct SortedFace { int v[3]; int orig; };
    std::vector<SortedFace> sf(F);
    for (int i = 0; i < F; i++) {
        int a = fp[i*3+0], b = fp[i*3+1], c = fp[i*3+2];
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        sf[i] = {{a, b, c}, i};
    }

    // Sort by sorted vertices
    std::sort(sf.begin(), sf.end(), [](const SortedFace& a, const SortedFace& b) {
        if (a.v[0] != b.v[0]) return a.v[0] < b.v[0];
        if (a.v[1] != b.v[1]) return a.v[1] < b.v[1];
        return a.v[2] < b.v[2];
    });

    // Mark first in each group
    std::vector<bool> keep(F, false);
    keep[sf[0].orig] = true;
    for (int i = 1; i < F; i++) {
        if (sf[i].v[0] != sf[i-1].v[0] || sf[i].v[1] != sf[i-1].v[1] || sf[i].v[2] != sf[i-1].v[2])
            keep[sf[i].orig] = true;
    }

    // Compact
    int new_F = 0;
    for (int i = 0; i < F; i++) if (keep[i]) new_F++;
    auto new_faces = alloc(new_F * 12);
    auto* dst = PTR<int>(new_faces);
    int idx = 0;
    for (int i = 0; i < F; i++) {
        if (keep[i]) {
            dst[idx*3+0] = fp[i*3+0];
            dst[idx*3+1] = fp[i*3+1];
            dst[idx*3+2] = fp[i*3+2];
            idx++;
        }
    }

    faces = new_faces;
    num_fcs = new_F;
    remove_unreferenced_vertices();
}

void MtlMesh::remove_degenerate_faces(float abs_thresh, float rel_thresh) {
    int F = num_fcs;

    // Use Metal kernel to mark faces
    auto mask = alloc(F);
    memset([mask contents], 1, F);

    CTX.dispatch("mark_degenerate_faces_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBytes:&abs_thresh length:sizeof(float) atIndex:2];
        [enc setBytes:&rel_thresh length:sizeof(float) atIndex:3];
        [enc setBytes:&F length:sizeof(int) atIndex:4];
        [enc setBuffer:mask offset:0 atIndex:5];
    }, F);

    // Compact on CPU
    auto* mp = PTR<uint8_t>(mask);
    auto* fp = PTR<int>(faces);
    int new_F = 0;
    for (int i = 0; i < F; i++) if (mp[i]) new_F++;

    auto new_faces = alloc(new_F * 12);
    auto* dst = PTR<int>(new_faces);
    int idx = 0;
    for (int i = 0; i < F; i++) {
        if (mp[i]) {
            dst[idx*3+0] = fp[i*3+0];
            dst[idx*3+1] = fp[i*3+1];
            dst[idx*3+2] = fp[i*3+2];
            idx++;
        }
    }
    faces = new_faces;
    num_fcs = new_F;
    remove_unreferenced_vertices();
}

void MtlMesh::remove_non_manifold_faces() {
    if (!edge2face) get_edge_face_adjacency();
    int F = num_fcs, E = num_edges_;
    if (F == 0 || E == 0) return;

    // Mark non-manifold faces for removal via Metal kernel
    auto keep_mask = alloc(F);
    memset([keep_mask contents], 1, F);

    CTX.dispatch("mark_non_manifold_faces_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edge2face offset:0 atIndex:0];
        [enc setBuffer:edge2face_offset offset:0 atIndex:1];
        [enc setBuffer:edge2face_cnt offset:0 atIndex:2];
        [enc setBytes:&E length:sizeof(int) atIndex:3];
        [enc setBuffer:keep_mask offset:0 atIndex:4];
    }, E);

    // Compact on CPU
    auto* mp = PTR<uint8_t>(keep_mask);
    auto* fp = PTR<int>(faces);
    int new_F = 0;
    for (int i = 0; i < F; i++) if (mp[i]) new_F++;

    auto new_faces = alloc(new_F * 12);
    auto* dst = PTR<int>(new_faces);
    int idx = 0;
    for (int i = 0; i < F; i++) {
        if (mp[i]) {
            dst[idx*3+0] = fp[i*3+0];
            dst[idx*3+1] = fp[i*3+1];
            dst[idx*3+2] = fp[i*3+2];
            idx++;
        }
    }
    faces = new_faces;
    num_fcs = new_F;
    remove_unreferenced_vertices();
    clear_cache();
}

void MtlMesh::remove_small_connected_components(float min_area) {
    if (!conn_comp_ids) get_connected_components();
    if (!face_areas) compute_face_areas();
    int F = num_fcs;
    if (F == 0) return;

    // Accumulate area per component (CPU)
    std::vector<float> comp_areas(num_conn_comps, 0.0f);
    auto* fa = PTR<float>(face_areas);
    auto* cc = PTR<int>(conn_comp_ids);
    for (int i = 0; i < F; i++) comp_areas[cc[i]] += fa[i];

    // Compact faces from large components
    auto* fp = PTR<int>(faces);
    int new_F = 0;
    for (int i = 0; i < F; i++) if (comp_areas[cc[i]] >= min_area) new_F++;

    auto new_faces = alloc(new_F * 12);
    auto* dst = PTR<int>(new_faces);
    int idx = 0;
    for (int i = 0; i < F; i++) {
        if (comp_areas[cc[i]] >= min_area) {
            dst[idx*3+0] = fp[i*3+0];
            dst[idx*3+1] = fp[i*3+1];
            dst[idx*3+2] = fp[i*3+2];
            idx++;
        }
    }
    faces = new_faces;
    num_fcs = new_F;
    remove_unreferenced_vertices();
}

void MtlMesh::fill_holes(float max_hole_perimeter) {
    if (!loop_boundaries) get_boundary_loops();
    if (num_bound_loops == 0) return;

    int num_loops = num_bound_loops;
    int L = num_loop_boundaries_;

    // Compute boundary edge lengths
    auto lengths = alloc(L * 4);
    CTX.dispatch("compute_loop_boundary_lengths_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:edges offset:0 atIndex:1];
        [enc setBuffer:loop_boundaries offset:0 atIndex:2];
        [enc setBytes:&L length:sizeof(int) atIndex:3];
        [enc setBuffer:lengths offset:0 atIndex:4];
    }, L);

    // Segmented sum → perimeter per loop
    auto perimeters = alloc(num_loops * 4);
    PRIMS.segmented_sum_float(lengths, perimeters, loop_boundaries_offset, num_loops);

    // Select loops with perimeter <= threshold
    auto* perim = PTR<float>(perimeters);
    auto* offsets = PTR<int>(loop_boundaries_offset);
    std::vector<int> fill_loops;
    for (int i = 0; i < num_loops; i++) {
        if (perim[i] <= max_hole_perimeter) fill_loops.push_back(i);
    }
    if (fill_loops.empty()) return;

    // Bounds-check every offset we're about to index. Upstream get_boundary_loops
    // is expected to produce monotonically non-decreasing offsets in [0, L],
    // but a corrupt loop_boundaries_offset was the proximate cause of a
    // segfault in the decoder output post-process.
    for (int loop_id : fill_loops) {
        TORCH_CHECK(loop_id >= 0 && loop_id < num_loops,
                    "fill_holes: loop_id ", loop_id, " out of range [0, ", num_loops, ")");
        int s = offsets[loop_id], e = offsets[loop_id + 1];
        TORCH_CHECK(s >= 0 && e >= s && e <= L,
                    "fill_holes: offsets[", loop_id, "..", loop_id + 1, "] = (",
                    s, ", ", e, ") out of range [0, ", L, "]");
    }

    // Compute midpoints for center vertices
    auto midpoints = alloc(L * 12);
    CTX.dispatch("compute_loop_boundary_midpoints_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:edges offset:0 atIndex:1];
        [enc setBuffer:loop_boundaries offset:0 atIndex:2];
        [enc setBytes:&L length:sizeof(int) atIndex:3];
        [enc setBuffer:midpoints offset:0 atIndex:4];
    }, L);

    // For each loop to fill: add a center vertex, create fan triangles.
    // Saturating int64 accumulation — a corrupt offsets array can otherwise
    // overflow int32 and flip the sign, producing a negative-sized alloc.
    int64_t total_new_faces_64 = 0;
    for (int loop_id : fill_loops)
        total_new_faces_64 += (int64_t)(offsets[loop_id + 1] - offsets[loop_id]);
    TORCH_CHECK(total_new_faces_64 >= 0 && total_new_faces_64 <= (int64_t)(INT_MAX / 12),
                "fill_holes: total_new_faces ", total_new_faces_64,
                " overflows safe alloc range");
    int total_new_faces = (int)total_new_faces_64;

    // Compute center vertices by averaging midpoints per loop (CPU)
    int num_new_verts = (int)fill_loops.size();
    auto new_verts = alloc(num_new_verts * 12);
    auto* mid_ptr = PTR<float>(midpoints);
    auto* nv_ptr = PTR<float>(new_verts);

    for (int ci = 0; ci < num_new_verts; ci++) {
        int li = fill_loops[ci];
        int start = offsets[li], end = offsets[li + 1];
        float cx = 0, cy = 0, cz = 0;
        int cnt = end - start;
        for (int j = start; j < end; j++) {
            cx += mid_ptr[j*3+0];
            cy += mid_ptr[j*3+1];
            cz += mid_ptr[j*3+2];
        }
        nv_ptr[ci*3+0] = cx / cnt;
        nv_ptr[ci*3+1] = cy / cnt;
        nv_ptr[ci*3+2] = cz / cnt;
    }

    // Build new faces: for each edge in each loop, create triangle (e0, e1, center)
    auto new_faces_buf = alloc((size_t)total_new_faces * 12);
    auto* nf_ptr = PTR<int>(new_faces_buf);
    auto* lb = PTR<int>(loop_boundaries);
    auto* ep = PTR<uint64_t>(edges);
    int fi = 0;
    int E = num_edges_;
    int V = num_verts;
    for (int ci = 0; ci < num_new_verts; ci++) {
        int li = fill_loops[ci];
        int center_vid = num_verts + ci;
        for (int j = offsets[li]; j < offsets[li + 1]; j++) {
            int eidx = lb[j];
            TORCH_CHECK(eidx >= 0 && eidx < E,
                        "fill_holes: loop_boundaries[", j, "] = ", eidx,
                        " out of range [0, ", E, ")");
            uint64_t e = ep[eidx];
            int e0 = (int)(e & 0xFFFFFFFF);
            int e1 = (int)(e >> 32);
            TORCH_CHECK(e0 >= 0 && e0 < V && e1 >= 0 && e1 < V,
                        "fill_holes: edge ", eidx, " has vertex index out of range (",
                        e0, ", ", e1, ") for V=", V);
            nf_ptr[fi*3+0] = e0;
            nf_ptr[fi*3+1] = e1;
            nf_ptr[fi*3+2] = center_vid;
            fi++;
        }
    }

    // Merge vertices and faces
    int old_V = num_verts, old_F = num_fcs;
    int new_total_V = old_V + num_new_verts;
    int new_total_F = old_F + total_new_faces;

    auto merged_verts = alloc(new_total_V * 12);
    memcpy([merged_verts contents], [vertices contents], old_V * 12);
    memcpy(PTR<float>(merged_verts) + old_V * 3, [new_verts contents], num_new_verts * 12);

    auto merged_faces = alloc(new_total_F * 12);
    memcpy([merged_faces contents], [faces contents], old_F * 12);
    memcpy(PTR<int>(merged_faces) + old_F * 3, [new_faces_buf contents], total_new_faces * 12);

    vertices = merged_verts;
    faces = merged_faces;
    num_verts = new_total_V;
    num_fcs = new_total_F;
    clear_cache();
}

void MtlMesh::repair_non_manifold_edges() {
    get_manifold_face_adjacency();
    int F = num_fcs, M = num_manifold_edges_;

    if (M == 0) return;

    // Step 1: Construct vertex adjacency pairs from manifold face adjacency
    auto vertex_adj_pairs = alloc(2 * M * 8);  // 2M int2
    CTX.dispatch("construct_vertex_adj_pairs_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:manifold_face_adj offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:vertex_adj_pairs offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(int) atIndex:3];
    }, M);

    // Step 2: Union-find on per-face-vertex IDs (F*3 IDs)
    int N = F * 3;
    auto comp_ids = alloc(N * 4);
    auto* cids = PTR<int>(comp_ids);
    for (int i = 0; i < N; i++) cids[i] = i;

    hook_and_compress(vertex_adj_pairs, 2 * M, comp_ids, N);
    int num_unique = compress_ids(comp_ids, N);

    // Step 3: Build representative face-vertex index for each unique vertex
    auto repr_buf = alloc(num_unique * 4);
    auto* repr = PTR<int>(repr_buf);
    for (int i = 0; i < num_unique; i++) repr[i] = -1;
    for (int i = 0; i < N; i++) {
        if (repr[cids[i]] == -1) repr[cids[i]] = i;
    }

    // Step 4: Build new vertices by indexing via representatives
    auto new_verts = alloc(num_unique * 12);
    CTX.dispatch("index_vertice_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:repr_buf offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:vertices offset:0 atIndex:2];
        [enc setBytes:&num_unique length:sizeof(int) atIndex:3];
        [enc setBuffer:new_verts offset:0 atIndex:4];
    }, num_unique);

    // Step 5: Build new faces from compressed IDs
    auto new_faces = alloc(F * 12);
    // face[i] = {comp_ids[i*3+0], comp_ids[i*3+1], comp_ids[i*3+2]}
    auto* fp = PTR<int>(new_faces);
    for (int i = 0; i < F; i++) {
        fp[i*3+0] = cids[i*3+0];
        fp[i*3+1] = cids[i*3+1];
        fp[i*3+2] = cids[i*3+2];
    }

    vertices = new_verts;
    faces = new_faces;
    num_verts = num_unique;
    // num_fcs stays the same
    clear_cache();
}

void MtlMesh::unify_face_orientations() {
    if (!manifold_face_adj) get_manifold_face_adjacency();
    int F = num_fcs, M = num_manifold_edges_;

    if (M == 0) return;

    // Step 1: Compute flip flags for each manifold edge
    auto flipped = alloc(M);
    CTX.dispatch("get_flip_flags_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:manifold_face_adj offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBytes:&M length:sizeof(int) atIndex:2];
        [enc setBuffer:flipped offset:0 atIndex:3];
    }, M);

    // Step 2: Oriented union-find (encode flip in LSB of comp_id)
    auto comp_ids = alloc(F * 4);
    auto* cids = PTR<int>(comp_ids);
    for (int i = 0; i < F; i++) cids[i] = i << 1;  // id=i, flip=0

    // Union-find over F components converges in O(log F) rounds under the
    // "hook-then-compress" pattern. Hard-cap at 64 iterations: well above the
    // theoretical bound (log2(2^31) = 31) and defensive against pathological
    // input where the kernel doesn't monotonically reduce component count.
    static constexpr int kMaxIters = 64;
    auto end_flag = alloc(4);
    int iter = 0;
    while (true) {
        PTR<int>(end_flag)[0] = 1;

        CTX.dispatch("hook_edges_with_orientation_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:manifold_face_adj offset:0 atIndex:0];
            [enc setBuffer:flipped offset:0 atIndex:1];
            [enc setBytes:&M length:sizeof(int) atIndex:2];
            [enc setBuffer:comp_ids offset:0 atIndex:3];
            [enc setBuffer:end_flag offset:0 atIndex:4];
        }, M);

        CTX.dispatch("compress_components_with_orientation_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:comp_ids offset:0 atIndex:0];
            [enc setBytes:&F length:sizeof(int) atIndex:1];
        }, F);

        if (PTR<int>(end_flag)[0] == 1) break;
        if (++iter >= kMaxIters) {
            TORCH_WARN("unify_face_orientations: union-find did not converge in ",
                       kMaxIters, " iterations (F=", F, ", M=", M,
                       "); aborting, mesh orientations may be partially unified");
            break;
        }
    }

    // Step 3: Flip faces where flip bit is set
    CTX.dispatch("inplace_flip_faces_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:faces offset:0 atIndex:0];
        [enc setBuffer:comp_ids offset:0 atIndex:1];
        [enc setBytes:&F length:sizeof(int) atIndex:2];
    }, F);

    clear_cache();
}

// ========== Simplification ==========

std::tuple<int, int> MtlMesh::simplify_step(float lambda_edge_length, float lambda_skinny, float threshold, bool timing) {
    get_vertex_face_adjacency();
    get_edges();
    get_boundary_info();

    int V = num_verts, F = num_fcs, E = num_edges_;

    // Step 1: Compute QEM per vertex
    auto qems = alloc(V * 40);  // QEM = 10 floats = 40 bytes
    CTX.dispatch("get_qem_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:vert2face offset:0 atIndex:2];
        [enc setBuffer:vert2face_offset offset:0 atIndex:3];
        [enc setBytes:&V length:sizeof(int) atIndex:4];
        [enc setBuffer:qems offset:0 atIndex:5];
    }, V);

    // Step 2: Compute edge collapse costs
    auto costs = alloc(E * 4);
    CTX.dispatch("get_edge_collapse_cost_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:vert2face offset:0 atIndex:2];
        [enc setBuffer:vert2face_offset offset:0 atIndex:3];
        [enc setBuffer:edges offset:0 atIndex:4];
        [enc setBuffer:vert_is_boundary offset:0 atIndex:5];
        [enc setBuffer:qems offset:0 atIndex:6];
        [enc setBytes:&E length:sizeof(int) atIndex:7];
        [enc setBytes:&lambda_edge_length length:sizeof(float) atIndex:8];
        [enc setBytes:&lambda_skinny length:sizeof(float) atIndex:9];
        [enc setBuffer:costs offset:0 atIndex:10];
    }, E);

    // Step 3: Propagate minimum cost to faces — single-pass 64-bit atomicMin
    auto prop_costs = alloc(F * 8);
    memset([prop_costs contents], 0xFF, F * 8);  // UINT64_MAX

    CTX.dispatch("propagate_cost_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:edges offset:0 atIndex:0];
        [enc setBuffer:vert2face offset:0 atIndex:1];
        [enc setBuffer:vert2face_offset offset:0 atIndex:2];
        [enc setBuffer:costs offset:0 atIndex:3];
        [enc setBytes:&E length:sizeof(int) atIndex:4];
        [enc setBuffer:prop_costs offset:0 atIndex:5];
    }, E);

    // Step 4: Collapse edges
    auto vert_kept = alloc(V * 4);
    auto face_kept = alloc(F * 4);
    auto* vk = PTR<int>(vert_kept);
    auto* fk = PTR<int>(face_kept);
    for (int i = 0; i < V; i++) vk[i] = 1;
    for (int i = 0; i < F; i++) fk[i] = 1;

    CTX.dispatch("collapse_edges_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:vertices offset:0 atIndex:0];
        [enc setBuffer:faces offset:0 atIndex:1];
        [enc setBuffer:edges offset:0 atIndex:2];
        [enc setBuffer:vert2face offset:0 atIndex:3];
        [enc setBuffer:vert2face_offset offset:0 atIndex:4];
        [enc setBuffer:costs offset:0 atIndex:5];
        [enc setBuffer:prop_costs offset:0 atIndex:6];
        [enc setBuffer:vert_is_boundary offset:0 atIndex:7];
        [enc setBytes:&E length:sizeof(int) atIndex:8];
        [enc setBytes:&threshold length:sizeof(float) atIndex:9];
        [enc setBuffer:vert_kept offset:0 atIndex:10];
        [enc setBuffer:face_kept offset:0 atIndex:11];
    }, E);

    // Step 5: Compact vertices and faces using prefix-sum maps.
    // ExclusiveSum on kept flags → vertices_map/faces_map.
    // vertices_map[v] = new index for vertex v (even if removed — prefix sum
    // maps removed vertices to same index as next kept vertex, which is what
    // the compress step relies on for parallel collapse races).

    // (V+1)*4 and (F+1)*4 — guard against sign-flip on meshes near INT_MAX.
    TORCH_CHECK((int64_t)(V + 1) * 4 <= (int64_t)INT_MAX,
                "simplify_step: (V+1)*4 overflows (V=", V, ")");
    TORCH_CHECK((int64_t)(F + 1) * 4 <= (int64_t)INT_MAX,
                "simplify_step: (F+1)*4 overflows (F=", F, ")");

    auto vert_map_buf = alloc((size_t)(V + 1) * 4);
    memcpy([vert_map_buf contents], [vert_kept contents], (size_t)V * 4);
    PTR<int>(vert_map_buf)[V] = 0;
    PRIMS.exclusive_sum(vert_map_buf, V + 1);
    auto* vert_map = PTR<int>(vert_map_buf);
    int new_V = vert_map[V];

    auto face_map_buf = alloc((size_t)(F + 1) * 4);
    memcpy([face_map_buf contents], [face_kept contents], (size_t)F * 4);
    PTR<int>(face_map_buf)[F] = 0;
    PRIMS.exclusive_sum(face_map_buf, F + 1);
    auto* face_map = PTR<int>(face_map_buf);
    int new_F = face_map[F];

    // Compress vertices
    auto new_verts = alloc(new_V * 12);
    auto* vsrc = PTR<float>(vertices);
    auto* vdst = PTR<float>(new_verts);
    for (int i = 0; i < V; i++) {
        if (vk[i]) {
            int j = vert_map[i];
            vdst[j*3+0] = vsrc[i*3+0];
            vdst[j*3+1] = vsrc[i*3+1];
            vdst[j*3+2] = vsrc[i*3+2];
        }
    }

    // Compress faces:
    // new_faces[face_map[i]] = { vert_map[old_faces[i].x/y/z] }
    auto new_faces = alloc((size_t)new_F * 12);
    auto* fsrc = PTR<int>(faces);
    auto* fdst = PTR<int>(new_faces);
    for (int i = 0; i < F; i++) {
        int new_id = face_map[i];
        int is_kept = face_map[i + 1] == new_id + 1;
        if (is_kept) {
            int a = fsrc[i*3+0], b = fsrc[i*3+1], c = fsrc[i*3+2];
            TORCH_CHECK(a >= 0 && a < V && b >= 0 && b < V && c >= 0 && c < V,
                        "simplify_step: face ", i, " has vertex index out of range (",
                        a, ", ", b, ", ", c, ") for V=", V);
            fdst[new_id*3+0] = vert_map[a];
            fdst[new_id*3+1] = vert_map[b];
            fdst[new_id*3+2] = vert_map[c];
        }
    }

    vertices = new_verts;
    faces = new_faces;
    num_verts = new_V;
    num_fcs = new_F;
    clear_cache();
    return {new_V, new_F};
}

// ========== Atlasing ==========

// CPU uint64 RLE (MetalPrimitives::run_length_encode only handles int keys)
static int rle_uint64(id<MTLBuffer> sorted, id<MTLBuffer> unique, id<MTLBuffer> counts, int N) {
    auto* sk = PTR<uint64_t>(sorted);
    auto* uk = PTR<uint64_t>(unique);
    auto* cnt = PTR<int>(counts);
    int num_runs = 0;
    for (int i = 0; i < N; ) {
        uint64_t key = sk[i];
        int run = 1;
        while (i + run < N && sk[i + run] == key) run++;
        uk[num_runs] = key;
        cnt[num_runs] = run;
        num_runs++;
        i += run;
    }
    return num_runs;
}

struct ChartData {
    id<MTLBuffer> cones;  // num_charts * 16 (float4)
    id<MTLBuffer> areas;  // num_charts * 4 (float)
};

// Rebuild chart cones and areas from face data after chart ID compression
static ChartData rebuild_chart_data(id<MTLBuffer> face_norms, id<MTLBuffer> face_ar,
                                     id<MTLBuffer> chart_ids_buf, int F, int nc,
                                     id<MTLDevice> dev) {
    auto mk = [&](size_t bytes) {
        return [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    };

    auto* fn = PTR<float>(face_norms);
    auto* cids = PTR<int>(chart_ids_buf);

    // Aggregate face normals per chart
    std::vector<float> agg(nc * 3, 0.0f);
    for (int i = 0; i < F; i++) {
        int c = cids[i];
        agg[c*3+0] += fn[i*3+0];
        agg[c*3+1] += fn[i*3+1];
        agg[c*3+2] += fn[i*3+2];
    }

    // Normalize + compute max half angles
    std::vector<float> max_ang(nc, 0.0f);
    for (int i = 0; i < F; i++) {
        int c = cids[i];
        float nx = agg[c*3], ny = agg[c*3+1], nz = agg[c*3+2];
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) { nx /= len; ny /= len; nz /= len; }
        float d = nx*fn[i*3] + ny*fn[i*3+1] + nz*fn[i*3+2];
        d = fmaxf(-1.0f, fminf(1.0f, d));
        float a = acosf(d);
        if (a > max_ang[c]) max_ang[c] = a;
    }

    auto cones_buf = mk(nc * 16);
    auto* cones = PTR<float>(cones_buf);
    for (int c = 0; c < nc; c++) {
        float nx = agg[c*3], ny = agg[c*3+1], nz = agg[c*3+2];
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) { nx /= len; ny /= len; nz /= len; }
        cones[c*4] = nx; cones[c*4+1] = ny; cones[c*4+2] = nz; cones[c*4+3] = max_ang[c];
    }

    auto areas_buf = mk(nc * 4);
    memset([areas_buf contents], 0, nc * 4);
    auto* ca = PTR<float>(areas_buf);
    auto* fa = PTR<float>(face_ar);
    for (int i = 0; i < F; i++) ca[cids[i]] += fa[i];

    return {cones_buf, areas_buf};
}

void MtlMesh::compute_charts(float threshold, int refine_iters, int global_iters,
                              float smooth, float area_w, float perim_w) {
    if (!manifold_face_adj) get_manifold_face_adjacency();
    if (!face_normals) compute_face_normals();
    if (!face_areas) compute_face_areas();
    if (!edge2face) get_edge_face_adjacency();

    int F = num_fcs, M = num_manifold_edges_;

    // Initialize chart_ids: one chart per face
    atlas_chart_ids = alloc(F * 4);
    auto* chart_ids = PTR<int>(atlas_chart_ids);
    for (int i = 0; i < F; i++) chart_ids[i] = i;

    int num_charts = F;

    // Initialize chart data from faces
    id<MTLBuffer> chart_normal_cones = nil;
    id<MTLBuffer> chart_areas = nil;
    { auto cd = rebuild_chart_data(face_normals, face_areas, atlas_chart_ids, F, num_charts, dev_);
      chart_normal_cones = cd.cones; chart_areas = cd.areas; }

    for (int global = 0; global < global_iters; global++) {
        // === Chart collapse phase ===

        // Build chart adjacency from face adjacency
        auto chart_adj = alloc(M * 8);
        auto chart_adj_length = alloc(M * 4);
        CTX.dispatch("init_chart_adj_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:vertices offset:0 atIndex:0];
            [enc setBuffer:faces offset:0 atIndex:1];
            [enc setBuffer:manifold_face_adj offset:0 atIndex:2];
            [enc setBuffer:atlas_chart_ids offset:0 atIndex:3];
            [enc setBytes:&M length:sizeof(int) atIndex:4];
            [enc setBuffer:chart_adj offset:0 atIndex:5];
            [enc setBuffer:chart_adj_length offset:0 atIndex:6];
        }, M);

        // Sort chart_adj + values for length aggregation
        auto sort_vals = alloc(M * 4);
        auto* svp = PTR<int>(sort_vals);
        for (int i = 0; i < M; i++) svp[i] = i;
        PRIMS.sort_pairs_uint64(chart_adj, sort_vals, M);

        // Remove ULONG_MAX entries (same-chart edges sort to end)
        auto* ca = PTR<uint64_t>(chart_adj);
        int valid_E = M;
        while (valid_E > 0 && ca[valid_E - 1] == UINT64_MAX) valid_E--;
        if (valid_E == 0) break;

        // RLE → unique chart edges (uint64 keys)
        auto unique_adj = alloc(valid_E * 8);
        auto adj_counts = alloc(valid_E * 4);
        int chart_E = rle_uint64(chart_adj, unique_adj, adj_counts, valid_E);

        // Aggregate lengths per unique chart edge (CPU)
        auto agg_lengths = alloc_zero(chart_E * 4);
        {
            auto* al = PTR<float>(agg_lengths);
            auto* cal = PTR<float>(chart_adj_length);
            auto* sv = PTR<int>(sort_vals);
            auto* ua = PTR<uint64_t>(unique_adj);
            int eidx = 0;
            for (int i = 0; i < valid_E; ) {
                uint64_t key = ua[eidx];
                float sum = 0;
                while (i < valid_E && ca[i] == key) { sum += cal[sv[i]]; i++; }
                al[eidx++] = sum;
            }
        }

        // Build chart-edge CSR adjacency
        auto chart2edge_cnt = alloc_zero(num_charts * 4);
        auto chart_perims = alloc_zero(num_charts * 4);
        CTX.dispatch("get_chart_edge_cnt_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:unique_adj offset:0 atIndex:0];
            [enc setBuffer:agg_lengths offset:0 atIndex:1];
            [enc setBytes:&chart_E length:sizeof(int) atIndex:2];
            [enc setBuffer:chart2edge_cnt offset:0 atIndex:3];
            [enc setBuffer:chart_perims offset:0 atIndex:4];
        }, chart_E);

        auto chart2edge_offset = alloc((num_charts + 1) * 4);
        memcpy([chart2edge_offset contents], [chart2edge_cnt contents], num_charts * 4);
        PTR<int>(chart2edge_offset)[num_charts] = 0;
        PRIMS.exclusive_sum(chart2edge_offset, num_charts + 1);
        int total_c2e = PTR<int>(chart2edge_offset)[num_charts];
        if (total_c2e == 0) break;

        auto chart2edge = alloc(total_c2e * 4);
        memset([chart2edge_cnt contents], 0, num_charts * 4);
        CTX.dispatch("get_chart_edge_adjacency_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:unique_adj offset:0 atIndex:0];
            [enc setBytes:&chart_E length:sizeof(int) atIndex:1];
            [enc setBuffer:chart2edge offset:0 atIndex:2];
            [enc setBuffer:chart2edge_offset offset:0 atIndex:3];
            [enc setBuffer:chart2edge_cnt offset:0 atIndex:4];
        }, chart_E);

        // Compute chart adjacency costs
        auto chart_adj_costs = alloc(chart_E * 4);
        CTX.dispatch("compute_chart_adjacency_cost_kernel", [&](id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:unique_adj offset:0 atIndex:0];
            [enc setBuffer:chart_normal_cones offset:0 atIndex:1];
            [enc setBuffer:agg_lengths offset:0 atIndex:2];
            [enc setBuffer:chart_perims offset:0 atIndex:3];
            [enc setBuffer:chart_areas offset:0 atIndex:4];
            [enc setBytes:&area_w length:sizeof(float) atIndex:5];
            [enc setBytes:&perim_w length:sizeof(float) atIndex:6];
            [enc setBytes:&chart_E length:sizeof(int) atIndex:7];
            [enc setBuffer:chart_adj_costs offset:0 atIndex:8];
        }, chart_E);

        // Multi-pass collapse: iterate until no more merges occur
        while (true) {
            auto prop_costs = alloc(num_charts * 8);
            auto* pc = PTR<uint64_t>(prop_costs);
            for (int i = 0; i < num_charts; i++) pc[i] = UINT64_MAX;

            CTX.dispatch("chart_propagate_cost_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:chart2edge offset:0 atIndex:0];
                [enc setBuffer:chart2edge_offset offset:0 atIndex:1];
                [enc setBuffer:chart_adj_costs offset:0 atIndex:2];
                [enc setBytes:&num_charts length:sizeof(int) atIndex:3];
                [enc setBuffer:prop_costs offset:0 atIndex:4];
            }, num_charts);

            auto chart_map = alloc(num_charts * 4);
            auto* cm = PTR<int>(chart_map);
            for (int i = 0; i < num_charts; i++) cm[i] = i;

            auto end_flag = alloc(4);
            PTR<int>(end_flag)[0] = 1;
            CTX.dispatch("chart_collapse_edges_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:unique_adj offset:0 atIndex:0];
                [enc setBuffer:chart_adj_costs offset:0 atIndex:1];
                [enc setBuffer:prop_costs offset:0 atIndex:2];
                [enc setBytes:&threshold length:sizeof(float) atIndex:3];
                [enc setBytes:&chart_E length:sizeof(int) atIndex:4];
                [enc setBuffer:chart_map offset:0 atIndex:5];
                [enc setBuffer:chart_normal_cones offset:0 atIndex:6];
                [enc setBuffer:end_flag offset:0 atIndex:7];
            }, chart_E);

            // No collapses occurred — converged
            if (PTR<int>(end_flag)[0] == 1) break;

            // Path compress and update chart_ids
            for (int i = 0; i < num_charts; i++)
                while (cm[cm[i]] != cm[i]) cm[i] = cm[cm[i]];
            for (int i = 0; i < F; i++) chart_ids[i] = cm[chart_ids[i]];

            // Compress chart IDs and rebuild for next collapse pass
            num_charts = compress_ids(atlas_chart_ids, F);
            { auto cd = rebuild_chart_data(face_normals, face_areas, atlas_chart_ids, F, num_charts, dev_);
              chart_normal_cones = cd.cones; chart_areas = cd.areas; }

            // Rebuild adjacency for next collapse pass
            chart_adj = alloc(M * 8);
            chart_adj_length = alloc(M * 4);
            CTX.dispatch("init_chart_adj_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:vertices offset:0 atIndex:0];
                [enc setBuffer:faces offset:0 atIndex:1];
                [enc setBuffer:manifold_face_adj offset:0 atIndex:2];
                [enc setBuffer:atlas_chart_ids offset:0 atIndex:3];
                [enc setBytes:&M length:sizeof(int) atIndex:4];
                [enc setBuffer:chart_adj offset:0 atIndex:5];
                [enc setBuffer:chart_adj_length offset:0 atIndex:6];
            }, M);

            sort_vals = alloc(M * 4);
            svp = PTR<int>(sort_vals);
            for (int i = 0; i < M; i++) svp[i] = i;
            PRIMS.sort_pairs_uint64(chart_adj, sort_vals, M);

            ca = PTR<uint64_t>(chart_adj);
            valid_E = M;
            while (valid_E > 0 && ca[valid_E - 1] == UINT64_MAX) valid_E--;
            if (valid_E == 0) break;

            unique_adj = alloc(valid_E * 8);
            adj_counts = alloc(valid_E * 4);
            chart_E = rle_uint64(chart_adj, unique_adj, adj_counts, valid_E);

            agg_lengths = alloc_zero(chart_E * 4);
            {
                auto* al = PTR<float>(agg_lengths);
                auto* cal = PTR<float>(chart_adj_length);
                auto* sv = PTR<int>(sort_vals);
                auto* ua = PTR<uint64_t>(unique_adj);
                int eidx = 0;
                for (int i = 0; i < valid_E; ) {
                    uint64_t key = ua[eidx];
                    float sum = 0;
                    while (i < valid_E && ca[i] == key) { sum += cal[sv[i]]; i++; }
                    al[eidx++] = sum;
                }
            }

            chart2edge_cnt = alloc_zero(num_charts * 4);
            chart_perims = alloc_zero(num_charts * 4);
            CTX.dispatch("get_chart_edge_cnt_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:unique_adj offset:0 atIndex:0];
                [enc setBuffer:agg_lengths offset:0 atIndex:1];
                [enc setBytes:&chart_E length:sizeof(int) atIndex:2];
                [enc setBuffer:chart2edge_cnt offset:0 atIndex:3];
                [enc setBuffer:chart_perims offset:0 atIndex:4];
            }, chart_E);

            chart2edge_offset = alloc((num_charts + 1) * 4);
            memcpy([chart2edge_offset contents], [chart2edge_cnt contents], num_charts * 4);
            PTR<int>(chart2edge_offset)[num_charts] = 0;
            PRIMS.exclusive_sum(chart2edge_offset, num_charts + 1);
            total_c2e = PTR<int>(chart2edge_offset)[num_charts];
            if (total_c2e == 0) break;

            chart2edge = alloc(total_c2e * 4);
            memset([chart2edge_cnt contents], 0, num_charts * 4);
            CTX.dispatch("get_chart_edge_adjacency_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:unique_adj offset:0 atIndex:0];
                [enc setBytes:&chart_E length:sizeof(int) atIndex:1];
                [enc setBuffer:chart2edge offset:0 atIndex:2];
                [enc setBuffer:chart2edge_offset offset:0 atIndex:3];
                [enc setBuffer:chart2edge_cnt offset:0 atIndex:4];
            }, chart_E);

            chart_adj_costs = alloc(chart_E * 4);
            CTX.dispatch("compute_chart_adjacency_cost_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:unique_adj offset:0 atIndex:0];
                [enc setBuffer:chart_normal_cones offset:0 atIndex:1];
                [enc setBuffer:agg_lengths offset:0 atIndex:2];
                [enc setBuffer:chart_perims offset:0 atIndex:3];
                [enc setBuffer:chart_areas offset:0 atIndex:4];
                [enc setBytes:&area_w length:sizeof(float) atIndex:5];
                [enc setBytes:&perim_w length:sizeof(float) atIndex:6];
                [enc setBytes:&chart_E length:sizeof(int) atIndex:7];
                [enc setBuffer:chart_adj_costs offset:0 atIndex:8];
            }, chart_E);
        }

        // Final compress after collapse convergence
        num_charts = compress_ids(atlas_chart_ids, F);
        { auto cd = rebuild_chart_data(face_normals, face_areas, atlas_chart_ids, F, num_charts, dev_);
          chart_normal_cones = cd.cones; chart_areas = cd.areas; }

        // === Refine phase ===
        for (int ri = 0; ri < refine_iters; ri++) {
            // Refine chart assignment
            auto pong_chart_ids = alloc(F * 4);
            CTX.dispatch("refine_charts_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:chart_normal_cones offset:0 atIndex:0];
                [enc setBuffer:face_normals offset:0 atIndex:1];
                [enc setBuffer:vertices offset:0 atIndex:2];
                [enc setBuffer:edges offset:0 atIndex:3];
                [enc setBuffer:face2edge offset:0 atIndex:4];
                [enc setBuffer:edge2face offset:0 atIndex:5];
                [enc setBuffer:edge2face_offset offset:0 atIndex:6];
                [enc setBytes:&F length:sizeof(int) atIndex:7];
                [enc setBytes:&smooth length:sizeof(float) atIndex:8];
                [enc setBuffer:atlas_chart_ids offset:0 atIndex:9];
                [enc setBuffer:pong_chart_ids offset:0 atIndex:10];
            }, F);

            memcpy([atlas_chart_ids contents], [pong_chart_ids contents], F * 4);
            chart_ids = PTR<int>(atlas_chart_ids);

            // Rebuild cones after each refinement iteration
            { auto cd = rebuild_chart_data(face_normals, face_areas, atlas_chart_ids, F, num_charts, dev_);
              chart_normal_cones = cd.cones; chart_areas = cd.areas; }
        }

        // Reconnect: ensure each chart is a single connected component
        auto cc_ids = alloc(F * 4);
        auto* cci = PTR<int>(cc_ids);
        for (int i = 0; i < F; i++) cci[i] = i;

        auto end_flag2 = alloc(4);
        while (true) {
            PTR<int>(end_flag2)[0] = 1;
            CTX.dispatch("hook_edges_if_same_chart_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:manifold_face_adj offset:0 atIndex:0];
                [enc setBuffer:atlas_chart_ids offset:0 atIndex:1];
                [enc setBytes:&M length:sizeof(int) atIndex:2];
                [enc setBuffer:cc_ids offset:0 atIndex:3];
                [enc setBuffer:end_flag2 offset:0 atIndex:4];
            }, M);

            CTX.dispatch("compress_components_kernel", [&](id<MTLComputeCommandEncoder> enc) {
                [enc setBuffer:cc_ids offset:0 atIndex:0];
                [enc setBytes:&F length:sizeof(int) atIndex:1];
            }, F);

            if (PTR<int>(end_flag2)[0] == 1) break;
        }

        memcpy([atlas_chart_ids contents], [cc_ids contents], F * 4);
        chart_ids = PTR<int>(atlas_chart_ids);
        num_charts = compress_ids(atlas_chart_ids, F);

        // Rebuild chart data for next global iteration
        { auto cd = rebuild_chart_data(face_normals, face_areas, atlas_chart_ids, F, num_charts, dev_);
          chart_normal_cones = cd.cones; chart_areas = cd.areas; }
    }

    atlas_num_charts = num_charts;

    // === Build output ===
    // Sort faces by chart_id
    auto sort_keys = alloc(F * 4);
    memcpy([sort_keys contents], [atlas_chart_ids contents], F * 4);
    auto sort_face_idx = alloc(F * 4);
    auto* sfi = PTR<int>(sort_face_idx);
    for (int i = 0; i < F; i++) sfi[i] = i;
    PRIMS.sort_pairs_int(sort_keys, sort_face_idx, F);

    // Expand to per-face-vertex pairs: (chart_id << 32 | vertex_id)
    auto pack = alloc(3 * F * 8);
    CTX.dispatch("expand_chart_ids_and_vertex_ids_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:sort_keys offset:0 atIndex:0];
        [enc setBuffer:sort_face_idx offset:0 atIndex:1];
        [enc setBuffer:faces offset:0 atIndex:2];
        [enc setBytes:&F length:sizeof(int) atIndex:3];
        [enc setBuffer:pack offset:0 atIndex:4];
    }, F);

    // Sort + unique packed values → vertex map
    auto dummy = alloc(3 * F * 4);
    PRIMS.sort_pairs_uint64(pack, dummy, 3 * F);

    auto unique_pack = alloc(3 * F * 8);
    auto pack_counts = alloc(3 * F * 4);
    int num_chart_verts = rle_uint64(pack, unique_pack, pack_counts, 3 * F);

    // Unpack vertex IDs and chart vertex offsets
    atlas_chart_vertex_map = alloc(num_chart_verts * 4);
    atlas_chart_vertex_offset = alloc((num_charts + 1) * 4);
    memset([atlas_chart_vertex_offset contents], 0, (num_charts + 1) * 4);

    int nv = num_chart_verts;
    CTX.dispatch("unpack_vertex_ids_kernel", [&](id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:unique_pack offset:0 atIndex:0];
        [enc setBytes:&nv length:sizeof(int) atIndex:1];
        [enc setBuffer:atlas_chart_vertex_map offset:0 atIndex:2];
        [enc setBuffer:atlas_chart_vertex_offset offset:0 atIndex:3];
    }, nv);

    // Build chart face offsets
    auto chart_face_counts = alloc(num_charts * 4);
    auto unique_chart_buf = alloc(num_charts * 4);
    PRIMS.run_length_encode(sort_keys, unique_chart_buf, chart_face_counts, F);

    atlas_chart_faces_offset = alloc((num_charts + 1) * 4);
    memcpy([atlas_chart_faces_offset contents], [chart_face_counts contents], num_charts * 4);
    PTR<int>(atlas_chart_faces_offset)[num_charts] = 0;
    PRIMS.exclusive_sum(atlas_chart_faces_offset, num_charts + 1);

    // Build remapped faces (chart-local vertex indices via binary search)
    auto* voff = PTR<int>(atlas_chart_vertex_offset);
    auto* pk = PTR<uint64_t>(unique_pack);

    atlas_chart_faces = alloc(F * 12);
    auto* acf = PTR<int>(atlas_chart_faces);
    auto* sk2 = PTR<int>(sort_keys);

    for (int i = 0; i < F; i++) {
        int chart = sk2[i];
        int orig_fi = PTR<int>(sort_face_idx)[i];
        auto* fp = PTR<int>(faces) + orig_fi * 3;

        for (int j = 0; j < 3; j++) {
            uint64_t key = ((uint64_t)chart << 32) | (uint64_t)fp[j];
            int lo = voff[chart];
            int hi = (chart + 1 < num_charts) ? voff[chart + 1] : num_chart_verts;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (pk[mid] < key) lo = mid + 1;
                else hi = mid;
            }
            acf[i*3+j] = lo;
        }
    }
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MtlMesh::read_atlas_charts() {
    TORCH_CHECK(atlas_chart_ids != nil, "Charts not computed");

    auto chart_ids_t = ints_to_tensor(atlas_chart_ids, num_fcs);

    int nv = 0;
    auto* voff = PTR<int>(atlas_chart_vertex_offset);
    if (atlas_num_charts > 0) nv = voff[atlas_num_charts];
    auto vmap_t = ints_to_tensor(atlas_chart_vertex_map, nv);

    auto faces_t = faces_to_tensor(atlas_chart_faces, num_fcs);
    auto voff_t = ints_to_tensor(atlas_chart_vertex_offset, atlas_num_charts + 1);
    auto foff_t = ints_to_tensor(atlas_chart_faces_offset, atlas_num_charts + 1);

    return {atlas_num_charts, chart_ids_t, vmap_t, faces_t, voff_t, foff_t};
}

} // namespace mtlmesh
