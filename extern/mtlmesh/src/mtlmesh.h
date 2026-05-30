// MtlMesh: Pure Metal mesh processing class
// All GPU data lives in id<MTLBuffer> with StorageModeShared.
// PyTorch tensors only at the Python I/O boundary (init/read).
#pragma once

#import <Metal/Metal.h>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <cstring>
#import "metal_context.h"
#import "metal_primitives.h"

namespace mtlmesh {

class MtlMesh {
public:
    // Core geometry
    id<MTLBuffer> vertices;        // V * 12 bytes (packed_float3, 12-byte stride)
    id<MTLBuffer> faces;           // F * 12 bytes (packed_int3, 12-byte stride)
    int num_verts = 0;
    int num_fcs = 0;

    // Geometric properties
    id<MTLBuffer> face_areas;      // F * 4 (float)
    id<MTLBuffer> face_normals;    // F * 12 (packed_float3)
    id<MTLBuffer> vertex_normals;  // V * 12 (packed_float3)

    // Connectivity — edges
    id<MTLBuffer> edges;           // E * 8 (uint64_t: min<<32|max)
    id<MTLBuffer> edge2face_cnt;   // E * 4 (int)
    id<MTLBuffer> edge2face_offset;// (E+1) * 4 (int)
    id<MTLBuffer> edge2face;       // variable * 4 (int)
    id<MTLBuffer> face2edge;       // F * 16 (Metal int3, 16-byte stride)
    int num_edges_ = 0;

    // Connectivity — boundaries
    id<MTLBuffer> boundaries;      // B * 4 (int, edge indices)
    id<MTLBuffer> vert_is_boundary;// V * 1 (uint8_t)
    id<MTLBuffer> vert_is_manifold;// V * 1 (uint8_t)
    int num_boundaries_ = 0;

    // Connectivity — vertex-face adjacency
    id<MTLBuffer> vert2face;       // variable * 4 (int)
    id<MTLBuffer> vert2face_cnt;   // (V+1) * 4 (int)
    id<MTLBuffer> vert2face_offset;// (V+1) * 4 (int)

    // Connectivity — vertex-edge adjacency
    id<MTLBuffer> vert2edge;       // variable * 4 (int)
    id<MTLBuffer> vert2edge_cnt;   // (V+1) * 4 (int)
    id<MTLBuffer> vert2edge_offset;// (V+1) * 4 (int)

    // Connectivity — vertex-boundary adjacency
    id<MTLBuffer> vert2bound;      // variable * 4 (int)
    id<MTLBuffer> vert2bound_cnt;  // (V+1) * 4 (int)
    id<MTLBuffer> vert2bound_offset;// (V+1) * 4 (int)

    // Connectivity — manifold adjacency
    id<MTLBuffer> manifold_face_adj;   // M * 8 (int2)
    id<MTLBuffer> manifold_bound_adj;  // MBV * 8 (int2)
    int num_manifold_edges_ = 0;
    int num_manifold_bound_verts_ = 0;

    // Connected components
    id<MTLBuffer> conn_comp_ids;       // F * 4 (int)
    id<MTLBuffer> bound_conn_comp_ids; // B * 4 (int)
    id<MTLBuffer> loop_boundaries;     // L * 4 (int)
    id<MTLBuffer> loop_boundaries_offset; // (nloops+1) * 4 (int)
    int num_conn_comps = 0;
    int num_bound_conn_comps = 0;
    int num_bound_loops = 0;
    int num_loop_boundaries_ = 0;

    // Atlasing
    int atlas_num_charts = 0;
    id<MTLBuffer> atlas_chart_ids;
    id<MTLBuffer> atlas_chart_vertex_map;
    id<MTLBuffer> atlas_chart_faces;       // int3 stride (16 bytes)
    id<MTLBuffer> atlas_chart_faces_offset;
    id<MTLBuffer> atlas_chart_vertex_offset;

    MtlMesh();
    ~MtlMesh();

    int num_vertices() const;
    int num_faces() const;
    int num_edges() const;
    int num_boundaries() const;
    int num_conneted_components() const;
    int num_boundary_conneted_components() const;
    int num_boundary_loops() const;

    void clear_cache();

    // I/O (torch::Tensor at Python boundary)
    void init(const torch::Tensor& vertices, const torch::Tensor& faces);
    std::tuple<torch::Tensor, torch::Tensor> read();
    torch::Tensor read_face_areas();
    torch::Tensor read_face_normals();
    torch::Tensor read_vertex_normals();
    torch::Tensor read_edges();
    torch::Tensor read_boundaries();
    torch::Tensor read_manifold_face_adjacency();
    torch::Tensor read_manifold_boundary_adjacency();
    std::tuple<int, torch::Tensor> read_connected_components();
    std::tuple<int, torch::Tensor> read_boundary_connected_components();
    std::tuple<int, torch::Tensor, torch::Tensor> read_boundary_loops();
    std::unordered_map<std::string, torch::Tensor> read_all_cache();

    // Geometry
    void compute_face_areas();
    void compute_face_normals();
    void compute_vertex_normals();

    // Connectivity
    void get_vertex_face_adjacency();
    void get_edges();
    void get_edge_face_adjacency();
    void get_vertex_edge_adjacency();
    void get_boundary_info();
    void get_vertex_boundary_adjacency();
    void get_vertex_is_manifold();
    void get_manifold_face_adjacency();
    void get_manifold_boundary_adjacency();
    void get_connected_components();
    void get_boundary_connected_components();
    void get_boundary_loops();

    // Cleanup
    void remove_faces(torch::Tensor& face_mask);
    void remove_unreferenced_vertices();
    void remove_duplicate_faces();
    void remove_degenerate_faces(float abs_thresh, float rel_thresh);
    void fill_holes(float max_hole_perimeter);
    void repair_non_manifold_edges();
    void remove_non_manifold_faces();
    void remove_small_connected_components(float min_area);
    void unify_face_orientations();

    // Simplification
    std::tuple<int, int> simplify_step(float lambda_edge_length, float lambda_skinny, float threshold, bool timing = false);

    // Atlasing
    void compute_charts(float threshold_cone_half_angle_rad, int refine_iterations,
                        int global_iterations, float smooth_strength,
                        float area_penalty_weight, float perimeter_area_ratio_weight);
    std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> read_atlas_charts();

private:
    id<MTLDevice> dev_;

    // Backing tensors for zero-copy Metal buffers created during init().
    // These keep the underlying memory alive for the lifetime of the MtlMesh.
    torch::Tensor init_verts_backing_;
    torch::Tensor init_faces_backing_;

    // Buffer helpers
    id<MTLBuffer> alloc(size_t bytes);
    id<MTLBuffer> alloc_zero(size_t bytes);

    // int3 stride conversion helpers (12-byte packed ↔ 16-byte Metal int3)
    id<MTLBuffer> faces_from_tensor(const torch::Tensor& t);  // [F,3] int32 → F*12
    torch::Tensor faces_to_tensor(id<MTLBuffer> buf, int F);  // F*12 → [F,3] int32

    // float3 helpers (packed_float3 is 12 bytes, matches [V,3] float32)
    id<MTLBuffer> verts_from_tensor(const torch::Tensor& t);  // [V,3] float32 → V*12
    torch::Tensor verts_to_tensor(id<MTLBuffer> buf, int N);  // N*12 → [N,3] float32

    // int2 helpers (int2 is 8 bytes, matches [M,2] int32)
    torch::Tensor int2_to_tensor(id<MTLBuffer> buf, int N);   // N*8 → [N,2] int32

    // int helpers
    id<MTLBuffer> ints_from_tensor(const torch::Tensor& t);
    torch::Tensor ints_to_tensor(id<MTLBuffer> buf, int N);

    // Helpers: compress IDs and union-find
    int compress_ids(id<MTLBuffer> ids, int N);
    void hook_and_compress(id<MTLBuffer> adj, int M, id<MTLBuffer> comp_ids, int N);
};

} // namespace mtlmesh
