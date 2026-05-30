#include "cumesh.h"


namespace cumesh {

CuMesh::CuMesh() {}

CuMesh::~CuMesh() {
    vertices.free();
    faces.free();
    face_areas.free();
    face_normals.free();
    vertex_normals.free();
    edges.free();
    boundaries.free();
    vert_is_boundary.free();
    vert_is_manifold.free();
    vert2edge.free();
    vert2edge_cnt.free();
    vert2edge_offset.free();
    vert2bound.free();
    vert2bound_cnt.free();
    vert2bound_offset.free();
    edge2face.free();
    edge2face_cnt.free();
    edge2face_offset.free();
    face2edge.free();
    vert2face.free();
    vert2face_cnt.free();
    vert2face_offset.free();
    manifold_face_adj.free();
    manifold_bound_adj.free();
    conn_comp_ids.free();
    bound_conn_comp_ids.free();
    loop_boundaries.free();
    loop_boundaries_offset.free();
    vertices_map.free();
    faces_map.free();
    edge_collapse_costs.free();
    propagated_costs.free();

    atlas_chart_ids.free();
    atlas_chart_vertex_map.free();
    atlas_chart_faces.free();
    atlas_chart_faces_offset.free();
    atlas_chart_vertex_offset.free();
    atlas_chart_uvs.free();

    atlas_chart_normal_cones.free();
    atlas_chart_adj.free();
    atlas_chart_adj_length.free();
    atlas_chart_perims.free();
    atlas_chart_areas.free();
    atlas_chart2edge.free();
    atlas_chart2edge_cnt.free();
    atlas_chart2edge_offset.free();

    temp_storage.free();
    cub_temp_storage.free();
}

int CuMesh::num_vertices() const {
    return vertices.size;
}

int CuMesh::num_faces() const {
    return faces.size;
}

int CuMesh::num_edges() const {
    return edges.size;
}

int CuMesh::num_boundaries() const {
    return boundaries.size;
}

int CuMesh::num_conneted_components() const {
    return num_conn_comps;
}

int CuMesh::num_boundary_conneted_components() const {
    return num_bound_conn_comps;
}

int CuMesh::num_boundary_loops() const {
    return num_bound_loops;
}

void CuMesh::clear_cache() {
    face_areas.free();
    face_normals.free();
    vertex_normals.free();
    edges.free();
    boundaries.free();
    vert_is_boundary.free();
    vert_is_manifold.free();
    vert2edge.free();
    vert2edge_cnt.free();
    vert2edge_offset.free();
    vert2bound.free();
    vert2bound_cnt.free();
    vert2bound_offset.free();
    edge2face.free();
    edge2face_cnt.free();
    edge2face_offset.free();
    face2edge.free();
    vert2face.free();
    vert2face_cnt.free();
    vert2face_offset.free();
    manifold_face_adj.free();
    manifold_bound_adj.free();
    conn_comp_ids.free();
    bound_conn_comp_ids.free();
    loop_boundaries.free();
    loop_boundaries_offset.free();
    vertices_map.free();
    faces_map.free();
    edge_collapse_costs.free();
    propagated_costs.free();

    atlas_chart_ids.free();
    atlas_chart_vertex_map.free();
    atlas_chart_faces.free();
    atlas_chart_faces_offset.free();
    atlas_chart_vertex_offset.free();
    atlas_chart_uvs.free();

    atlas_chart_normal_cones.free();
    atlas_chart_adj.free();
    atlas_chart_adj_length.free();
    atlas_chart_perims.free();
    atlas_chart_areas.free();
    atlas_chart2edge.free();
    atlas_chart2edge_cnt.free();
    atlas_chart2edge_offset.free();

    temp_storage.free();
    cub_temp_storage.free();
}

} // namespace cumesh
