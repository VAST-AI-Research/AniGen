#include "shared.h"


namespace cumesh {


/**
 * Hook edges
 * @param adj: the buffer for adjacency, shape (M)
 * @param M: the number of adjacency
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param end_flag: flag to indicate if any union operation happened
 */
__global__ void hook_edges_kernel(
    const int2* adj,
    const int M,
    int* conn_comp_ids,
    int* end_flag
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= M) return;

    // get adjacent faces
    int f0 = adj[tid].x;
    int f1 = adj[tid].y;

    // union
    // find roots
    int root0 = conn_comp_ids[f0];
    while (root0 != conn_comp_ids[root0]) {
        root0 = conn_comp_ids[root0];
    }
    int root1 = conn_comp_ids[f1];
    while (root1 != conn_comp_ids[root1]) {
        root1 = conn_comp_ids[root1];
    }

    if (root0 == root1) return;

    int high = max(root0, root1);
    int low = min(root0, root1);
    atomicMin(&conn_comp_ids[high], low);
    *end_flag = 0;
}


/**
 * Compress connected components
 * @param conn_comp_ids: the buffer for connected component ids, shape (F)
 * @param F: the number of faces
 */
__global__ void compress_components_kernel(
    int* conn_comp_ids,
    const int F
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= F) return;

    int p = conn_comp_ids[tid];
    while (p != conn_comp_ids[p]) {
        p = conn_comp_ids[p];
    }
    conn_comp_ids[tid] = p;
}


} // namespace cumesh