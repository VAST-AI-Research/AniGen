// #include <Eigen/Dense>
#include <gpu/common.h>
#include <gpu/triangle.cuh>
#include <gpu/bvh.cuh>
#include <gpu/pcg32.h>

#include <stack>
#include <iostream>
#include <cstdio>
#include <functional>
#include <limits>

using namespace Eigen;
using namespace cubvh;


namespace cubvh {

constexpr float MAX_DIST = 1000.0f;
constexpr float MAX_DIST_SQ = MAX_DIST*MAX_DIST;

__global__ void signed_distance_watertight_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds);
__global__ void signed_distance_raystab_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds);
__global__ void unsigned_distance_kernel(uint32_t n_elements, const Vector3f* __restrict__ positions, float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds);
__global__ void raytrace_kernel(uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles);

struct DistAndIdx {
    float dist;
    uint32_t idx;

    // Sort in descending order!
    __host__ __device__ bool operator<(const DistAndIdx& other) {
        return dist < other.dist;
    }
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1}; t1 = t2; t2 = tmp;
    }
}

// Sorting networks from http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html#N4L5D3
template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);

        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);

        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);

        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);

        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);

        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);

        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

template <uint32_t BRANCHING_FACTOR>
class TriangleBvhWithBranchingFactor : public TriangleBvh {
public:
    // Stackless traversal using threaded BVH (escape links). Assumes escape_idx is populated.
    __host__ __device__ static std::pair<int, float> ray_intersect_stackless(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        float mint = MAX_DIST;
        int shortest_idx = -1;

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float tbb = node.bb.ray_intersect(ro, rd).x();
            if (tbb >= mint) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                // descend first child; siblings are reached via escape links
                idx = node.left_idx;
            }
        }

        return {shortest_idx, mint};
    }

    __host__ __device__ static std::pair<int, float> closest_triangle_stackless(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        float shortest_distance_sq = max_distance_sq;
        int shortest_idx = -1;

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float dbb = node.bb.distance_sq(point);
            if (dbb > shortest_distance_sq) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float dist_sq = triangles[i].distance_sq(point);
                    if (dist_sq <= shortest_distance_sq) {
                        shortest_distance_sq = dist_sq;
                        shortest_idx = i;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }

        if (shortest_idx == -1) {
            shortest_idx = 0;
            shortest_distance_sq = 0.0f;
        }

        return {shortest_idx, std::sqrt(shortest_distance_sq)};
    }

    // For normal averaging around a point on the surface. Returns unnormalized normal.
    __host__ __device__ static Vector3f avg_normal_around_point_stackless(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        static constexpr float EPSILON = 1e-6f;

        float total_weight = 0;
        Vector3f result = Vector3f::Zero();

        int idx = 0;
        while (idx != -1) {
            const TriangleBvhNode& node = bvhnodes[idx];

            float dbb = node.bb.distance_sq(point);
            if (dbb >= EPSILON) {
                idx = node.escape_idx;
                continue;
            }

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    if (triangles[i].distance_sq(point) < EPSILON) {
                        float weight = 1; // TODO: cot weight
                        result += triangles[i].normal();
                        total_weight += weight;
                    }
                }
                idx = node.escape_idx;
            } else {
                idx = node.left_idx;
            }
        }

        return result / total_weight;
    }

    __host__ __device__ static std::pair<int, float> ray_intersect(Ref<const Vector3f> ro, Ref<const Vector3f> rd, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
        FixedIntStack query_stack;
        query_stack.push(0);

        float mint = MAX_DIST;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                int first_child = node.left_idx;
                int end_child = node.right_idx;
                int child_count = end_child - first_child;

                if (child_count <= 0 || child_count > static_cast<int>(BRANCHING_FACTOR)) {
                    return ray_intersect_stackless(ro, rd, bvhnodes, triangles);
                }

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    int child_idx = first_child + static_cast<int>(i);
                    if (child_idx < end_child) {
                        children[i] = {bvhnodes[child_idx].bb.ray_intersect(ro, rd).x(), static_cast<uint32_t>(child_idx)};
                    } else {
                        children[i] = {std::numeric_limits<float>::infinity(), UINT32_MAX};
                    }
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (int i = (int)BRANCHING_FACTOR - 1; i >= 0; --i) {
                    uint32_t child_idx = children[i].idx;
                    if (child_idx == UINT32_MAX) {
                        continue;
                    }
                    if (children[i].dist < mint) {
                        query_stack.push(static_cast<int>(child_idx));
                    }
                }
            }

            if (query_stack.overflowed()) {
                // Fallback to stackless traversal to guarantee correctness
                return ray_intersect_stackless(ro, rd, bvhnodes, triangles);
            }
        }

        return {shortest_idx, mint};
    }

    __host__ __device__ static std::pair<int, float> closest_triangle(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        FixedIntStack query_stack;
        query_stack.push(0);

        float shortest_distance_sq = max_distance_sq;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    float dist_sq = triangles[i].distance_sq(point);
                    if (dist_sq <= shortest_distance_sq) {
                        shortest_distance_sq = dist_sq;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];

                int first_child = node.left_idx;
                int end_child = node.right_idx;
                int child_count = end_child - first_child;

                if (child_count <= 0 || child_count > static_cast<int>(BRANCHING_FACTOR)) {
                    return closest_triangle_stackless(point, bvhnodes, triangles, shortest_distance_sq);
                }

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    int child_idx = first_child + static_cast<int>(i);
                    if (child_idx < end_child) {
                        children[i] = {bvhnodes[child_idx].bb.distance_sq(point), static_cast<uint32_t>(child_idx)};
                    } else {
                        children[i] = {std::numeric_limits<float>::infinity(), UINT32_MAX};
                    }
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (int i = (int)BRANCHING_FACTOR - 1; i >= 0; --i) {
                    uint32_t child_idx = children[i].idx;
                    if (child_idx == UINT32_MAX) {
                        continue;
                    }
                    if (children[i].dist <= shortest_distance_sq) {
                        query_stack.push(static_cast<int>(child_idx));
                    }
                }
            }

            if (query_stack.overflowed()) {
                return closest_triangle_stackless(point, bvhnodes, triangles, shortest_distance_sq);
            }
        }

        if (shortest_idx == -1) {
            // printf("No closest triangle found. This must be a bug! %d\n", BRANCHING_FACTOR);
            shortest_idx = 0;
            shortest_distance_sq = 0.0f;
        }

        return {shortest_idx, std::sqrt(shortest_distance_sq)};
    }

    // Assumes that "point" is a location on a triangle
    __host__ __device__ static Vector3f avg_normal_around_point(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles) {
    FixedIntStack query_stack;
    query_stack.push(0);

        static constexpr float EPSILON = 1e-6f;

        float total_weight = 0;
        Vector3f result = Vector3f::Zero();

        while (!query_stack.empty()) {
            int idx = query_stack.pop();

            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx-1;
                for (int i = -node.left_idx-1; i < end; ++i) {
                    if (triangles[i].distance_sq(point) < EPSILON) {
                        float weight = 1; // TODO: cot weight
                        result += triangles[i].normal();
                        total_weight += weight;
                    }
                }
            } else {
                int first_child = node.left_idx;
                int end_child = node.right_idx;

                for (int child_idx = first_child; child_idx < end_child; ++child_idx) {
                    if (bvhnodes[child_idx].bb.distance_sq(point) < EPSILON) {
                        query_stack.push(child_idx);
                    }
                }
            }
        }

        if (query_stack.overflowed()) {
            return avg_normal_around_point_stackless(point, bvhnodes, triangles);
        }

        return result / total_weight;
    }

    __host__ __device__ static std::pair<int, float> signed_distance_watertight(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ) {
        auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

        const Triangle& tri = triangles[res.first];
        Vector3f closest_point = tri.closest_point(point);
        Vector3f avg_normal = avg_normal_around_point(closest_point, bvhnodes, triangles);

        return {res.first, std::copysignf(res.second, avg_normal.dot(point - closest_point))};
    }

    __host__ __device__ static std::pair<int, float> signed_distance_raystab(const Vector3f& point, const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, float max_distance_sq = MAX_DIST_SQ, pcg32 rng={}) {
        auto res = closest_triangle(point, bvhnodes, triangles, max_distance_sq);

        Vector2f offset = {rng.next_float(), rng.next_float()};

        static constexpr uint32_t N_STAB_RAYS = 32;
        for (uint32_t i = 0; i < N_STAB_RAYS; ++i) {
            // Use a Fibonacci lattice (with random offset) to regularly
            // distribute the stab rays over the sphere.
            // ref: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
            Vector3f d = fibonacci_dir<N_STAB_RAYS>(i, offset);

            // If any of the stab rays goes outside the mesh, the SDF is positive.
            if (ray_intersect(point, -d, bvhnodes, triangles).first < 0 || ray_intersect(point, d, bvhnodes, triangles).first < 0) {
                return {res.first, res.second};
            }
        }

        return {res.first, -res.second};
    }


    void signed_distance_gpu(uint32_t n_elements, uint32_t mode, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, uint32_t n_triangles, cudaStream_t stream) override {

        const Vector3f* positions_vec = (const Vector3f*)positions;
        Vector3f* uvw_vec = (Vector3f*)uvw;

        if (mode == 0) {
            // watertight
            linear_kernel(signed_distance_watertight_kernel, 0u, stream,
                n_elements,
                positions_vec,
                distances,
                face_id,
                uvw_vec,
                m_nodes_gpu.data(),
                gpu_triangles,
                n_triangles,
                false
            );

        } else {
            // raystab
            linear_kernel(signed_distance_raystab_kernel, 0u, stream,
                n_elements,
                positions_vec,
                distances,
                face_id,
                uvw_vec,
                m_nodes_gpu.data(),
                gpu_triangles,
                n_triangles,
                false
            );
        }
    }

    void unsigned_distance_gpu(uint32_t n_elements, const float* positions, float* distances, int64_t* face_id, float* uvw, const Triangle* gpu_triangles, uint32_t n_triangles, cudaStream_t stream) override {

        const Vector3f* positions_vec = (const Vector3f*)positions;
        Vector3f* uvw_vec = (Vector3f*)uvw;

        linear_kernel(unsigned_distance_kernel, 0u, stream,
            n_elements,
            positions_vec,
            distances,
            face_id,
            uvw_vec,
            m_nodes_gpu.data(),
            gpu_triangles,
            n_triangles,
            false
        );
    }

    void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d, float* positions, int64_t* face_id, float* depth, const Triangle* gpu_triangles, cudaStream_t stream) override {

        // cast float* to Vector3f*
        const Vector3f* rays_o_vec = (const Vector3f*)rays_o;
        const Vector3f* rays_d_vec = (const Vector3f*)rays_d;
        Vector3f* positions_vec = (Vector3f*)positions;
        
        linear_kernel(raytrace_kernel, 0u, stream,
            n_elements,
            rays_o_vec,
            rays_d_vec,
            positions_vec,
            face_id,
            depth,
            m_nodes_gpu.data(),
            gpu_triangles
        );

    }

    void build(std::vector<Triangle>& triangles, uint32_t n_primitives_per_leaf) override {
        m_nodes.clear();

        // Root
        m_nodes.emplace_back();
        m_nodes.front().bb = BoundingBox(std::begin(triangles), std::end(triangles));

        struct BuildNode {
            int node_idx;
            std::vector<Triangle>::iterator begin;
            std::vector<Triangle>::iterator end;
        };

        std::stack<BuildNode> build_stack;
        build_stack.push({0, std::begin(triangles), std::end(triangles)});

    while (!build_stack.empty()) {
            const BuildNode& curr = build_stack.top();
            size_t node_idx = curr.node_idx;

            std::array<BuildNode, BRANCHING_FACTOR> children;
            children[0].begin = curr.begin;
            children[0].end = curr.end;

            build_stack.pop();

            // Partition the triangles into the children
            int n_children = 1;
            while (n_children < BRANCHING_FACTOR) {
                for (int i = n_children - 1; i >= 0; --i) {
                    auto& child = children[i];

                    // Choose axis with maximum standard deviation
                    Vector3f mean = Vector3f::Zero();
                    for (auto it = child.begin; it != child.end; ++it) {
                        mean += it->centroid();
                    }
                    mean /= (float)std::distance(child.begin, child.end);

                    Vector3f var = Vector3f::Zero();
                    for (auto it = child.begin; it != child.end; ++it) {
                        Vector3f diff = it->centroid() - mean;
                        var += diff.cwiseProduct(diff);
                    }
                    var /= (float)std::distance(child.begin, child.end);

                    Vector3f::Index axis;
                    var.maxCoeff(&axis);

                    auto m = child.begin + std::distance(child.begin, child.end)/2;
                    std::nth_element(child.begin, m, child.end, [&](const Triangle& tri1, const Triangle& tri2) { return tri1.centroid(axis) < tri2.centroid(axis); });

                    children[i*2].begin = children[i].begin;
                    children[i*2+1].end = children[i].end;
                    children[i*2].end = children[i*2+1].begin = m;
                }

                n_children *= 2;
            }

            // Create next build nodes
            m_nodes[node_idx].left_idx = (int)m_nodes.size();
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                auto& child = children[i];
                assert(child.begin != child.end);
                child.node_idx = (int)m_nodes.size();

                m_nodes.emplace_back();
                m_nodes.back().bb = BoundingBox(child.begin, child.end);

                if (std::distance(child.begin, child.end) <= n_primitives_per_leaf) {
                    m_nodes.back().left_idx = -(int)std::distance(std::begin(triangles), child.begin)-1;
                    m_nodes.back().right_idx = -(int)std::distance(std::begin(triangles), child.end)-1;
                } else {
                    build_stack.push(child);
                }
            }
            m_nodes[node_idx].right_idx = (int)m_nodes.size();
        }

        // Thread the BVH with escape links for stackless traversal.
        // Initialize all escape indices to -1.
        for (auto& n : m_nodes) n.escape_idx = -1;

        // Recursive lambda to assign escape links in pre-order (child order 0..BRANCHING_FACTOR-1)
        std::function<void(int,int)> thread_bvh = [&](int node_idx, int escape_idx) {
            TriangleBvhNode& node = m_nodes[node_idx];
            node.escape_idx = escape_idx;
            if (node.left_idx < 0) return; // leaf
            int first_child = node.left_idx;
            int end_child = node.right_idx; // exclusive
            for (int c = first_child; c < end_child; ++c) {
                int next_escape = (c+1 < end_child) ? (c+1) : escape_idx;
                thread_bvh(c, next_escape);
            }
        };

        if (!m_nodes.empty()) {
            thread_bvh(0, -1);
        }

        m_nodes_gpu.resize_and_copy_from_host(m_nodes);

        // std::cout << "[INFO] Built TriangleBvh: nodes=" << m_nodes.size() << std::endl;
    }

    TriangleBvhWithBranchingFactor() {}
};

using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
}

const std::vector<TriangleBvhNode>& TriangleBvh::host_nodes() const {
    return m_nodes;
}

void TriangleBvh::set_nodes(const std::vector<TriangleBvhNode>& nodes) {
    m_nodes = nodes;

    if (!m_nodes.empty()) {
        for (auto& node : m_nodes) {
            node.escape_idx = -1;
        }

        std::function<void(int, int)> thread_bvh = [&](int node_idx, int escape_idx) {
            if (node_idx < 0 || node_idx >= static_cast<int>(m_nodes.size())) {
                return;
            }

            TriangleBvhNode& node = m_nodes[node_idx];
            node.escape_idx = escape_idx;

            if (node.left_idx < 0) {
                return;
            }

            int first_child = node.left_idx;
            int end_child = node.right_idx;

            if (first_child < 0 || first_child >= static_cast<int>(m_nodes.size())) {
                return;
            }

            if (end_child <= first_child) {
                return;
            }

            if (end_child > static_cast<int>(m_nodes.size())) {
                end_child = static_cast<int>(m_nodes.size());
            }

            for (int c = first_child; c < end_child; ++c) {
                int next_escape = (c + 1 < end_child) ? (c + 1) : escape_idx;
                thread_bvh(c, next_escape);
            }
        };

        thread_bvh(0, -1);
    }

    m_nodes_gpu.resize_and_copy_from_host(m_nodes);
}

__global__ void signed_distance_watertight_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;

    Vector3f point = positions[i];

    auto res = TriangleBvh4::signed_distance_watertight(point, bvhnodes, triangles, max_distance*max_distance);

    // write 
    distances[i] = res.second;
    int tri_idx = res.first;
    if (tri_idx < 0 || tri_idx >= static_cast<int>(n_triangles)) {
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }

    face_id[i] = triangles[tri_idx].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[res.first].closest_point(point);
        // query uvw
        uvw[i] = triangles[tri_idx].barycentric(cpoint);
    }
}

__global__ void signed_distance_raystab_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;
    pcg32 rng;
    rng.advance(i * 2);

    Vector3f point = positions[i];

    auto res = TriangleBvh4::signed_distance_raystab(point, bvhnodes, triangles, max_distance*max_distance, rng);

    // write 
    distances[i] = res.second;
    int tri_idx = res.first;
    if (tri_idx < 0 || tri_idx >= static_cast<int>(n_triangles)) {
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }

    face_id[i] = triangles[tri_idx].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[res.first].closest_point(point);
        // query uvw
        uvw[i] = triangles[tri_idx].barycentric(cpoint);
    }
}

__global__ void unsigned_distance_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ positions,
    float* __restrict__ distances, int64_t* __restrict__ face_id, Vector3f* __restrict__ uvw,
    const TriangleBvhNode* __restrict__ bvhnodes, const Triangle* __restrict__ triangles, uint32_t n_triangles, bool use_existing_distances_as_upper_bounds
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    float max_distance = use_existing_distances_as_upper_bounds ? distances[i] : MAX_DIST;

    Vector3f point = positions[i];

    auto res = TriangleBvh4::closest_triangle(point, bvhnodes, triangles, max_distance*max_distance);

    // write 
    distances[i] = res.second;
    int tri_idx = res.first;
    if (tri_idx < 0 || tri_idx >= static_cast<int>(n_triangles)) {
        face_id[i] = -1;
        if (uvw) {
            uvw[i] = Vector3f::Zero();
        }
        return;
    }

    face_id[i] = triangles[tri_idx].id;

    // optional output
    if (uvw) {
        // get closest point
        Vector3f cpoint = triangles[tri_idx].closest_point(point);
        // query uvw
        uvw[i] = triangles[tri_idx].barycentric(cpoint);
    }
}

__global__ void raytrace_kernel(
    uint32_t n_elements, const Vector3f* __restrict__ rays_o, const Vector3f* __restrict__ rays_d, 
    Vector3f* __restrict__ positions, int64_t* __restrict__ face_id, float* __restrict__ depth, 
    const TriangleBvhNode* __restrict__ nodes, const Triangle* __restrict__ triangles
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Vector3f ro = rays_o[i];
    Vector3f rd = rays_d[i];

    auto res = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles);

    // write depth
    depth[i] = res.second;
 
    // intersection point is written back to positions.
    // non-intersect point reaches at most 10 depth
    positions[i] = ro + res.second * rd;

    // write face_id
    if (res.first >= 0) {
        face_id[i] = triangles[res.first].id;
    } else {
        face_id[i] = -1;
    }
}
    
}