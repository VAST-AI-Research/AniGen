#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

#include <Eigen/Dense>

namespace cubvh {
namespace cpu {

using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;

struct HoleFillOptions {
    // If true, run a point-in-triangle test to avoid creating triangles
    // that contain other boundary vertices. Slightly slower but safer.
    bool checkContainment = true;
    // Numerical tolerance when checking convexity/containment in 2D.
    float eps = 1e-8f;
    // Safety bound to avoid infinite loops on degenerate inputs.
    int maxWalk = 100000;
    // If true, print summary and per-ear logs during hole filling.
    bool verbose = false;
};

namespace detail {

inline uint64_t pack_undirected(int a, int b) {
    if (a > b) std::swap(a, b);
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

inline uint64_t pack_directed(int a, int b) {
    return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

// Build boundary adjacency (outgoing directed boundary edges per vertex) and list of all boundary directed edges.
inline void build_boundary(const std::vector<Vec3i>& F,
                           std::unordered_map<int, std::vector<int>>& out_adj,
                           std::vector<std::pair<int,int>>& boundary_dir_edges) {
    std::unordered_map<uint64_t, int> undirected_count;
    undirected_count.reserve(F.size() * 3);

    // First pass: count undirected edges
    for (const auto& f : F) {
        int a = f[0], b = f[1], c = f[2];
        uint64_t kab = pack_undirected(a,b);
        uint64_t kbc = pack_undirected(b,c);
        uint64_t kca = pack_undirected(c,a);
        ++undirected_count[kab];
        ++undirected_count[kbc];
        ++undirected_count[kca];
    }

    // Second pass: collect boundary directed edges
    out_adj.clear();
    boundary_dir_edges.clear();
    out_adj.reserve(undirected_count.size());

    auto add_dir_if_boundary = [&](int u, int v) {
        if (undirected_count[pack_undirected(u,v)] == 1) {
            out_adj[u].push_back(v);
            boundary_dir_edges.emplace_back(u, v);
        }
    };

    for (const auto& f : F) {
        int a = f[0], b = f[1], c = f[2];
        add_dir_if_boundary(a,b);
        add_dir_if_boundary(b,c);
        add_dir_if_boundary(c,a);
    }
}

// Extract closed boundary loops by walking boundary directed edges.
inline std::vector<std::vector<int>> extract_loops(
    const std::unordered_map<int, std::vector<int>>& out_adj,
    const std::vector<std::pair<int,int>>& boundary_dir_edges,
    const HoleFillOptions& opt) {

    std::unordered_set<uint64_t> visited;
    visited.reserve(boundary_dir_edges.size()*2);
    std::vector<std::vector<int>> loops;
    loops.reserve(boundary_dir_edges.size() / 3);

    for (const auto& e : boundary_dir_edges) {
        int start = e.first;
        int next  = e.second;
        uint64_t key = pack_directed(start, next);
        if (visited.find(key) != visited.end()) continue;

        std::vector<int> loop;
        loop.reserve(64);
        loop.push_back(start);

        int prev = start;
        int curr = next;
        visited.insert(key);

        int steps = 0;
        bool closed = false;
        while (steps++ < opt.maxWalk) {
            loop.push_back(curr);
            if (curr == start) { closed = true; break; }

            auto it = out_adj.find(curr);
            if (it == out_adj.end() || it->second.empty()) break; // dead end

            // Prefer edge that doesn't go back to prev; if multiple exist, pick the first unvisited.
            const auto& outs = it->second;
            int pick = -1;
            for (int v : outs) {
                if (v == prev) continue;
                uint64_t k2 = pack_directed(curr, v);
                if (visited.find(k2) == visited.end()) { pick = v; break; }
            }
            if (pick == -1) {
                // fallback: if only back edge available, use it once to try to close
                for (int v : outs) { pick = v; break; }
            }
            if (pick == -1) break;

            visited.insert(pack_directed(curr, pick));
            prev = curr;
            curr = pick;
        }

        if (closed && loop.size() > 2) {
            // Remove duplicated last == first if present
            if (!loop.empty() && loop.front() == loop.back()) loop.pop_back();
            // Filter out tiny loops or degenerate duplicates
            bool ok = true;
            if ((int)loop.size() < 3) ok = false;
            if (ok) loops.emplace_back(std::move(loop));
        }
    }

    return loops;
}

// Fit a best-fit plane and project points to 2D
inline void project_to_plane(const std::vector<Vec3f>& V, const std::vector<int>& loop,
                             std::vector<Eigen::Vector2f>& P2, Eigen::Vector3f& n_out) {
    // Compute centroid
    Eigen::Vector3f c(0,0,0);
    for (int vid : loop) c += V[vid];
    c /= float(loop.size());

    // Covariance
    Eigen::Matrix3f C = Eigen::Matrix3f::Zero();
    for (int vid : loop) {
        Eigen::Vector3f d = V[vid] - c;
        C += d * d.transpose();
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(C);
    // Normal is eigenvector with smallest eigenvalue
    Eigen::Vector3f n = es.eigenvectors().col(0);
    n.normalize();
    n_out = n;

    // Build an orthonormal basis (u,v) on the plane
    Eigen::Vector3f t = (std::abs(n.x()) < 0.9f) ? Eigen::Vector3f(1,0,0) : Eigen::Vector3f(0,1,0);
    Eigen::Vector3f u = (t - t.dot(n) * n).normalized();
    Eigen::Vector3f v = n.cross(u);

    P2.resize(loop.size());
    for (size_t i = 0; i < loop.size(); ++i) {
        const Eigen::Vector3f& p = V[loop[i]];
        Eigen::Vector3f d = p - c;
        P2[i] = Eigen::Vector2f(d.dot(u), d.dot(v));
    }
}

inline float polygon_signed_area_2d(const std::vector<Eigen::Vector2f>& P) {
    double a = 0.0;
    size_t n = P.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        a += double(P[j].x()) * double(P[i].y()) - double(P[i].x()) * double(P[j].y());
    }
    return float(0.5 * a);
}

inline bool point_in_triangle_2d(const Eigen::Vector2f& p,
                                 const Eigen::Vector2f& a,
                                 const Eigen::Vector2f& b,
                                 const Eigen::Vector2f& c,
                                 float eps) {
    // Barycentric technique
    Eigen::Vector2f v0 = b - a;
    Eigen::Vector2f v1 = c - a;
    Eigen::Vector2f v2 = p - a;
    float d00 = v0.dot(v0);
    float d01 = v0.dot(v1);
    float d11 = v1.dot(v1);
    float d20 = v2.dot(v0);
    float d21 = v2.dot(v1);
    float denom = d00 * d11 - d01 * d01;
    if (std::abs(denom) < eps) return false; // degenerate
    float v = (d11 * d20 - d01 * d21) / denom;
    float w = (d00 * d21 - d01 * d20) / denom;
    float u = 1.0f - v - w;
    return u >= -eps && v >= -eps && w >= -eps;
}

inline float angle_score(const Eigen::Vector2f& pm,
                         const Eigen::Vector2f& p,
                         const Eigen::Vector2f& pp) {
    Eigen::Vector2f d0 = p - pm;
    Eigen::Vector2f d1 = pp - p;
    float cross_z = d0.x() * d1.y() - d0.y() * d1.x();
    float dot = d0.dot(d1);
    return std::atan2(std::abs(cross_z), -dot);
}

} // namespace detail

// Fill holes in-place: append new triangles to F using indices into V.
inline void fill_holes_inplace(const std::vector<Vec3f>& V,
                               std::vector<Vec3i>& F,
                               const HoleFillOptions& opt = {}) {
    using namespace detail;

    // 1) Build boundary adjacency & collect directed boundary edges
    std::unordered_map<int, std::vector<int>> out_adj;
    std::vector<std::pair<int,int>> boundary_dir_edges;
    build_boundary(F, out_adj, boundary_dir_edges);
    if (boundary_dir_edges.empty()) {
        if (opt.verbose) {
            std::cout << "[fill_holes] No boundary edges found. Nothing to fill.\n";
        }
        return;
    }

    // Build directed edge -> face index map and face normals for orientation alignment
    std::unordered_map<uint64_t, int> edge2face;
    edge2face.reserve(F.size() * 3);
    std::vector<Eigen::Vector3f> face_normals(F.size());
    for (size_t fi = 0; fi < F.size(); ++fi) {
        const auto& f = F[fi];
        int a = f[0], b = f[1], c = f[2];
        Eigen::Vector3f n = (V[b] - V[a]).cross(V[c] - V[a]); // respect input winding
        face_normals[fi] = n;
        edge2face[pack_directed(a,b)] = (int)fi;
        edge2face[pack_directed(b,c)] = (int)fi;
        edge2face[pack_directed(c,a)] = (int)fi;
    }

    // 2) Extract closed loops
    auto loops = extract_loops(out_adj, boundary_dir_edges, opt);
    if (opt.verbose) {
        std::cout << "[fill_holes] Found " << loops.size() << " boundary loop(s).\n";
    }
    if (loops.empty()) return;

    // 3) Triangulate each loop with ear clipping prioritizing sharp convex ears
    for (size_t li = 0; li < loops.size(); ++li) {
        const auto& loop = loops[li];
        if (opt.verbose) {
            std::cout << "[fill_holes] Loop " << li << " size: " << loop.size() << "\n";
        }
        if (loop.size() < 3) continue;

        // Project to 2D for robust ear clipping
        std::vector<Eigen::Vector2f> P2;
        Eigen::Vector3f n_plane;
        project_to_plane(V, loop, P2, n_plane);

        float area = polygon_signed_area_2d(P2);
        // if (std::abs(area) < opt.eps) {
        //     if (opt.verbose) {
        //         std::cout << "[fill_holes] Loop " << li << " nearly degenerate. Skipping.\n";
        //     }
        //     continue; // nearly degenerate
        // }
        float orient = (area > 0.0f) ? 1.0f : -1.0f; // CCW positive

        // Estimate desired orientation from adjacent face normals along the boundary loop
        Eigen::Vector3f target_n(0,0,0);
        int L = (int)loop.size();
        for (int i = 0; i < L; ++i) {
            int u = loop[i];
            int v = loop[(i + 1) % L];
            auto it = edge2face.find(pack_directed(u, v));
            if (it != edge2face.end()) {
                target_n += face_normals[it->second];
            } else {
                auto it2 = edge2face.find(pack_directed(v, u));
                if (it2 != edge2face.end()) target_n += face_normals[it2->second];
            }
        }
        if (target_n.squaredNorm() == 0.0f) target_n = n_plane;
        target_n.normalize();

        auto fix_winding = [&](Vec3i& tri) {
            const Eigen::Vector3f& A = V[tri[0]];
            const Eigen::Vector3f& B = V[tri[1]];
            const Eigen::Vector3f& C = V[tri[2]];
            Eigen::Vector3f ntri = (B - A).cross(C - A);
            if (ntri.dot(target_n) < 0.0f) std::swap(tri[1], tri[2]);
        };

        // Indices into 'loop' / P2 for current polygon boundary
        std::vector<int> idx(loop.size());
        for (size_t i = 0; i < loop.size(); ++i) idx[i] = int(i);

        auto is_convex = [&](int i)->bool {
            int nvert = (int)idx.size();
            int i0 = idx[(i - 1 + nvert) % nvert];
            int i1 = idx[i];
            int i2 = idx[(i + 1) % nvert];
            const auto& a = P2[i0];
            const auto& b = P2[i1];
            const auto& c = P2[i2];
            Eigen::Vector2f ab = b - a;
            Eigen::Vector2f bc = c - b;
            float cz = ab.x() * bc.y() - ab.y() * bc.x();
            return orient * cz > 0.0f; // same sign as polygon orientation
        };

        auto triangle_contains_no_other = [&](int i)->bool {
            if (!opt.checkContainment) return true;
            int nvert = (int)idx.size();
            int i0 = idx[(i - 1 + nvert) % nvert];
            int i1 = idx[i];
            int i2 = idx[(i + 1) % nvert];
            const auto& a = P2[i0];
            const auto& b = P2[i1];
            const auto& c = P2[i2];
            for (int k = 0; k < nvert; ++k) {
                if (k == (i - 1 + nvert) % nvert || k == i || k == (i + 1) % nvert) continue;
                if (point_in_triangle_2d(P2[idx[k]], a, b, c, opt.eps)) return false;
            }
            return true;
        };

        // Repeatedly clip ears
        int guard = 0;
        while (idx.size() >= 3 && guard++ < opt.maxWalk) {
            if (idx.size() == 3) {
                // Final triangle
                Vec3i tri(loop[idx[0]], loop[idx[1]], loop[idx[2]]);
                fix_winding(tri);
                F.emplace_back(tri);
                if (opt.verbose) {
                    std::cout << "[fill_holes] Loop " << li << " add final tri: ("
                              << tri[0] << ", " << tri[1] << ", " << tri[2] << ")\n";
                }
                break;
            }

            int nvert = (int)idx.size();
            int best_i = -1;
            float best_score = std::numeric_limits<float>::infinity();

            for (int i = 0; i < nvert; ++i) {
                if (!is_convex(i)) continue;
                if (!triangle_contains_no_other(i)) continue;
                int ip = idx[(i - 1 + nvert) % nvert];
                int ic = idx[i];
                int in = idx[(i + 1) % nvert];
                float score = angle_score(P2[ip], P2[ic], P2[in]);
                if (score < best_score) { best_score = score; best_i = i; }
            }

            if (best_i == -1) {
                // Fallback: allow any convex vertex (skip containment)
                for (int i = 0; i < nvert; ++i) {
                    if (!is_convex(i)) continue;
                    int ip = idx[(i - 1 + nvert) % nvert];
                    int ic = idx[i];
                    int in = idx[(i + 1) % nvert];
                    float score = angle_score(P2[ip], P2[ic], P2[in]);
                    if (score < best_score) { best_score = score; best_i = i; }
                }
            }

            if (best_i == -1) {
                // As a last resort, clip the first available ear to make progress
                int i = 0;
                int ip = idx[(i - 1 + nvert) % nvert];
                int ic = idx[i];
                int in = idx[(i + 1) % nvert];
                Vec3i tri(loop[ip], loop[ic], loop[in]);
                fix_winding(tri);
                F.emplace_back(tri);
                if (opt.verbose) {
                    std::cout << "[fill_holes] Loop " << li << " add fallback tri: ("
                              << tri[0] << ", " << tri[1] << ", " << tri[2] << ")\n";
                }
                idx.erase(idx.begin() + i);
                continue;
            }

            int ip = idx[(best_i - 1 + nvert) % nvert];
            int ic = idx[best_i];
            int in = idx[(best_i + 1) % nvert];
            Vec3i tri(loop[ip], loop[ic], loop[in]);
            fix_winding(tri);
            F.emplace_back(tri);
            if (opt.verbose) {
                std::cout << "[fill_holes] Loop " << li << " add tri: ("
                          << tri[0] << ", " << tri[1] << ", " << tri[2] << ")\n";
            }
            idx.erase(idx.begin() + best_i);
        }
    }
}

// Return the new triangles that would be added (without modifying input F)
inline std::vector<Vec3i> fill_holes(const std::vector<Vec3f>& V,
                                     const std::vector<Vec3i>& F,
                                     const HoleFillOptions& opt = {}) {
    std::vector<Vec3i> F_all = F; // copy
    fill_holes_inplace(V, F_all, opt);
    // Return only the newly-added ones
    std::vector<Vec3i> added;
    if (F_all.size() > F.size()) {
        added.insert(added.end(), F_all.begin() + (ptrdiff_t)F.size(), F_all.end());
    }
    return added;
}

} // namespace cpu
} // namespace cubvh
