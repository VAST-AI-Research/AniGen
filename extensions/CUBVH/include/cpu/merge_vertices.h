#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <numeric>

#include <Eigen/Dense>

namespace cubvh {
namespace cpu {

using Vec3f = Eigen::Vector3f;
using Vec3i = Eigen::Vector3i;

// Merge vertices of a mesh that are closer than a given threshold.
// Simple uniform grid hashing + Union-Find. Produces averaged vertex positions.
// Degenerate faces (with repeated vertex indices) are removed; duplicate faces (unordered) removed.
inline void merge_vertices(
    const std::vector<Vec3f>& V_in,
    const std::vector<Vec3i>& F_in,
    float threshold,
    std::vector<Vec3f>& V_out,
    std::vector<Vec3i>& F_out) {

    const size_t N = V_in.size();
    if (N == 0) { V_out.clear(); F_out.clear(); return; }
    if (threshold <= 0.0f) { V_out = V_in; F_out = F_in; return; }
    const float thresh2 = threshold * threshold;

    // DSU
    std::vector<int> parent(N), rankv(N,0);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x){ while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    auto unite = [&](int a, int b){ a = find(a); b = find(b); if (a==b) return; if (rankv[a] < rankv[b]) std::swap(a,b); parent[b]=a; if (rankv[a]==rankv[b]) ++rankv[a]; };

    // Spatial hashing (uniform grid cells of size = threshold)
    struct KeyHash { size_t operator()(const int64_t k) const noexcept { return std::hash<int64_t>()(k); } };
    std::unordered_map<int64_t, std::vector<int>, KeyHash> grid; grid.reserve(N*2);
    auto cell_key = [&](int ix, int iy, int iz)->int64_t {
        // Large primes mixing
        return int64_t(ix) * 73856093LL ^ int64_t(iy) * 19349663LL ^ int64_t(iz) * 83492791LL;
    };

    std::vector<int> ix(N), iy(N), iz(N);
    const float inv = 1.0f / threshold;
    for (int i=0;i<(int)N;++i) {
        ix[i] = (int)std::floor(V_in[i].x() * inv);
        iy[i] = (int)std::floor(V_in[i].y() * inv);
        iz[i] = (int)std::floor(V_in[i].z() * inv);
    }

    for (int i=0;i<(int)N;++i) {
        // Check neighbor cells (27)
        for (int dx=-1; dx<=1; ++dx) for (int dy=-1; dy<=1; ++dy) for (int dz=-1; dz<=1; ++dz) {
            int64_t key = cell_key(ix[i]+dx, iy[i]+dy, iz[i]+dz);
            auto it = grid.find(key);
            if (it == grid.end()) continue;
            for (int j : it->second) {
                if (j >= i) continue; // only previous vertices
                float d2 = (V_in[i]-V_in[j]).squaredNorm();
                if (d2 <= thresh2) unite(i,j);
            }
        }
        // Insert this vertex into its own cell
        grid[cell_key(ix[i],iy[i],iz[i])].push_back(i);
    }

    // Aggregate clusters: root -> new index
    std::unordered_map<int,int> root2new; root2new.reserve(N);
    std::vector<Vec3f> accum; accum.reserve(N);
    std::vector<int> counts; counts.reserve(N);

    for (int i=0;i<(int)N;++i) {
        int r = find(i);
        auto it = root2new.find(r);
        if (it == root2new.end()) {
            int idx = (int)root2new.size();
            root2new[r] = idx;
            accum.push_back(V_in[i]);
            counts.push_back(1);
        } else {
            int idx = it->second;
            accum[idx] += V_in[i];
            counts[idx]++;
        }
    }

    // Compute centroids
    V_out.resize(accum.size());
    for (size_t i=0;i<accum.size();++i) V_out[i] = accum[i] / float(counts[i]);

    // Build mapping old -> new
    std::vector<int> map_old2new(N);
    for (int i=0;i<(int)N;++i) map_old2new[i] = root2new[find(i)];

    // Remap faces, remove degenerates, remove duplicates
    F_out.clear(); F_out.reserve(F_in.size());
    struct TriHash {
        size_t operator()(const std::array<int,3>& t) const noexcept {
            return (size_t)(t[0]*73856093) ^ (size_t)(t[1]*19349663) ^ (size_t)(t[2]*83492791);
        }
    };
    std::unordered_set<std::array<int,3>, TriHash> seen;
    seen.reserve(F_in.size()*2);

    for (const auto& f : F_in) {
        int a = map_old2new[f[0]];
        int b = map_old2new[f[1]];
        int c = map_old2new[f[2]];
        if (a==b || b==c || c==a) continue; // degenerate
        // Avoid duplicates disregarding orientation but keep original winding
        std::array<int,3> key = {a,b,c};
        std::array<int,3> key_sorted = key;
        std::sort(key_sorted.begin(), key_sorted.end());
        if (seen.insert(key_sorted).second) {
            F_out.emplace_back(a,b,c);
        }
    }
}

} // namespace cpu
} // namespace cubvh
