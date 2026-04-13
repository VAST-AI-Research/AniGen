#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <functional>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cassert>

#include <Eigen/Dense>

namespace cubvh {
namespace cpu {
namespace qd {

// Basic 3D vector (templated)
template <typename T>
struct Vec3T {
	T x, y, z;
	Vec3T() : x(T(0)), y(T(0)), z(T(0)) {}
	Vec3T(T X, T Y, T Z) : x(X), y(Y), z(Z) {}
	T& operator[](int i) { return i==0?x:(i==1?y:z); }
	T  operator[](int i) const { return i==0?x:(i==1?y:z); }
	Vec3T operator+(const Vec3T& o) const { return Vec3T(x+o.x,y+o.y,z+o.z); }
	Vec3T operator-(const Vec3T& o) const { return Vec3T(x-o.x,y-o.y,z-o.z); }
	Vec3T operator*(T s) const { return Vec3T(x*s,y*s,z*s); }
	Vec3T operator/(T s) const { return Vec3T(x/s,y/s,z/s); }
	Vec3T& operator+=(const Vec3T& o){ x+=o.x;y+=o.y;z+=o.z; return *this; }
	Vec3T& operator-=(const Vec3T& o){ x-=o.x;y-=o.y;z-=o.z; return *this; }
	bool operator==(const Vec3T& o) const { return x==o.x && y==o.y && z==o.z; }
};

template <typename T>
inline T dot(const Vec3T<T>& a, const Vec3T<T>& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
template <typename T>
inline Vec3T<T> cross(const Vec3T<T>& a, const Vec3T<T>& b){
	return Vec3T<T>(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
template <typename T>
inline T norm(const Vec3T<T>& v){ return std::sqrt(dot(v,v)); }
template <typename T>
inline T squared_norm(const Vec3T<T>& v){ return dot(v,v); }
template <typename T>
inline Vec3T<T> normalize(const Vec3T<T>& v){ T n=norm(v); return n>T(0)? v/n : Vec3T<T>(); }

// Basic 3x3 matrix (full, templated)
template <typename T>
struct Mat3T {
	// Row-major storage
	T m[9];
	Mat3T(){ for(int i=0;i<9;++i) m[i]=T(0); }
	static Mat3T identity(){ Mat3T A; A(0,0)=A(1,1)=A(2,2)=T(1); return A; }
	T& operator()(int r,int c){ return m[r*3+c]; }
	T  operator()(int r,int c) const{ return m[r*3+c]; }
	Mat3T& operator+=(const Mat3T& B){ for(int i=0;i<9;++i) m[i]+=B.m[i]; return *this; }
	Mat3T operator+(const Mat3T& B) const{ Mat3T C=*this; C+=B; return C; }
	Mat3T operator*(T s) const{ Mat3T C; for(int i=0;i<9;++i) C.m[i]=m[i]*s; return C; }
	Vec3T<T> operator*(const Vec3T<T>& v) const{
		return Vec3T<T>(
			m[0]*v.x + m[1]*v.y + m[2]*v.z,
			m[3]*v.x + m[4]*v.y + m[5]*v.z,
			m[6]*v.x + m[7]*v.y + m[8]*v.z
		);
	}
	Mat3T transpose() const{
		Mat3T Tt; for(int r=0;r<3;++r) for(int c=0;c<3;++c) Tt(r,c)=(*this)(c,r); return Tt;
	}
};

template <typename T>
inline Mat3T<T> outer(const Vec3T<T>& a, const Vec3T<T>& b){
	Mat3T<T> A; A(0,0)=a.x*b.x; A(0,1)=a.x*b.y; A(0,2)=a.x*b.z;
			  A(1,0)=a.y*b.x; A(1,1)=a.y*b.y; A(1,2)=a.y*b.z;
			  A(2,0)=a.z*b.x; A(2,1)=a.z*b.y; A(2,2)=a.z*b.z; return A;
}

template <typename T>
inline T det(const Mat3T<T>& A){
	return A(0,0)*(A(1,1)*A(2,2)-A(1,2)*A(2,1))
		 - A(0,1)*(A(1,0)*A(2,2)-A(1,2)*A(2,0))
		 + A(0,2)*(A(1,0)*A(2,1)-A(1,1)*A(2,0));
}

template <typename T>
inline Mat3T<T> adjugate(const Mat3T<T>& A){
	Mat3T<T> C;
	C(0,0) =  (A(1,1)*A(2,2) - A(1,2)*A(2,1));
	C(0,1) = -(A(1,0)*A(2,2) - A(1,2)*A(2,0));
	C(0,2) =  (A(1,0)*A(2,1) - A(1,1)*A(2,0));
	C(1,0) = -(A(0,1)*A(2,2) - A(0,2)*A(2,1));
	C(1,1) =  (A(0,0)*A(2,2) - A(0,2)*A(2,0));
	C(1,2) = -(A(0,0)*A(2,1) - A(0,1)*A(2,0));
	C(2,0) =  (A(0,1)*A(1,2) - A(0,2)*A(1,1));
	C(2,1) = -(A(0,0)*A(1,2) - A(0,2)*A(1,0));
	C(2,2) =  (A(0,0)*A(1,1) - A(0,1)*A(1,0));
	return C.transpose();
}

template <typename T>
inline bool invert(const Mat3T<T>& A, Mat3T<T>& inv){
	T d = det(A);
	if (std::fabs(d) < T(1e-12)) return false;
	Mat3T<T> adj = adjugate(A);
	inv = adj * (T(1)/d);
	return true;
}

// Symmetric quadric: Q(x) = x^T A x + 2 b^T x + c, with A symmetric (templated)
template <typename T>
struct QuadricT {
	// Store upper triangular of A (a00, a01, a02, a11, a12, a22), b(3), c
	T a00=T(0),a01=T(0),a02=T(0),a11=T(0),a12=T(0),a22=T(0);
	T b0=T(0),b1=T(0),b2=T(0);
	T c=T(0);

	QuadricT() = default;
	explicit QuadricT(T C) : c(C) {}
	QuadricT(const Mat3T<T>& A, const Vec3T<T>& b, T C){
		a00=A(0,0); a01=A(0,1); a02=A(0,2);
		a11=A(1,1); a12=A(1,2); a22=A(2,2);
		b0=b.x; b1=b.y; b2=b.z; c=C;
	}

	Mat3T<T> A() const{
		Mat3T<T> M; M(0,0)=a00; M(0,1)=a01; M(0,2)=a02;
				   M(1,0)=a01; M(1,1)=a11; M(1,2)=a12;
				   M(2,0)=a02; M(2,1)=a12; M(2,2)=a22; return M;
	}
	Vec3T<T> b() const{ return Vec3T<T>(b0,b1,b2); }
	T operator()(const Vec3T<T>& x) const{
		Vec3T<T> Ax = A()*x;
		return dot(x,Ax) + T(2)*dot(b(),x) + c;
	}
	bool isZero() const{
		return std::fabs(a00)+std::fabs(a01)+std::fabs(a02)+std::fabs(a11)+std::fabs(a12)+std::fabs(a22)+
		       std::fabs(b0)+std::fabs(b1)+std::fabs(b2)+std::fabs(c) < T(1e-15);
	}
	QuadricT& operator+=(const QuadricT& q){
		a00+=q.a00; a01+=q.a01; a02+=q.a02; a11+=q.a11; a12+=q.a12; a22+=q.a22;
		b0+=q.b0; b1+=q.b1; b2+=q.b2; c+=q.c; return *this;
	}
	QuadricT operator+(const QuadricT& q) const{ QuadricT r=*this; r+=q; return r; }
	QuadricT operator*(T s) const{
		QuadricT r; r.a00=a00*s; r.a01=a01*s; r.a02=a02*s; r.a11=a11*s; r.a12=a12*s; r.a22=a22*s;
		r.b0=b0*s; r.b1=b1*s; r.b2=b2*s; r.c=c*s; return r;
	}
	T trace() const{ return a00 + a11 + a22; }
};

template <typename T>
inline QuadricT<T> planeQuadric(const Vec3T<T>& n_unit, const Vec3T<T>& p){
	// distance to plane: n·x - d, with d = n·p
	T d = dot(n_unit, p);
	Mat3T<T> A = outer(n_unit, n_unit);
	Vec3T<T> b = n_unit * (-d);
	T c = d*d;
	return QuadricT<T>(A,b,c);
}

template <typename T>
inline QuadricT<T> pointQuadric(const Vec3T<T>& p){
	Mat3T<T> I = Mat3T<T>::identity();
	Vec3T<T> b = Vec3T<T>(-p.x, -p.y, -p.z);
	T c = dot(p,p);
	return QuadricT<T>(I,b,c);
}

template <typename T>
inline bool minimizer(const QuadricT<T>& q, const std::vector<Vec3T<T>>& choices, Vec3T<T>& x_out, T& cost_out){
	Mat3T<T> A = q.A();
	Mat3T<T> Ai;
	Vec3T<T> rhs = Vec3T<T>(-q.b0, -q.b1, -q.b2);
	bool ok = invert(A, Ai);
	if (ok){
		// x = A^{-1} * (-b)
		x_out = Ai * rhs;
		cost_out = q(x_out);
		return true;
	}
	// fallback: choose best among provided points
	x_out = choices.empty() ? Vec3T<T>() : choices[0];
	cost_out = q(x_out);
	for(size_t i=1;i<choices.size();++i){ T c = q(choices[i]); if (c < cost_out){ cost_out=c; x_out=choices[i]; } }
	return false;
}

// Triangle mesh (indexed, templated)
template <typename T>
struct MeshT {
	std::vector<Vec3T<T>> vertices;
	std::vector<std::array<int,3>> faces; // CCW
};

// Internal helpers (templated)
template <typename T>
inline Vec3T<T> faceNormal(const MeshT<T>& m, const std::array<int,3>& f){
	const Vec3T<T>& a = m.vertices[f[0]];
	const Vec3T<T>& b = m.vertices[f[1]];
	const Vec3T<T>& c = m.vertices[f[2]];
	return cross(b-a, c-a);
}

template <typename T>
inline T triangleArea(const MeshT<T>& m, const std::array<int,3>& f){
	return T(0.5) * norm(faceNormal<T>(m,f));
}

inline bool validFace(const std::array<int,3>& f){
	return f[0]!=f[1] && f[1]!=f[2] && f[2]!=f[0];
}

// Edge key utility
struct EdgeKey {
	int a,b;
	EdgeKey() : a(-1), b(-1) {}
	EdgeKey(int i, int j){ if (i<j){ a=i; b=j; } else { a=j; b=i; } }
	bool operator==(const EdgeKey& o) const { return a==o.a && b==o.b; }
};

struct EdgeKeyHash { size_t operator()(const EdgeKey& k) const { return (size_t)k.a*73856093u ^ (size_t)k.b*19349663u; } };

template <typename T>
struct EdgeInfo {
	int v0, v1; // endpoints (unordered stored with v0<v1)
	std::vector<int> adjacent_faces; // indices into mesh.faces
	Vec3T<T> collapse_pos;
	T cost;
	bool valid = true;
};

// Decimator: simple greedy QEM-based
template <typename T>
class DecimatorT {
public:
	explicit DecimatorT(MeshT<T> mesh) : m(mesh) {}

	// Reduce mesh to target vertex count (or until no improvement).
	void decimate(int target_vertices){
		target_vertices = std::max(0, target_vertices);
		// Build adjacency and initial structures
		buildAdjacency();
		buildVertexQuadrics();
		buildEdgeMap();
		// Priority queue with lazy updates
		using QEntry = std::pair<T, EdgeKey>;
		auto cmp = [](const QEntry& a, const QEntry& b){ return a.first > b.first; };
		std::priority_queue<QEntry, std::vector<QEntry>, decltype(cmp)> pq(cmp);
		for (const auto& kv : edges) {
			const EdgeInfo<T>& e = kv.second;
			if (e.valid)
				pq.emplace(e.cost, kv.first);
		}
		// Greedy collapses until target or no candidates
		while ((int)activeVertexCount() > target_vertices && !pq.empty()) {
			auto [cost, key] = pq.top(); pq.pop();
			auto it = edges.find(key);
			if (it == edges.end()) continue;
			EdgeInfo<T>& e = it->second;
			if (!e.valid) continue;
			// Recheck stale cost
			if (std::fabs(e.cost - cost) > T(1e-9)) {
				pq.emplace(e.cost, key);
				continue;
			}
			// Collapse feasibility
			if (!canCollapse(e)) { e.valid=false; continue; }
			// Apply collapse
			if (!applyCollapse(e)) { e.valid=false; continue; }
			// Mark this edge invalid
			e.valid = false;
			// Recompute QEM for affected vertices and update incident edges
			updateAroundVertex(e.v0, pq);
		}
		compact();
	}

	// Parallel-style decimator: selects a vertex-disjoint independent set per
	// iteration, collapses, then rebuilds adjacency and costs.
	void parallelDecimate(int target_vertices){
		// Initialize
		target_vertices = std::max(0, target_vertices);
		buildAdjacency();
		buildVertexQuadrics();
		buildEdgeMap();

		for (int iter = 0; iter < 256; ++iter){
			if ((int)activeVertexCount() <= target_vertices) break;
			// Build candidate list (valid edges) and sort by cost
			std::vector<EdgeKey> candidates; candidates.reserve(edges.size());
			for (const auto& kv : edges){ if (kv.second.valid) candidates.push_back(kv.first); }
			if (candidates.empty()) break;
			std::sort(candidates.begin(), candidates.end(), [&](const EdgeKey& a, const EdgeKey& b){ return edges[a].cost < edges[b].cost; });

			// Select independent set (vertex-disjoint) with simple topology filter
			std::vector<char> used(m.vertices.size(), 0);
			std::vector<EdgeKey> batch; batch.reserve(candidates.size()/4 + 1);
			for (const EdgeKey& key : candidates){
				auto it = edges.find(key);
				EdgeInfo<T>& e = it->second;
				if (!e.valid) continue;
				if (used[e.v0] || used[e.v1]) continue;
				if (tooManyCommonNeighbors(e)) continue;
				if (!canCollapse(e)) continue;
				batch.push_back(key);
				used[e.v0] = used[e.v1] = 1;
			}

			if (batch.empty()) break;

			// Collapse batch sequentially (they are vertex-disjoint)
			std::unordered_set<int> deleted_vertices; deleted_vertices.reserve(batch.size());
			for (const EdgeKey& key : batch){
				auto it = edges.find(key);
				if (it == edges.end()) continue;
				EdgeInfo<T>& e = it->second;
				if (!e.valid) continue;
				int v1 = e.v1;
				if (!applyCollapse(e)) { e.valid=false; continue; }
				e.valid = false;
				if (v1>=0) deleted_vertices.insert(v1);
			}
			// Rebuild adjacency and costs for correctness and speed
			buildAdjacency();
			buildVertexQuadrics();
			buildEdgeMap();
		}

		compact();
	}
	// Count common neighbors of an edge’s endpoints; reject if too many.
	bool tooManyCommonNeighbors(const EdgeInfo<T>& e) const {
		// Collect neighbor sets for v0 and v1, excluding the opposite endpoint
		// and excluding the third vertices of faces incident to this edge.
		std::unordered_set<int> n0; n0.reserve(16);
		std::unordered_set<int> n1; n1.reserve(16);

		// Determine third vertices of faces adjacent to this edge
		std::unordered_set<int> third; third.reserve(4);
		for (int fi : e.adjacent_faces){
			if (fi < 0 || fi >= (int)m.faces.size()) continue;
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			for (int k=0;k<3;++k){
				int vk = f[k];
				if (vk!=e.v0 && vk!=e.v1) third.insert(vk);
			}
		}

		auto collect = [&](int v, std::unordered_set<int>& out){
			for (int fi : v_faces[v]){
				if (fi < 0 || fi >= (int)m.faces.size()) continue;
				const auto& f = m.faces[fi];
				if (!validFace(f)) continue;
				for (int k=0;k<3;++k){
					int u = f[k];
					if (u==v) continue;
					if (u==e.v0 || u==e.v1) continue; // exclude edge endpoints
					if (third.find(u) != third.end()) continue; // exclude third vertices
					out.insert(u);
				}
			}
		};
		collect(e.v0, n0);
		collect(e.v1, n1);

		// Count intersection
		int common = 0;
		if (n0.size() < n1.size()){
			for (int u : n0) common += (n1.find(u) != n1.end());
		} else {
			for (int u : n1) common += (n0.find(u) != n0.end());
		}
		// Boundary edges should have <=1 common neighbor; interior <=2
		int max_common = (int)e.adjacent_faces.size() == 1 ? 1 : 2;
		return common > max_common;
	}

	const MeshT<T>& mesh() const { return m; }

private:
	MeshT<T> m;
	std::vector<QuadricT<T>> vq; // per-vertex quadrics
	std::vector<std::vector<int>> v_faces; // incident face indices
	std::unordered_map<EdgeKey, EdgeInfo<T>, EdgeKeyHash> edges;
	std::vector<char> v_deleted; // 1 if removed
	std::vector<char> v_boundary; // 1 if boundary vertex

	size_t currentVertexCount() const { return m.vertices.size(); }
	size_t activeVertexCount() const {
		std::vector<char> used(m.vertices.size(), 0);
		for (const auto& f : m.faces){
			if (!validFace(f)) continue;
			if (f[0]>=0 && f[0]<(int)used.size()) used[f[0]] = 1;
			if (f[1]>=0 && f[1]<(int)used.size()) used[f[1]] = 1;
			if (f[2]>=0 && f[2]<(int)used.size()) used[f[2]] = 1;
		}
		size_t cnt=0; for(char u: used) if(u) ++cnt; return cnt;
	}

	void buildAdjacency(){
		v_faces.assign(m.vertices.size(), {});
		v_deleted.assign(m.vertices.size(), 0);
		for (int fi=0; fi<(int)m.faces.size(); ++fi){
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			v_faces[f[0]].push_back(fi);
			v_faces[f[1]].push_back(fi);
			v_faces[f[2]].push_back(fi);
		}
	}

	void buildVertexQuadrics(){
		vq.assign(m.vertices.size(), QuadricT<T>());
		for (size_t v=0; v<m.vertices.size(); ++v){
			QuadricT<T> qsum;
			for (int fi : v_faces[v]){
				const auto& f = m.faces[fi];
				if (!validFace(f)) continue;
				Vec3T<T> a = m.vertices[f[0]];
				Vec3T<T> b = m.vertices[f[1]];
				Vec3T<T> c = m.vertices[f[2]];
				Vec3T<T> n = faceNormal<T>(m, f);
				T area = T(0.5)*norm(n);
				if (area <= T(0)) continue;
				Vec3T<T> nu = normalize(n);
				// quadric through the specific vertex position
				Vec3T<T> xi = m.vertices[v];
				qsum += planeQuadric(nu, xi) * area;
			}
			vq[v] = qsum;
		}
	}

	void buildEdgeMap(){
		edges.clear();
		edges.reserve(m.faces.size()*2);
		auto add_edge = [&](int i, int j, int fi){
			if (i==j) return;
			EdgeKey key(i,j);
			auto it = edges.find(key);
			if (it==edges.end()){
				EdgeInfo<T> info; info.v0 = std::min(i,j); info.v1 = std::max(i,j);
				info.cost = std::numeric_limits<T>::infinity();
				info.adjacent_faces.clear(); info.collapse_pos = Vec3T<T>(); info.valid=true;
				it = edges.emplace(key, info).first;
			}
			it->second.adjacent_faces.push_back(fi);
		};
		for (int fi=0; fi<(int)m.faces.size(); ++fi){
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			add_edge(f[0], f[1], fi);
			add_edge(f[1], f[2], fi);
			add_edge(f[2], f[0], fi);
		}
		// mark boundary vertices
		v_boundary.assign(m.vertices.size(), 0);
		for (const auto& kv : edges){
			const EdgeInfo<T>& e = kv.second;
			if ((int)e.adjacent_faces.size() == 1){
				if (e.v0 >=0 && e.v0 < (int)v_boundary.size()) v_boundary[e.v0] = 1;
				if (e.v1 >=0 && e.v1 < (int)v_boundary.size()) v_boundary[e.v1] = 1;
			}
		}
		// compute initial costs
		for (auto& kv : edges){ updateEdgeCost(kv.second); }
	}

	void updateEdgeCost(EdgeInfo<T>& e){
		if (v_deleted[e.v0] || v_deleted[e.v1]) { e.valid=false; return; }
		QuadricT<T> q = vq[e.v0] + vq[e.v1];
		Vec3T<T> x0 = m.vertices[e.v0];
		Vec3T<T> x1 = m.vertices[e.v1];
		Vec3T<T> xm = (x0 + x1) * T(0.5);
		if (q.trace() <= T(1e-12)) {
			// Stabilize with area-scaled point quadric at midpoint
			T area_sum = T(0);
			for (int fi : e.adjacent_faces){
				if (fi < 0 || fi >= (int)m.faces.size()) continue;
				const auto& f = m.faces[fi];
				if (!validFace(f)) continue;
				area_sum += triangleArea<T>(m, f);
			}
			T scale = std::max(T(1e-12), area_sum) * T(1e-9);
			q = q + pointQuadric<T>(xm) * scale;
		}
		Vec3T<T> x; T cost;
		minimizer<T>(q, {x0,x1,xm}, x, cost);
		e.collapse_pos = x; e.cost = cost;
		e.valid = true;
	}

	bool canCollapse(const EdgeInfo<T>& e){
		// Keep openings intact: forbid collapsing any edge incident to a boundary vertex
		if (isBoundaryVertex(e.v0) || isBoundaryVertex(e.v1)) return false;
		// Prevent drastic normal flips: for each adjacent face, simulate collapse
		Vec3T<T> x = e.collapse_pos;
		for (int fi : e.adjacent_faces){
			if (fi < 0 || fi >= (int)m.faces.size()) continue;
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			std::array<Vec3T<T>,3> P = { m.vertices[f[0]], m.vertices[f[1]], m.vertices[f[2]] };
			if (f[0]==e.v0 || f[0]==e.v1) P[0] = x;
			if (f[1]==e.v0 || f[1]==e.v1) P[1] = x;
			if (f[2]==e.v0 || f[2]==e.v1) P[2] = x;
			Vec3T<T> e0 = P[1]-P[0];
			Vec3T<T> e1 = P[2]-P[0];
			if (squared_norm(e0) < T(1e-24) || squared_norm(e1) < T(1e-24)) continue;
			Vec3T<T> n_before = faceNormal<T>(m, f);
			Vec3T<T> n_after = cross(e0, e1);
			T nb2 = squared_norm(n_before);
			T na2 = squared_norm(n_after);
			if (nb2>T(0) && na2>T(0)){
				T ndot = dot(n_before, n_after);
				// Reject flips
				if (ndot < T(0)) return false;
				// Reject large rotations: cos(theta) < 0.5
				if (ndot*ndot < T(0.25) * nb2 * na2) return false;
			}
		}
		return true;
	}

	bool applyCollapse(const EdgeInfo<T>& e){
		// Merge v1 into v0, set position to collapse_pos
		int v0 = e.v0, v1 = e.v1;
		if (v_deleted[v0] || v_deleted[v1]) return false;
		m.vertices[v0] = e.collapse_pos;
		// Gather affected faces: union of v_faces[v0] and v_faces[v1]
		std::vector<int> affected = v_faces[v0];
		affected.insert(affected.end(), v_faces[v1].begin(), v_faces[v1].end());
		// Update faces: replace v1 with v0; mark degenerate faces invalid as {-1,-1,-1}
		for (int fi : affected){
			if (fi < 0 || fi >= (int)m.faces.size()) continue;
			auto& f = m.faces[fi];
			for (int k=0;k<3;++k){ if (f[k]==v1) f[k]=v0; }
			if (!validFace(f)) { f = { -1,-1,-1 }; }
		}
		// Update adjacency: clear v_faces[v0], rebuild by scanning affected faces
		v_faces[v0].clear();
		for (int fi : affected){
			if (fi < 0 || fi >= (int)m.faces.size()) continue;
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			if (f[0]==v0 || f[1]==v0 || f[2]==v0) v_faces[v0].push_back(fi);
		}
		// Mark v1 deleted
		v_deleted[v1] = 1;
		v_faces[v1].clear();
		// Remove edges incident to v1
		// We leave them invalid; they’ll be skipped lazily
		return true;
	}

	void compact(){
		// Remove isolated vertices and reindex
		std::vector<char> used(m.vertices.size(), 0);
		for (auto& f : m.faces){
			if (!validFace(f)) continue;
			for(int k=0;k<3;++k) if (f[k]>=0 && f[k]<(int)used.size()) used[f[k]] = 1;
		}

		std::vector<int> newIndex(m.vertices.size(), -1);
		std::vector<Vec3T<T>> nv; nv.reserve(m.vertices.size());
		for (size_t i=0;i<m.vertices.size(); ++i){ if (used[i]){ newIndex[i] = (int)nv.size(); nv.push_back(m.vertices[i]); } }
		// Compact faces: drop invalid ones and remap indices
		std::vector<std::array<int,3>> nf; nf.reserve(m.faces.size());
		for (auto& f : m.faces){
			if (!validFace(f)) continue;
			std::array<int,3> g = { newIndex[f[0]], newIndex[f[1]], newIndex[f[2]] };
			if (validFace(g)) nf.push_back(g);
		}
		m.vertices.swap(nv);
		m.faces.swap(nf);
	}

	bool isBoundaryVertex(int v) const {
		if (v < 0 || v >= (int)v_boundary.size()) return false;
		return v_boundary[v] != 0;
	}

	void updateVertexQEM(int v){
		QuadricT<T> qsum;
		for (int fi : v_faces[v]){
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			Vec3T<T> n = faceNormal<T>(m,f);
			T area = T(0.5)*norm(n);
			if (area <= T(0)) continue;
			Vec3T<T> nu = normalize(n);
			qsum += planeQuadric<T>(nu, m.vertices[v]) * area;
		}
		vq[v] = qsum;
	}

	template <typename PQ>
	void updateAroundVertex(int v, PQ &pq){
		// Recompute QEM for v and neighbors, update incident edges in queue
		updateVertexQEM(v);
		// Collect neighbor vertices from incident faces
		std::unordered_set<int> nbr;
		for (int fi : v_faces[v]){
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			for (int k=0;k<3;++k){ if (f[k]!=v) nbr.insert(f[k]); }
		}
		for (int u : nbr){ updateVertexQEM(u); }
		// Update edges (v,u)
		for (int u : nbr){
			if (v_deleted[u]) continue;
			EdgeKey key(v,u);
			auto it = edges.find(key);
			if (it == edges.end()){
				// create edge if faces define it
				EdgeInfo<T> info; info.v0 = std::min(v,u); info.v1 = std::max(v,u); info.valid=true;
				// find adjacency faces with both v and u
				info.adjacent_faces.clear();
				// scan v_faces[v]
				for (int fi : v_faces[v]){
					const auto& f = m.faces[fi];
					if (!validFace(f)) continue;
					bool hasu = (f[0]==u || f[1]==u || f[2]==u);
					bool hasv = (f[0]==v || f[1]==v || f[2]==v);
					if (hasu && hasv) info.adjacent_faces.push_back(fi);
				}
				edges.emplace(key, info);
				updateEdgeCost(edges.find(key)->second);
				pq.emplace(edges.find(key)->second.cost, key);
			} else {
				updateEdgeCost(it->second);
				if (it->second.valid) pq.emplace(it->second.cost, key);
			}
		}
	}

	void updateAroundVertexLocal(int v){
		// Recompute QEM for v and neighbors, update incident edges without queue
		updateVertexQEM(v);
		std::unordered_set<int> nbr;
		for (int fi : v_faces[v]){
			const auto& f = m.faces[fi];
			if (!validFace(f)) continue;
			for (int k=0;k<3;++k){ if (f[k]!=v) nbr.insert(f[k]); }
		}
		for (int u : nbr){ updateVertexQEM(u); }
		// Update or create edges (v,u)
		for (int u : nbr){
			if (v_deleted[u]) continue;
			EdgeKey key(v,u);
			auto it = edges.find(key);
			if (it == edges.end()){
				EdgeInfo<T> info; info.v0 = std::min(v,u); info.v1 = std::max(v,u); info.valid=true;
				info.adjacent_faces.clear();
				// scan v_faces[v]
				for (int fi : v_faces[v]){
					const auto& f = m.faces[fi];
					if (!validFace(f)) continue;
					bool hasu = (f[0]==u || f[1]==u || f[2]==u);
					bool hasv = (f[0]==v || f[1]==v || f[2]==v);
					if (hasu && hasv) info.adjacent_faces.push_back(fi);
				}
				edges.emplace(key, info);
				updateEdgeCost(edges.find(key)->second);
			} else {
				updateEdgeCost(it->second);
			}
		}
	}
};

// Backward-compatible double-precision aliases and float alternatives
using Vec3d = Vec3T<double>;
using Vec3f = Vec3T<float>;
using Mat3d = Mat3T<double>;
using Mat3f = Mat3T<float>;
using Quadricd = QuadricT<double>;
using Quadricf = QuadricT<float>;
using Meshd = MeshT<double>;
using Meshf = MeshT<float>;
using Decimatord = DecimatorT<double>;
using Decimatorf = DecimatorT<float>;

// Preserve previous non-templated type names as double-precision defaults
using Vec3 = Vec3d;
using Mat3 = Mat3d;
using Quadric = Quadricd;
using Mesh = Meshd;
using Decimator = Decimatord;

} // namespace qd
} // namespace cpu
} // namespace cubvh
