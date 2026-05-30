"""Benchmark: mtlmesh bounds-check overhead after 2A fix.

The 2A fix added CPU-side bounds checks to remove_faces /
remove_unreferenced_vertices / fill_holes / simplify_step to prevent the
segfaults that shivampkumar reported. Checks are O(F) on face counts;
this bench measures that the overhead is below noise for typical meshes.
"""
import time
import torch
import cumesh


def icosphere_faces(subdiv=3):
    """Build an icosphere-like triangulated mesh of roughly the size we
    care about for the bench. Just enough to exercise fill_holes + simplify
    with realistic data shapes."""
    # Icosahedron base
    t = (1 + 5 ** 0.5) / 2
    verts = torch.tensor([
        [-1,  t, 0], [1,  t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1,  t], [0, 1,  t], [0, -1, -t], [0, 1, -t],
        [ t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=torch.float32)
    tris = torch.tensor([
        [0, 11, 5],  [0, 5, 1],   [0, 1, 7],   [0, 7, 10],  [0, 10, 11],
        [1, 5, 9],   [5, 11, 4],  [11, 10, 2], [10, 7, 6],  [7, 1, 8],
        [3, 9, 4],   [3, 4, 2],   [3, 2, 6],   [3, 6, 8],   [3, 8, 9],
        [4, 9, 5],   [2, 4, 11],  [6, 2, 10],  [8, 6, 7],   [9, 8, 1],
    ], dtype=torch.int32)

    # Subdivide
    for _ in range(subdiv):
        new_verts = [verts]
        new_tris = []
        midpoints = {}
        vcount = [verts.shape[0]]
        for f in tris:
            a, b, c = f.tolist()
            def midpt(u, v):
                key = (min(u, v), max(u, v))
                if key not in midpoints:
                    mid = (verts[u] + verts[v]) / 2
                    mid = mid / mid.norm()
                    new_verts.append(mid.unsqueeze(0))
                    midpoints[key] = vcount[0]
                    vcount[0] += 1
                return midpoints[key]
            ab, bc, ca = midpt(a, b), midpt(b, c), midpt(c, a)
            new_tris += [
                [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca],
            ]
        verts = torch.cat(new_verts, 0)
        tris = torch.tensor(new_tris, dtype=torch.int32)

    return verts, tris


def bench(label, fn, warmup=2, iters=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ms = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:48s} {ms:8.3f} ms/call")
    return ms


print("=" * 72)
print("mtlmesh bench — bounds-check overhead (M3 Max)")
print("=" * 72)

for subdiv in [2, 3, 4]:
    verts, tris = icosphere_faces(subdiv)
    print(f"\nicosphere subdiv={subdiv}: V={verts.shape[0]} F={tris.shape[0]}")

    def fh():
        m = cumesh.CuMesh()
        m.init(verts, tris)
        m.fill_holes(0.01)  # sphere has no holes; triggers the cold path
    def rf():
        m = cumesh.CuMesh()
        m.init(verts, tris)
        mask = torch.ones(tris.shape[0], dtype=torch.bool)
        mask[::10] = False  # remove every 10th face
        m.remove_faces(mask)
    def uf():
        m = cumesh.CuMesh()
        m.init(verts, tris)
        m.unify_face_orientations()

    bench("init", lambda: cumesh.CuMesh().init(verts, tris))
    bench("fill_holes (sphere, no-op)", fh)
    bench("remove_faces (10% removed)", rf)
    bench("unify_face_orientations", uf)
