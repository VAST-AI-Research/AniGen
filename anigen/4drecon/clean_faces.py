"""Drop mesh triangles that bridge far-apart skeleton bones (e.g. a face webbing a robot's hand to its
hip) via AniGen's part-splitting logic. Default (surgical): remove a face only if its vertices' primary
bones are > --threshold apart, so limbs are never deleted. --aggressive also removes faces touching
spread-weight vertices (can delete a limb). Only faces are dropped; the pristine set is kept as
`faces_orig` (idempotent).
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # repo root -> anigen.*
import argparse
import numpy as np
import networkx as nx
from anigen.utils.mesh_part_splitting import (
    compute_geodesic_distances, get_primary_joint, identify_problematic_triangles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", required=True, help="rig npz (vertices/faces/skin_weights/joints/parents)")
    ap.add_argument("--threshold", type=int, default=3, help="drop a face if its vertices' primary bones "
                    "are > this many bones apart on the skeleton (<= is kept). 0 disables.")
    ap.add_argument("--aggressive", action="store_true", help="ALSO apply Method 1 (spread-weight) -- "
                    "removes the full web but DELETES limbs on smoothly-skinned meshes. See module docstring.")
    ap.add_argument("--out", default=None, help="output npz (default: overwrite --rig)")
    args = ap.parse_args()
    out = args.out or args.rig
    if args.threshold <= 0:
        print(f"clean_faces: threshold {args.threshold} <= 0 -> disabled, no-op"); return
    d = dict(np.load(args.rig))
    # source from the pristine faces if a previous run backed them up -> idempotent / re-tunable
    faces = (d["faces_orig"] if "faces_orig" in d else d["faces"]).astype(np.int32)
    skin = d["skin_weights"].astype(np.float32)          # (V, M) dense
    parents = d["parents"].astype(np.int64)
    M = skin.shape[1]

    # skeleton graph from parents (bone i <-> parent[i]); bone ids are 0..M-1 (== node ids here)
    G = nx.Graph()
    G.add_nodes_from(range(M))
    for i in range(M):
        p = int(parents[i])
        if 0 <= p < M and p != i:
            G.add_edge(i, p, weight=1)
    joint_indices = list(range(M))
    geo = compute_geodesic_distances(G, joint_indices)

    # dense skin -> top-4 (bone-index, weight) format expected by the splitter
    top4 = np.argsort(-skin, axis=1)[:, :4]
    joints4 = top4.astype(np.int32)
    weights4 = np.take_along_axis(skin, top4, axis=1).astype(np.float32)

    if args.aggressive:
        prob_tri, _ = identify_problematic_triangles(faces, joints4, weights4, joint_indices, geo, args.threshold)
        prob_tri = set(prob_tri)
        method = "Method1+2 (aggressive; may delete limbs)"
    else:
        # Method 2 only (surgical): primary-bone span > threshold -> a genuine cross-limb bridge face
        prim = np.array([get_primary_joint(joints4, weights4, v) for v in range(len(joints4))])
        fp = prim[faces]
        def span(a, b):
            return geo.get((int(a), int(b)), float("inf"))
        prob_tri = set(int(k) for k in range(len(faces))
                       if max(span(fp[k, 0], fp[k, 1]), span(fp[k, 0], fp[k, 2]), span(fp[k, 1], fp[k, 2])) > args.threshold)
        method = "Method2 (surgical; keeps limbs)"

    keep = np.array([i for i in range(len(faces)) if i not in prob_tri], dtype=np.int64)
    new_faces = faces[keep]
    print(f"{os.path.basename(args.rig)}: removed {len(prob_tri)}/{len(faces)} faces "
          f"({100*len(prob_tri)/max(1,len(faces)):.2f}%) spanning primary bones >{args.threshold} apart  [{method}]")
    if "faces_orig" not in d:
        d["faces_orig"] = faces
    d["faces"] = new_faces.astype(np.int32)
    np.savez(out, **d)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
