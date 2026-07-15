#!/usr/bin/env python3
"""Bake a Blender-authored animation of an AniGen asset into the per-frame point-cloud sequence that
edit.py / compose_and_render.py consume. This is the "edit it yourself" path: import the asset's
`results/<asset>/mesh.glb` (rigged) into Blender, animate the armature, save a .blend, then:

    python edit.py ... --motion_source your_edit.blend

Contract: the mesh keeps AniGen's vertex order (don't add/remove verts), and the asset stays in its
canonical/rest world (edit.py's compositor handles scene placement). Per frame we evaluate the deformed
mesh vertices; colours come from rig_fit. Runs headless Blender (found on PATH or $BLENDER).

Robust fallback if your Blender export differs: just export a folder of per-frame `<t:05d>.npz`
(arrays xyz[N,3], rgb[N,3] uint8) yourself and pass that folder as --motion_source.
"""
import os, argparse, subprocess, tempfile

_BAKE = r'''
import bpy, sys, os, numpy as np
argv = sys.argv[sys.argv.index("--") + 1:]
out_dir, colors_npy, f0, f1 = argv[0], argv[1], int(argv[2]), int(argv[3])
os.makedirs(out_dir, exist_ok=True)
rgb = np.load(colors_npy)                                   # [N,3] uint8 (AniGen vertex_colors)
mesh_obj = next(o for o in bpy.data.objects if o.type == "MESH")
dg = bpy.context.evaluated_depsgraph_get()
for i, fr in enumerate(range(f0, f1)):
    bpy.context.scene.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = mesh_obj.evaluated_get(dg)
    m = ev.to_mesh()
    xyz = np.array([(mesh_obj.matrix_world @ v.co)[:] for v in m.vertices], dtype=np.float32)
    ev.to_mesh_clear()
    if xyz.shape[0] != rgb.shape[0]:
        raise SystemExit(f"vertex count {xyz.shape[0]} != AniGen {rgb.shape[0]} (keep the original mesh topology)")
    np.savez(os.path.join(out_dir, f"{i:05d}.npz"), xyz=xyz, rgb=rgb)
print(f"[blender] baked {f1 - f0} frames -> {out_dir}")
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True); ap.add_argument("--blend", required=True)
    ap.add_argument("--frames", type=int, default=49); ap.add_argument("--export", required=True)
    ap.add_argument("--save-motion", default=None)
    a = ap.parse_args()
    import numpy as np
    results = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", a.asset)
    d = np.load(os.path.join(results, "rig_fit.npz"))
    colors = (d["vertex_colors"] * 255).astype("uint8")
    cnpy = tempfile.mktemp(suffix=".npy"); np.save(cnpy, colors)
    script = tempfile.mktemp(suffix=".py"); open(script, "w").write(_BAKE)
    blender = os.environ.get("BLENDER", "blender")
    run = [blender, "--background", a.blend, "--python", script, "--", a.export, cnpy, "0", str(a.frames)]
    print("+ " + " ".join(run), flush=True)
    subprocess.run(run, check=True)
    if a.save_motion:                                      # reuse the fit's E_fit/scale for placement
        m = np.load(os.path.join(results, "motion_fit.npz"), allow_pickle=True)
        np.savez(a.save_motion, bone6=m["bone6"], r6=m["r6"], tg=m["tg"], scale=m["scale"],
                 E_fit=m["E_fit"], K_norm=m["K_norm"], W=m["W"], H=m["H"],
                 iou=np.zeros(len(m["bone6"]), "float32"),
                 names=np.array([f"frame_{i:03d}" for i in range(len(m["bone6"]))]))
        print(f"[blender] wrote placement motion -> {a.save_motion}")


if __name__ == "__main__":
    main()
