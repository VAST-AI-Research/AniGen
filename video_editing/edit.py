#!/usr/bin/env python3
"""Edit an AniGen asset into a NEW motion + camera and render it back into its own video with VACE.

    python edit.py --asset <name> --recon <recon_dir> --mask <mask> --name <run> \
        --motion_source <preset | "text prompt" | file.blend | pc_dir> \
        --camera_source <default | file.npz | "text prompt">

motion_source : a preset (camel_rear_up / robot_jumping_jacks / robot_combo), OR your own animation
                exported from Blender (file.blend, or a folder of per-frame xyz/rgb npz), OR a "text
                prompt" that our agent turns into the motion (needs ANTHROPIC_API_KEY).
camera_source : "default" = the original recon camera; OR a .npz trajectory (c2w + intr); OR a "text
                prompt" that our camera agent turns into a trajectory (needs ANTHROPIC_API_KEY).

The motion / camera / compositing steps run in the AniGen env; the final VACE render runs in the
FreeOrbit4D env. Override interpreters with the ANIGEN_PY / VACE_PY env vars.
"""
import os, sys, subprocess, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

HERE = os.path.dirname(os.path.abspath(__file__))
AG, FO = common.ANIGEN_ROOT, common.FREEORBIT4D_ROOT
ANIGEN_PY = os.environ.get("ANIGEN_PY", f"{AG}/.venv/bin/python")
VACE_PY = os.environ.get("VACE_PY", f"{FO}/.venv/bin/python")
PRESETS = {"camel_rear_up", "robot_jumping_jacks", "robot_combo"}
ENV = {**os.environ, "HF_HUB_DISABLE_XET": "1"}


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=ENV)


def ff(*args):
    run(["ffmpeg", "-y", "-loglevel", "error", *args])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asset", required=True, help="AniGen results/<asset> (rig_fit.npz + motion_fit.npz)")
    ap.add_argument("--recon", required=True, help="recon dir (depth/cameras + subject mask); see the README preprocess step")
    ap.add_argument("--mask", required=True, help="subject mask: a recon dynamic_mask dir OR a DAVIS annotation dir")
    ap.add_argument("--mask-kind", choices=["mask-dir", "gt-mask"], default="mask-dir")
    ap.add_argument("--motion_source", required=True)
    ap.add_argument("--camera_source", default="default")
    ap.add_argument("--name", required=True, help="run name for outputs")
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--gpu", type=int, default=0)
    # optional per-asset placement tweaks (rarely needed; e.g. the robot is shrunk so it stays in frame)
    ap.add_argument("--asset-scale", type=float, default=1.0)
    ap.add_argument("--asset-fwd", type=float, default=0.0)
    a = ap.parse_args()

    rig = f"{AG}/results/{a.asset}/rig_fit.npz"
    pc = f"/tmp/{a.name}_pc"
    motion_npz = f"{AG}/results/{a.asset}/reanim_{a.name}.npz"
    out = f"{HERE}/outputs/{a.name}"

    # ---- STEP 1: motion -> point-cloud sequence (pc) + motion npz ----
    ms = a.motion_source
    if ms in PRESETS:
        run([ANIGEN_PY, f"{HERE}/reanimate.py", a.asset, ms, "--frames", a.frames,
             "--full-texture", "--export", pc, "--save-motion", motion_npz])
    elif ms.endswith(".blend"):
        run([ANIGEN_PY, f"{HERE}/blender_export.py", "--asset", a.asset, "--blend", ms,
             "--frames", a.frames, "--export", pc, "--save-motion", motion_npz])
    elif os.path.isdir(ms):
        pc = ms                                            # a pre-exported per-frame xyz/rgb npz folder
        if not os.path.exists(motion_npz):
            motion_npz = f"{AG}/results/{a.asset}/motion_fit.npz"
    else:                                                  # free text -> motion agent
        run([ANIGEN_PY, f"{HERE}/agent_motion.py", a.asset, ms, "--frames", a.frames,
             "--full-texture", "--export", pc, "--save-motion", motion_npz])

    # ---- STEP 2: camera -> optional trajectory .npz (default = the recon camera) ----
    cam_npz = None
    cs = a.camera_source
    if cs and cs not in ("default", "recon"):
        if cs.endswith((".npy", ".npz")):
            cam_npz = cs
        else:                                              # free text -> camera agent
            cam_npz = f"/tmp/{a.name}_cam.npz"
            run([ANIGEN_PY, f"{HERE}/camera_agent.py", "--recon", a.recon, "--mask", a.mask,
                 "--frames", a.frames, "--instruction", cs, "--out", cam_npz])

    # ---- STEP 3: remove the subject + fill holes, then compose the asset into the scene ----
    hole = f"{out}/holefill"
    run([ANIGEN_PY, f"{HERE}/fill_bg_holes.py", "--recon", a.recon, f"--{a.mask_kind}", a.mask,
         "--out", hole, "--nf", a.frames, "--gpu", a.gpu])
    comp = [ANIGEN_PY, f"{HERE}/compose_and_render.py", "--recon", a.recon, f"--{a.mask_kind}", a.mask,
            "--bg-npz", f"{hole}/bg_augmented.npz", "--asset-pc", pc, "--motion", motion_npz, "--rig", rig,
            "--out", out, "--nf", a.frames, "--gpu", a.gpu,
            "--asset-scale", a.asset_scale, "--asset-fwd", a.asset_fwd]
    if cam_npz:
        comp += ["--camera-npy", cam_npz]
    run(comp)

    # ---- STEP 4: VACE render (Wan2.2-VACE-Fun-A14B, depth + reference conditioned) ----
    d = f"{FO}/outputs/rendering/{a.name}/inference"; os.makedirs(d, exist_ok=True)
    ff("-i", f"{out}/depth_video.mp4", "-vf", "scale=832:480:flags=area", "-fps_mode", "passthrough",
       "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", f"{d}/rendered_depths.mp4")
    ff("-i", f"{a.recon}/video.mp4", "-vf", "select=eq(n\\,0),scale=832:480", "-frames:v", "1", f"{d}/reference_image.png")
    ff("-i", f"{out}/video_src_original.mp4", "-vf", "scale=832:480", "-fps_mode", "passthrough",
       "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", f"{d}/original_images.mp4")
    run(["env", f"CUDA_VISIBLE_DEVICES={a.gpu}", "bash", "-c",
         f"cd {FO} && HF_HUB_DISABLE_XET=1 {VACE_PY} scripts/2_0_Wan2.2-VACE-Fun-A14B.py "
         f"--config configs/scenes/camel.yaml --data_dir outputs/rendering/{a.name}"])
    print(f"DONE -> {FO}/outputs/rendering/{a.name}/inference/output_video.mp4")


if __name__ == "__main__":
    main()
