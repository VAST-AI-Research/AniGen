# anigen/4drecon — Monocular 4D reconstruction

Fit an AniGen-generated asset to animate exactly the same as an input monocular video. The animatable mesh is produced from
the first frame; its camera and frame-0 pose are estimated with **VGGT-Omega**; per-frame skeleton
motion and 6DoF pose are then optimised by **differentiable rendering (nvdiffrast)** and matching supervisions from **LoMa**. Finally a joint
**pose + skin + texture + shape** refinement runs over the whole video.

## 🚀 One command — video (or frames) in, animated rig out

```bash
bash anigen/4drecon/run.sh --video clip.mp4                # or:  --frames frames_dir/
                           [--mask mask.mp4 | masks_dir/]  # optional; if omitted, SAM3 auto-masks
                           [--prompt "robot"] [--name myseq]
```

<table align="center" width="100%">
<tr>
<td width="75%" align="center"><img src="../../assets/4drecon/bear_fit.gif" width="100%"></td>
<td width="25%" align="center"><img src="../../assets/4drecon/bear_showcase.gif" width="100%"></td>
</tr>
<tr>
<td width="75%" align="center"><img src="../../assets/4drecon/camel_fit.gif" width="100%"></td>
<td width="25%" align="center"><img src="../../assets/4drecon/camel_showcase.gif" width="100%"></td>
</tr>
<tr>
<td width="75%" align="center"><img src="../../assets/4drecon/g1_1_fit.gif" width="100%"></td>
<td width="25%" align="center"><img src="../../assets/4drecon/g1_1_showcase.gif" width="100%"></td>
</tr>
<tr>
<td width="75%" align="center"><img src="../../assets/4drecon/g1_3_fit.gif" width="100%"></td>
<td width="25%" align="center"><img src="../../assets/4drecon/g1_3_showcase.gif" width="100%"></td>
</tr>
<tr>
<td width="75%" align="center"><img src="../../assets/4drecon/as2_1_fit.gif" width="100%"></td>
<td width="25%" align="center"><img src="../../assets/4drecon/as2_1_showcase.gif" width="100%"></td>
</tr>
</table>

## Install

```bash
pip install imageio imageio-ffmpeg matplotlib opencv-python rembg einops timm kornia transformers scipy
```

## Third-party checkouts (cloned separately, not committed)

```bash
# LoMa local feature matcher (appearance correspondences)
git clone https://github.com/davnords/LoMa.git extensions/LoMa       # LOMA_ROOT defaults to extensions/LoMa
#   matcher weights: extensions/LoMa/loma_G.pth
#   aux weights (dad / dedode_descriptor_G / dinov2_vitl14) auto-download once via torch.hub

# VGGT-Omega (camera + frame-0 pose)
git clone https://github.com/facebookresearch/vggt-omega extensions/vggt-omega   # set VGGT_OMEGA_CKPT
```

Overridable env vars (`paths.py`): `ANIGEN_DAVIS_ROOT`, `LOMA_ROOT`, `VGGT_OMEGA_REPO`, `VGGT_OMEGA_CKPT`.

## Data layout

`$ANIGEN_DAVIS_ROOT/JPEGImages/<seq>/*.jpg` (+ `Annotations/<seq>/*.png` masks); outputs go to
`results/<seq>/`. For a raw video: extract frames, then `prep_video.py --seq <seq>` makes masks
(SAM3 text-prompted VOS or BiRefNet, offline) + `assets/<seq>_rgba.png`.

## Pipeline

| # | Script | Output |
|---|---|---|
| prep | `prep_video.py --seq S` | masks + `assets/S_rgba.png` |
| 0 | `example.py` (repo root) | `results/S/mesh.glb`, `skeleton.glb` |
| 1 | `export_rig.py` + `symmetrize_colors.py` + `clean_faces.py` | `results/S/rig.npz` |
| 2 | `render_views.py` | `results/S/views/` |
| 3 | `pose_init.py` | `results/S/vggt_pose.npz` (VGGT-Omega camera + pose) |
| 4 | `refine_pose.py` | `results/S/pose0.npz` |
| 5 | `fit_video_loma.py` | `results/S/motion_loma.npz` (per-frame LoMa pose fit) |
| 6 | `refine_skin_texture_loma.py` | `results/S/rig_fit.npz` + `motion_fit.npz` |
| 7 | `render_results.py` / `render_skeleton.py` / `render_showcase.py` | `results/S/renders/*.mp4` |

Run everything: `CUDA_VISIBLE_DEVICES=0 bash anigen/4drecon/run_all.sh <seq>`

## Method

* Fixed full-frame camera + frame-0 pose from **VGGT-Omega**; apparent motion = per-frame root
  rigid transform + per-bone rotations via LBS/FK (Z-up canonical world).
* **LoMa per-frame fit** (`fit_video_loma.py`): match the rendered textured mesh against the masked
  frame by appearance; the correspondences (geodesic-FPS-covered so every limb owns one,
  error-weighted so mis-posed limbs pull hardest) drag limbs onto the real ones, with a multi-init
  pick/keep-best guard against drift. `--multiview 1` adds occlusion-robust multi-view matching.
* **Joint refinement** (`refine_skin_texture_loma.py`): jointly optimise pose + shared skin +
  per-vertex texture + coarse shape over all frames. Skin and shape are passed through Laplacian
  smoothing before driving LBS, so the smoothing itself is the regulariser.
* **Colour calibration** (coverage-gated): if ≥70% of the surface is observed (front-facing area),
  the full per-vertex texture is optimised; below 70% the original AniGen texture is kept and a small
  global colour **MLP** (original→video colour) corrects its tone/lighting. Being colour→colour with
  no spatial input, it extrapolates cleanly to the unseen surface (we compared affine / MLP /
  MLP+Fourier and use the plain MLP).

<p align="center"><img src="../../assets/4drecon/colormap_comparison.gif" width="90%"><br>
<sub><em>Colour-calibration methods (orbit) — GT · original · affine (underfitting) · MLP+Fourier (overfitting) · MLP (ours)</em></sub></p>
