#!/bin/bash
# End-to-end 4D reconstruction for one DAVIS-style sequence:
#   AniGen rig -> VGGT-Omega pose init -> nvdiffrast pose refine -> CoTracker3 tracks
#   -> SO(3) parent-relative skeleton motion fit -> original / cyclic / skeleton renders.
#
# Usage:   CUDA_VISIBLE_DEVICES=0 bash anigen/4drecon/run_all.sh [SEQ]      (default SEQ=bear)
# Prereqs: run from the repo root; frames at $ANIGEN_DAVIS_ROOT/JPEGImages/<SEQ>/*.jpg
#          then:  python anigen/4drecon/prep_video.py --seq <SEQ>   (masks + assets/<SEQ>_rgba.png)
#          third-party trackers cloned per anigen/4drecon/README.md.
set -e
cd "$(dirname "$0")/../.."          # repo root
SEQ=${1:-bear}
PY=${PYTHON:-python}
D=anigen/4drecon
OUT=results/$SEQ

echo "===== [0/8] generate rigged mesh ($SEQ) ====="
$PY example.py --image_path assets/${SEQ}_rgba.png --output_name $SEQ --output_dir results/

echo "===== [1/8] export rig npz + symmetrize colors ====="
$PY $D/export_rig.py --image_path assets/${SEQ}_rgba.png --out $OUT/rig.npz
$PY $D/symmetrize_colors.py --rig $OUT/rig.npz

echo "===== [2/8] render static views for pose init ====="
$PY $D/render_views.py --rig $OUT/rig.npz --out $OUT/views

echo "===== [3/8] VGGT-Omega pose init ====="
$PY $D/pose_init.py --rig $OUT/rig.npz --views $OUT/views --real_rgba assets/${SEQ}_rgba.png --out $OUT/vggt_pose.npz

echo "===== [4/8] nvdiffrast pose refinement (frame 0) ====="
$PY $D/refine_pose.py --seq $SEQ

echo "===== [5/8] CoTracker3 point tracks ====="
$PY $D/run_cotracker.py --seq $SEQ

echo "===== [6/8] skeleton motion fit (SO(3) parent-relative accel + CoTracker3) ====="
# harder sequences (thin limbs, heavy self-occlusion) benefit from stronger regularization:
#   --w_reg_bone 0.5 --w_temp_bone 8 --w_accel_bone 60   (used for camel)
$PY $D/fit_video.py --seq $SEQ

echo "===== [7/8] render original + cyclic + track viz ====="
$PY $D/render_results.py --seq $SEQ
$PY $D/viz_tracks.py --seq $SEQ

echo "===== [8/8] skeleton visualization ====="
$PY $D/render_skeleton.py --seq $SEQ

echo "DONE -> $OUT/renders/{original_view,cyclic_view,skeleton_original,skeleton_cyclic}.mp4"
