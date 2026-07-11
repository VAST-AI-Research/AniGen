#!/bin/bash
# End-to-end monocular 4D reconstruction for one DAVIS-style sequence.
# Usage:  CUDA_VISIBLE_DEVICES=0 bash anigen/4drecon/run_all.sh [SEQ]   (default bear)
# Prereqs: frames at $ANIGEN_DAVIS_ROOT/JPEGImages/<SEQ>/*.jpg, then prep_video.py --seq <SEQ>;
#          VGGT-Omega + LoMa checked out (see README).
# Env: REVERSE=1 (anchor last frame), SMOOTH_ROOT=<s> / SMOOTH_TIME=<s> (high-fps de-jitter),
#      CLEAN_FACES=<t> (drop faces spanning >t bones; 0 disables), SKIP_GLB=1, ANIGEN_SS_FLOW=<path>.
set -e
cd "$(dirname "$0")/../.."
SEQ=${1:-bear}
PY=${PYTHON:-python}
D=anigen/4drecon
OUT=results/$SEQ
RG=$OUT/rig_fit.npz
MOT=$OUT/motion_fit.npz
SSFLOW=${ANIGEN_SS_FLOW:-ckpts/anigen/ss_flow_solo}
REV=${REVERSE:-0}
if [ "$REV" = "1" ]; then POSE_FRAME="--frame -1"; FIT_REV="--reverse 1"; else POSE_FRAME=""; FIT_REV=""; fi
SMOOTH=${SMOOTH_ROOT:-0}; [ "$SMOOTH" != "0" ] && FIT_SMOOTH="--smooth_root $SMOOTH" || FIT_SMOOTH=""

echo "===== [0/7] generate rigged mesh ($SEQ) ====="
if [ "${SKIP_GLB:-0}" != "1" ]; then
  $PY example.py --image_path assets/${SEQ}_rgba.png --output_name $SEQ --output_dir results/ --ss_flow_path $SSFLOW
fi

echo "===== [1/7] export rig + symmetrize colours + face split ====="
$PY $D/export_rig.py --image_path assets/${SEQ}_rgba.png --out $OUT/rig.npz --ss_flow_path $SSFLOW
$PY $D/symmetrize_colors.py --rig $OUT/rig.npz
$PY $D/clean_faces.py --rig $OUT/rig.npz --threshold ${CLEAN_FACES:-3}

echo "===== [2/7] render static views for pose init ====="
$PY $D/render_views.py --rig $OUT/rig.npz --out $OUT/views

echo "===== [3/7] VGGT-Omega camera + coarse pose ====="
$PY $D/pose_init.py --rig $OUT/rig.npz --views $OUT/views --real_rgba assets/${SEQ}_rgba.png --out $OUT/vggt_pose.npz

echo "===== [4/7] refine anchor-frame rigid pose -> pose0.npz ====="
$PY $D/refine_pose.py --seq $SEQ $POSE_FRAME

echo "===== [5/7] LoMa per-frame pose fit -> motion_loma.npz ====="
$PY $D/fit_video_loma.py --seq $SEQ $FIT_REV $FIT_SMOOTH

echo "===== [6/7] joint skin+texture(+shape) refinement -> rig_fit.npz + motion_fit.npz ====="
SMT=${SMOOTH_TIME:-0}; [ "$SMT" != "0" ] && REF_SMOOTH="--smooth_time $SMT" || REF_SMOOTH=""
$PY $D/refine_skin_texture_loma.py --seq $SEQ $REF_SMOOTH

echo "===== [7/7] render original + cyclic + skeleton + showcase ====="
$PY $D/render_results.py  --seq $SEQ --rig $RG --motion $MOT
$PY $D/render_skeleton.py --seq $SEQ --rig $RG --motion $MOT
$PY $D/render_showcase.py --seq $SEQ --rig $RG --motion $MOT

echo "DONE -> $OUT/renders/*.mp4"
