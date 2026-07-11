#!/bin/bash
# One-command monocular 4D reconstruction from a raw video or an image folder.
#
#   bash anigen/4drecon/run.sh --video clip.mp4        [--mask mask.mp4]   [--prompt "robot"] [--name myseq]
#   bash anigen/4drecon/run.sh --frames frames_dir/    [--mask masks_dir/] [--prompt "robot"] [--name myseq]
#
# --video / --frames : input (one required).  --mask : optional foreground mask (video OR image folder);
#   if omitted, masks are produced automatically with SAM3 (text --prompt, default "object").
# Also honours run_all.sh env: CLEAN_FACES, SMOOTH_ROOT, SMOOTH_TIME, REVERSE, ANIGEN_SS_FLOW.
set -e
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
VIDEO=""; FRAMES=""; MASK=""; PROMPT="object"; NAME=""
while [ $# -gt 0 ]; do case "$1" in
  --video)  VIDEO="$2";  shift 2;;
  --frames) FRAMES="$2"; shift 2;;
  --mask)   MASK="$2";   shift 2;;
  --prompt) PROMPT="$2"; shift 2;;
  --name)   NAME="$2";   shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done
[ -z "$VIDEO$FRAMES" ] && { echo "need --video <file> or --frames <dir>"; exit 1; }
if [ -n "$NAME" ]; then SEQ="$NAME"; else SRC="${VIDEO:-$FRAMES}"; SEQ=$(basename "${SRC%/}"); SEQ="${SEQ%.*}"; fi
DAVIS=${ANIGEN_DAVIS_ROOT:-data/davis}
JDIR="$DAVIS/JPEGImages/$SEQ"; ADIR="$DAVIS/Annotations/$SEQ"
mkdir -p "$JDIR"

echo "===== [prep] frames -> $JDIR ====="
$PY - "$VIDEO" "$FRAMES" "$JDIR" <<'PY'
import sys, os, glob, imageio.v2 as imageio
from PIL import Image
video, frames, jdir = sys.argv[1:4]
if video:
    for i, fr in enumerate(imageio.get_reader(video)):
        Image.fromarray(fr).convert("RGB").save(os.path.join(jdir, f"{i:05d}.jpg"))
    print("extracted", i + 1, "frames from video")
else:
    fs = sorted(glob.glob(os.path.join(frames, "*.jpg")) + glob.glob(os.path.join(frames, "*.png")))
    for i, f in enumerate(fs):
        Image.open(f).convert("RGB").save(os.path.join(jdir, f"{i:05d}.jpg"))
    print("copied", len(fs), "frames")
PY

if [ -n "$MASK" ]; then
  echo "===== [prep] provided mask -> $ADIR + assets/${SEQ}_rgba.png ====="
  mkdir -p "$ADIR"
  $PY - "$MASK" "$ADIR" "$JDIR" "assets/${SEQ}_rgba.png" <<'PY'
import sys, os, glob, imageio.v2 as imageio, numpy as np
from PIL import Image
mask, adir, jdir, rgba = sys.argv[1:5]
if os.path.isdir(mask):
    ms = sorted(glob.glob(os.path.join(mask, "*.png")) + glob.glob(os.path.join(mask, "*.jpg")))
    for i, f in enumerate(ms):
        Image.open(f).convert("L").save(os.path.join(adir, f"{i:05d}.png"))
else:
    for i, fr in enumerate(imageio.get_reader(mask)):
        Image.fromarray(np.asarray(Image.fromarray(fr).convert("L"))).save(os.path.join(adir, f"{i:05d}.png"))
f0 = sorted(glob.glob(os.path.join(jdir, "*.jpg")))[0]
m0 = sorted(glob.glob(os.path.join(adir, "*.png")))[0]
im = Image.open(f0).convert("RGB"); mk = Image.open(m0).convert("L").resize(im.size)
Image.fromarray(np.dstack([np.array(im), (np.array(mk) > 127).astype(np.uint8) * 255]), "RGBA").save(rgba)
print("wrote masks + rgba ->", rgba)
PY
else
  echo "===== [prep] no mask given -> SAM3 auto-mask (prompt='$PROMPT') ====="
  $PY anigen/4drecon/prep_video.py --seq "$SEQ" --model sam3 --prompt "$PROMPT"
fi

echo "===== [run] full pipeline for $SEQ ====="
bash anigen/4drecon/run_all.sh "$SEQ"
echo "DONE -> results/$SEQ/renders/"
