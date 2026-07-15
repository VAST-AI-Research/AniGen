"""Media I/O for the recon-format scene data (video / depths / masks / cameras). Vendored into AniGen
so video_editing is self-contained (no external repo needed). Depends only on numpy / cv2 / imageio.

Recon dir layout consumed here: video.mp4, depths/NNNNN.exr (single-channel float), sky_mask/NNNNN.png,
dynamic_mask/NNNNN.png, cameras.npz (keys cam_c2w[F,4,4], intrinsics[F,4]=fx,fy,cx,cy)."""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # let cv2 read the .exr depth maps

from os import listdir, makedirs, path
import cv2
import imageio
import numpy as np


def load_video(video_path, desc="Loading video"):
    """Return (video[F,H,W,3] uint8 RGB, fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # cv2 reads BGR
    cap.release()
    return np.stack(frames, 0).astype(np.uint8), fps


def save_video(output_path, video, fps, quality=None, imageio_params=None):
    """Write [F,H,W,3] uint8 to mp4/gif via imageio."""
    imageio_params = dict(imageio_params or {})
    if quality is not None:
        imageio_params["quality"] = quality
    if path.splitext(output_path)[1] == ".gif":
        imageio_params["loop"] = 0
    else:
        imageio_params.setdefault("macro_block_size", 1)       # let H.264 handle non-16-multiple sizes
    makedirs(path.split(output_path)[0] or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, **imageio_params)
    for i in range(video.shape[0]):
        writer.append_data(video[i])
    writer.close()


def load_depths(input_folder, dtype=np.float32, desc="Loading depths"):
    """Return [F,H,W] float depth from the folder's NNNNN.exr files (single Y channel)."""
    files = sorted(f for f in listdir(input_folder) if f.endswith(".exr"))
    assert files, f"No EXR files found in {input_folder}"
    frames = []
    for f in files:
        d = cv2.imread(path.join(input_folder, f), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise ValueError(f"Failed to read EXR {f}; ensure OpenCV has OpenEXR support (OPENCV_IO_ENABLE_OPENEXR=1)")
        if d.ndim == 3:
            d = d[..., 0]
        frames.append(d)
    return np.stack(frames, 0).astype(dtype)


def save_masks(output_folder, masks):
    """Write [F,H,W] bool masks as NNNNN.png (0/255)."""
    makedirs(output_folder, exist_ok=True)
    for i in range(masks.shape[0]):
        cv2.imwrite(path.join(output_folder, f"{i:05d}.png"), masks[i].astype(np.uint8) * 255)


def load_masks(input_folder, desc="Loading masks"):
    """Return [F,H,W] bool from the folder's NNNNN.png files (>127 = True)."""
    files = sorted(f for f in listdir(input_folder) if f.endswith(".png"))
    assert files, f"No PNG files found in {input_folder}"
    return np.stack([cv2.imread(path.join(input_folder, f), cv2.IMREAD_GRAYSCALE) > 127 for f in files])


def save_cameras(output_path, cam_c2w, intrinsics):
    makedirs(path.split(output_path)[0] or ".", exist_ok=True)
    np.savez(output_path, cam_c2w=cam_c2w, intrinsics=intrinsics)


def load_cameras(input_path, force_same_first_frame=False):
    """Return (cam_c2w[F,4,4], intrinsics[F,4]) from a cameras.npz."""
    data = np.load(input_path)
    if force_same_first_frame:
        return data["cam_c2w_fsff"], data["intrinsics_fsff"]
    return data["cam_c2w"], data["intrinsics"]
