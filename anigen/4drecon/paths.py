"""Central path / data configuration for the 4drecon library.

Everything is resolved from environment variables with repo-relative defaults, so no machine-
specific paths are baked in. Override any of these before running:

    ANIGEN_DAVIS_ROOT        DAVIS-layout data root: <root>/JPEGImages/<seq>, <root>/Annotations/<seq>
    ANIGEN_4DRECON_TP        third-party checkout root (cloned separately; see README)

Stage outputs are written to results/<seq>/ (relative to the repo root).
    COTRACKER_REPO / COTRACKER_CKPT
    VGGT_OMEGA_REPO / VGGT_OMEGA_CKPT
    SPATRACKER_REPO / SPATRACKER_FRONT_CKPT / SPATRACKER_OFFLINE_CKPT
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))


def _env(name, default):
    return os.environ.get(name, default)


DAVIS_ROOT = _env("ANIGEN_DAVIS_ROOT", os.path.join(REPO_ROOT, "data", "davis"))
THIRD_PARTY = _env("ANIGEN_4DRECON_TP", os.path.join(_HERE, "third_party"))

# --- optional third-party trackers / pose models (cloned by the user; see README) ---
COTRACKER_REPO = _env("COTRACKER_REPO", os.path.join(THIRD_PARTY, "co-tracker"))
COTRACKER_CKPT = _env("COTRACKER_CKPT", os.path.join(COTRACKER_REPO, "checkpoints", "scaled_offline.pth"))

VGGT_OMEGA_REPO = _env("VGGT_OMEGA_REPO", os.path.join(THIRD_PARTY, "vggt-omega"))
VGGT_OMEGA_CKPT = _env("VGGT_OMEGA_CKPT", os.path.join(THIRD_PARTY, "vggt-omega", "ckpt", "vggt_omega_1b_512.pt"))

SPATRACKER_REPO = _env("SPATRACKER_REPO", os.path.join(THIRD_PARTY, "SpaTrackerV2"))
SPATRACKER_FRONT_CKPT = _env("SPATRACKER_FRONT_CKPT", os.path.join(THIRD_PARTY, "SpatialTrackerV2_Front"))
SPATRACKER_OFFLINE_CKPT = _env("SPATRACKER_OFFLINE_CKPT", os.path.join(THIRD_PARTY, "SpatialTrackerV2-Offline"))


def davis_paths(seq):
    """(frames_dir, ann_dir) for a DAVIS sequence under DAVIS_ROOT."""
    return (os.path.join(DAVIS_ROOT, "JPEGImages", seq),
            os.path.join(DAVIS_ROOT, "Annotations", seq))
