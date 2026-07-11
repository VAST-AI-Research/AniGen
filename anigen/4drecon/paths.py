"""Central path / data configuration for the 4drecon library.

Everything is resolved from environment variables with repo-relative defaults, so no machine-
specific paths are baked in. Override any of these before running:

    ANIGEN_DAVIS_ROOT        DAVIS-layout data root: <root>/JPEGImages/<seq>, <root>/Annotations/<seq>
    ANIGEN_4DRECON_TP        third-party checkout root (cloned separately; see README)

Stage outputs are written to results/<seq>/ (relative to the repo root).
    VGGT_OMEGA_REPO / VGGT_OMEGA_CKPT
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))


def _env(name, default):
    return os.environ.get(name, default)


DAVIS_ROOT = _env("ANIGEN_DAVIS_ROOT", os.path.join(REPO_ROOT, "data", "davis"))
THIRD_PARTY = _env("ANIGEN_4DRECON_TP", os.path.join(REPO_ROOT, "extensions"))

# --- VGGT-Omega camera / pose model (cloned by the user; see README) ---
VGGT_OMEGA_REPO = _env("VGGT_OMEGA_REPO", os.path.join(THIRD_PARTY, "vggt-omega"))
VGGT_OMEGA_CKPT = _env("VGGT_OMEGA_CKPT", os.path.join(THIRD_PARTY, "vggt-omega", "ckpt", "vggt_omega_1b_512.pt"))


def davis_paths(seq):
    """(frames_dir, ann_dir) for a DAVIS sequence under DAVIS_ROOT."""
    return (os.path.join(DAVIS_ROOT, "JPEGImages", seq),
            os.path.join(DAVIS_ROOT, "Annotations", seq))
