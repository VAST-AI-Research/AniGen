"""Shared paths for the AniGen video-editing pipeline.

This package lives at ``<repos>/AniGen/video_editing``. The point-cloud renderer and media IO are
vendored locally (``_pointcloud.py`` / ``_media.py``), so the only external repo is the sibling
**FreeOrbit4D**, which provides the VACE generation backend. Its root is auto-detected as a sibling of
AniGen and can be overridden with the ``FREEORBIT4D_ROOT`` environment variable.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIGEN_ROOT = os.path.dirname(_HERE)                       # <repos>/AniGen
_REPOS = os.path.dirname(ANIGEN_ROOT)                      # <repos>

FREEORBIT4D_ROOT = os.environ.get("FREEORBIT4D_ROOT", os.path.join(_REPOS, "FreeOrbit4D"))
RESULTS_DIR = os.path.join(ANIGEN_ROOT, "results")         # AniGen per-asset fits live here
