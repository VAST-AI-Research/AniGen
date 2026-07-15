#!/usr/bin/env python3
"""Ex 3 - camel -> rear up, original (moving) camera. Run: python examples/camel_rear_up.py

Point ANIGEN_EDIT_DATA at the folder holding your recon dirs (default: video_editing/data)."""
import subprocess, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ANIGEN_EDIT_DATA", f"{H}/data")
subprocess.run(["python3", f"{H}/edit.py",
    "--asset", "camel", "--name", "camel_rear_up",
    "--recon", f"{DATA}/camel_edit/recon",
    "--mask", f"{DATA}/camel_edit/recon/dynamic_mask_both",
    "--motion_source", "camel_rear_up", "--camera_source", "default",
    "--gpu", os.environ.get("GPU", "0")], check=True)
