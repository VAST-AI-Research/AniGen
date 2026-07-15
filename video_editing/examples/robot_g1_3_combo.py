#!/usr/bin/env python3
"""Ex 2 - a second Unitree G1 -> hands-on-hips/wave/waist/squat combo, camera orbiting. Run: python examples/robot_g1_3_combo.py

Point ANIGEN_EDIT_DATA at the folder holding your recon dirs (default: video_editing/data)."""
import subprocess, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ANIGEN_EDIT_DATA", f"{H}/data")
subprocess.run(["python3", f"{H}/edit.py",
    "--asset", "unitree_g1_3", "--name", "robot_g1_3_combo", "--frames", "41",
    "--recon", f"{DATA}/robot_g1_3_edit/recon",
    "--mask", f"{DATA}/robot_g1_3_edit/annotations", "--mask-kind", "gt-mask",
    "--motion_source", "robot_combo",
    "--camera_source", "slowly orbit the robot about 25 degrees",
    "--gpu", os.environ.get("GPU", "0")], check=True)
