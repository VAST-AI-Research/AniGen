#!/usr/bin/env python3
"""Ex 1 - Unitree G1 -> jumping jacks (+ lean/recover), camera slowly orbiting. Run: python examples/robot_g1_jumping_jacks.py

Point ANIGEN_EDIT_DATA at the folder holding your recon dirs (default: video_editing/data)."""
import subprocess, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("ANIGEN_EDIT_DATA", f"{H}/data")
subprocess.run(["python3", f"{H}/edit.py",
    "--asset", "unitree_g1_1", "--name", "robot_g1_jj",
    "--recon", f"{DATA}/robot_g1_edit/recon",
    "--mask", f"{DATA}/robot_g1_edit/recon/dynamic_mask",
    "--motion_source", "robot_jumping_jacks",
    "--camera_source", "slowly orbit the robot about 25 degrees", "--asset-scale", "0.8",
    "--gpu", os.environ.get("GPU", "0")], check=True)
