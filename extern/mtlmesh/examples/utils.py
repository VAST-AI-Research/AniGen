import os
import requests
import tarfile
import trimesh

BUNNY_ZIP_URL = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
CACHE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "cache")


def download_file(url, path):
    print(f"Downloading from {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Saved to {path}")


def extract_tar(path, extract_dir):
    print(f"Extracting {path} to {extract_dir}...")
    with tarfile.open(path) as tar:
        tar.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def get_bunny() -> trimesh.Trimesh:
    BUNNY_PATH = os.path.join(CACHE_DIR, "bunny", "reconstruction", "bun_zipper.ply")
    if not os.path.exists(BUNNY_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        download_file(BUNNY_ZIP_URL, os.path.join(CACHE_DIR, "bunny.tar.gz"))
        extract_tar(os.path.join(CACHE_DIR, "bunny.tar.gz"), CACHE_DIR)
    return trimesh.load(BUNNY_PATH)
