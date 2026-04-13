import os


HF_REPO_ID = "VAST-AI/AniGen"
CKPTS_DIR = "ckpts"


def ensure_ckpts(local_dir: str = ".") -> None:
    """
    If the local ``ckpts/`` directory does not exist, download it from the
    Hugging Face model repo ``VAST-AI/AniGen``.
    """
    ckpts_path = os.path.join(local_dir, CKPTS_DIR)
    if os.path.isdir(ckpts_path):
        return

    print(f"'{ckpts_path}' not found locally. "
          f"Downloading checkpoints from HuggingFace ({HF_REPO_ID}) ...")

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{CKPTS_DIR}/**"],
        local_dir=local_dir,
    )

    print("Checkpoint download complete.")
