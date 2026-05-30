import os, importlib

def test_bootstrap_sets_env_before_pipeline_import():
    for k in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_BACKEND", "PYTORCH_ENABLE_MPS_FALLBACK"):
        os.environ.pop(k, None)
    import anigen_mps
    anigen_mps.configure_mps_environment()
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"
    assert os.environ["ATTN_BACKEND"] == "naive"          # dense: real matmul+softmax
    assert os.environ["SPARSE_ATTN_BACKEND"] == "naive"   # sparse: our fp32 fallback
    assert os.environ["SPARSE_BACKEND"] == "spconv"       # selects conv module family
    assert os.environ["SPARSE_CONV_BACKEND"] == "flex_gemm"

def test_resolve_device_mps():
    import torch, anigen_mps
    dev = anigen_mps.resolve_device("mps")
    assert dev.type in ("mps", "cpu")  # cpu only if MPS truly unavailable
