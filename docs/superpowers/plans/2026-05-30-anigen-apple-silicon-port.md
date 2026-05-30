# AniGen Apple Silicon Inference Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run AniGen's unmodified image-to-rigged-3D inference pipeline natively on Apple Silicon (MPS/Metal) across all model variants, by reusing the Metal toolkit from the Pixal3D port.

**Architecture:** A thin `anigen_mps` bootstrap applies device shims (env config, attention routing, fp32 upcast) before importing the pipeline, plus a small number of targeted in-tree edits where monkeypatching is fragile (sparse-conv backend, device defaults, rasterizer backend). CUDA-only ops are replaced: spconv→flex_gemm (Metal), nvdiffrast/utils3d rasterizer→CPU fallback, pytorch3d KNN→scipy cKDTree, attention→naive fp32 (MPS SDPA is banned). Correctness-first: get valid GLBs on all variants with CPU fallbacks on the hard gaps; optimization is a separate Phase 2 plan.

**Tech Stack:** Python 3.10, PyTorch 2.12 (MPS), `flex_gemm`/`mtldiffrast`/`cumesh`/`mtlbvh` (vendored Metal forks from `github.com/pawel-mazurkiewicz/<pkg>`), scipy, trimesh, xatlas, pymeshfix.

**Spec:** `docs/superpowers/specs/2026-05-30-anigen-apple-silicon-port-design.md`

---

## File Structure

**New files:**
- `requirements-mac.txt` — Mac dependency pins (no CUDA wheels)
- `extern/` — vendored Metal packages (`mtlgemm`, `mtldiffrast`, `mtlmesh`, `mtlbvh`) + `extern/README.md` provenance
- `anigen_mps/__init__.py` — bootstrap: env config + device resolver, imported before the pipeline
- `anigen_mps/knn_cpu.py` — scipy cKDTree drop-in for `pytorch3d.ops.knn_points` / `ball_query`
- `example_mps.py` — Mac CLI entrypoint (bootstrap → run pipeline)
- `anigen/modules/sparse/conv/conv_flex_gemm.py` — flex_gemm sparse-conv backend
- `anigen/modules/sparse/attention/fallback_attn.py` — naive fp32 sparse attention
- `tests/mps/` — numerical-parity micro-tests for the shims
- `scripts/probe_sparse_ops.py` — inference-time sparse-conv op census (decision gate)

**Modified files (targeted edits):**
- `anigen/modules/sparse/__init__.py` — accept `BACKEND='flex_gemm'`
- `anigen/modules/sparse/attention/full_attn.py` — wire `naive` into dispatcher
- `anigen/modules/sparse/attention/__init__.py` — accept `naive` backend value
- `example.py:77` — `.cuda()` → `.to(args.device)`
- `anigen/pipelines/base.py:67` — device-aware `.to()`
- `anigen/renderers/mesh_renderer.py:52` — device-aware rasterizer context
- `anigen/utils/postprocessing_utils.py` — `RastContext(backend=...)` device-aware

---

## Phase 0 — Environment & Vendoring

### Task 1: Mac Python 3.10 venv + torch 2.12 + MPS smoke check

**Files:**
- Create: `requirements-mac.txt`
- Create: `scripts/setup_mac.sh`

- [ ] **Step 1: Write `requirements-mac.txt`**

```text
# AniGen Apple Silicon inference environment — Python 3.10
# torch>=2.11 required on Darwin for flex_gemm (at::mps::dispatch_sync_with_rethrow).
# CUDA-only wheels (spconv-cuXX, nvdiffrast, flash-attn, CUDA pytorch3d) are intentionally absent.
torch==2.12.0
torchvision==0.27.0
numpy>=1.24,<2
scipy==1.15.3
transformers==5.9.0
safetensors==0.7.0
huggingface-hub==1.16.4
einops==0.8.2
trimesh==4.10.1
xatlas==0.0.11
pymeshfix
pyvista
python-igraph
opencv-python
pillow
imageio
tqdm
omegaconf
easydict
utils3d @ git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
```

- [ ] **Step 2: Write `scripts/setup_mac.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY310="${PY310:-/opt/homebrew/opt/python@3.10/bin/python3.10}"

"$PY310" -m venv "$ROOT/.venv-mac"
"$ROOT/.venv-mac/bin/pip" install -U pip wheel setuptools
"$ROOT/.venv-mac/bin/pip" install -r "$ROOT/requirements-mac.txt"

# Vendored Metal packages (Task 2 must have populated extern/).
for p in mtlbvh mtlmesh mtlgemm mtldiffrast; do
  if [ -d "$ROOT/extern/$p" ]; then
    "$ROOT/.venv-mac/bin/pip" install -e "$ROOT/extern/$p" --no-build-isolation
  fi
done
echo "setup_mac.sh complete"
```

- [ ] **Step 3: Create the venv and verify MPS is available**

Run:
```bash
bash scripts/setup_mac.sh
.venv-mac/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
Expected: prints `2.12.0 True`. (The `extern/` loop is a no-op until Task 2 — that is fine.)

- [ ] **Step 4: Commit**

```bash
git add requirements-mac.txt scripts/setup_mac.sh
git commit -m "build(mac): add Python 3.10 / torch 2.12 Apple Silicon env"
```

---

### Task 2: Vendor the four Metal forks into `extern/` and verify imports

**Files:**
- Create: `extern/mtlgemm`, `extern/mtldiffrast`, `extern/mtlmesh`, `extern/mtlbvh` (git clones of the user's forks)
- Create: `extern/README.md`

- [ ] **Step 1: Clone the four forks into `extern/`**

Run:
```bash
mkdir -p extern
for p in mtlbvh mtlmesh mtlgemm mtldiffrast; do
  git clone "git@github.com:pawel-mazurkiewicz/${p}.git" "extern/${p}"
done
```
Expected: four package directories under `extern/`. (`natten-mps` and `o_voxel` are intentionally NOT vendored — AniGen uses neither.)

- [ ] **Step 2: Write `extern/README.md` (provenance)**

```markdown
# Vendored Metal packages

Forks of Pedro Naugusto's Apple Silicon Metal libraries, carrying our fixes
not yet upstreamed. Source: github.com/pawel-mazurkiewicz/<pkg>.

| dir | import name | replaces |
|---|---|---|
| mtlgemm | flex_gemm | spconv sparse conv |
| mtldiffrast | mtldiffrast | nvdiffrast |
| mtlmesh | cumesh | cumesh mesh post-processing |
| mtlbvh | mtlbvh | CUDA BVH (used by mtlmesh) |

Drop the fork pin and switch to upstream if/when these fixes merge.
Requires the Xcode Metal toolchain (`xcrun`) to compile `.metal` -> `.metallib`.
```

- [ ] **Step 3: Install editable and verify imports**

Run:
```bash
for p in mtlbvh mtlmesh mtlgemm mtldiffrast; do
  .venv-mac/bin/pip install -e "extern/$p" --no-build-isolation
done
.venv-mac/bin/python -c "import flex_gemm, mtldiffrast, cumesh, mtlbvh; print('metal imports OK')"
```
Expected: prints `metal imports OK`. If a `.metallib` compile fails, confirm Xcode command-line tools + Metal toolchain are installed (`xcrun --find metal`).

- [ ] **Step 4: Verify flex_gemm exposes the submanifold conv API**

Run:
```bash
.venv-mac/bin/python -c "from flex_gemm.ops.spconv import sparse_submanifold_conv3d, set_algorithm, set_hashmap_ratio; print('flex_gemm spconv API OK')"
```
Expected: prints `flex_gemm spconv API OK`.

- [ ] **Step 5: Commit**

```bash
git add extern/README.md .gitmodules 2>/dev/null || git add extern/README.md
git commit -m "build(mac): vendor mtlgemm/mtldiffrast/mtlmesh/mtlbvh Metal forks"
```
(If you cloned as plain directories rather than submodules, also `git add extern/<pkg>` per your preference for how to track vendored sources.)

---

## Phase 1 — Device-Agnostic Plumbing

### Task 3: `anigen_mps` bootstrap (env config + device resolver)

**Files:**
- Create: `anigen_mps/__init__.py`
- Test: `tests/mps/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/mps/test_bootstrap.py
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
```

- [ ] **Step 2: Run it to confirm failure**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anigen_mps'`.

- [ ] **Step 3: Implement `anigen_mps/__init__.py`**

```python
"""Apple Silicon bootstrap for AniGen inference.

Import and call configure_mps_environment() BEFORE importing anigen.* so that
the sparse-conv / attention backend env vars are read at module import time.
"""
import os


def configure_mps_environment() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Dense attention: AniGen's 'naive' path is real fp32 matmul+softmax (not SDPA). Safe on MPS.
    os.environ.setdefault("ATTN_BACKEND", "naive")
    # Sparse attention: route to our fp32 fallback (Task 7). MPS fused SDPA is banned
    # (the >~18-20k-token cliff returns catastrophically wrong output — proven in Pixal3D).
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "naive")
    # Sparse conv: select the spconv-family module (basic.py), but route the actual
    # kernel to flex_gemm via SPARSE_CONV_BACKEND (Task 9 reads this).
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("SPCONV_ALGO", "native")


def resolve_device(requested: str = "mps"):
    import torch
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[anigen_mps] MPS unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


configure_mps_environment()
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_bootstrap.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add anigen_mps/__init__.py tests/mps/test_bootstrap.py
git commit -m "feat(mps): add anigen_mps bootstrap (env + device resolver)"
```

---

### Task 4: Device-hardcoding sweep (honor the device argument)

**Files:**
- Modify: `example.py:77`
- Modify: `anigen/pipelines/base.py:67`
- Modify: `anigen/renderers/mesh_renderer.py:52`
- Modify: `anigen/utils/postprocessing_utils.py` (RastContext sites)

- [ ] **Step 1: Fix `example.py` — replace the hardcoded `.cuda()`**

In `example.py`, change line 77:
```python
# before:
pipeline.cuda()
# after:
pipeline.to(args.device)
```

- [ ] **Step 2: Fix `anigen/pipelines/base.py:67`**

```python
# before:
self.to(torch.device("cuda"))
# after:
self.to(torch.device(self.device if hasattr(self, "device") else "cuda"))
```
(`AnigenImageTo3DPipeline.__init__` already receives `device`; ensure it is stored as `self.device`. If not, add `self.device = device` in `anigen/pipelines/anigen_image_to_3d.py` where `device` is accepted.)

- [ ] **Step 3: Fix `anigen/renderers/mesh_renderer.py:52` — device-aware rasterizer context**

```python
# before:
self.glctx = dr.RasterizeCudaContext(device=device)
# after:
if str(device).startswith("cuda"):
    self.glctx = dr.RasterizeCudaContext(device=device)
else:
    # MPS/CPU: nvdiffrast has no Metal backend; use mtldiffrast (Task 11 wires this).
    import mtldiffrast.torch as mdr
    self.glctx = mdr.RasterizeContext(device=device)
```
(The exact `mtldiffrast` context class is confirmed in Task 11; if its constructor differs, adjust there.)

- [ ] **Step 4: Make `utils3d` RastContext backend device-aware in `postprocessing_utils.py`**

Replace every `utils3d.torch.RastContext(backend='cuda')` (lines ~60, 78, 249, 354, 359) with a module-level helper. Add near the top of `anigen/utils/postprocessing_utils.py`:
```python
import torch as _torch
def _rast_backend():
    return 'cuda' if _torch.cuda.is_available() else 'cpu'
```
Then replace each call:
```python
# before:
rastctx = utils3d.torch.RastContext(backend='cuda')
# after:
rastctx = utils3d.torch.RastContext(backend=_rast_backend())
```

- [ ] **Step 5: Verify the inference modules import under the bootstrap**

Run:
```bash
.venv-mac/bin/python -c "import anigen_mps; from anigen.pipelines import anigen_image_to_3d; print('pipeline import OK')"
```
Expected: prints `pipeline import OK` with no CUDA/import errors. (spconv/flash-attn imports are lazy or guarded; if an import fails here, note which and confirm it is addressed by Task 7/9 before proceeding.)

- [ ] **Step 6: Commit**

```bash
git add example.py anigen/pipelines/base.py anigen/pipelines/anigen_image_to_3d.py anigen/renderers/mesh_renderer.py anigen/utils/postprocessing_utils.py
git commit -m "fix(mps): honor device argument across inference path"
```

---

## Phase 2 — Attention

### Task 5: Dense attention via `naive` backend (parity check)

**Files:**
- Test: `tests/mps/test_dense_attn_naive.py`

AniGen's dense `_naive_sdpa` (`anigen/modules/attention/full_attn.py:23-35`) is real fp32 matmul+softmax and is selected by `ATTN_BACKEND=naive` (set by the bootstrap). No code change needed — only a parity guard.

- [ ] **Step 1: Write the parity test**

```python
# tests/mps/test_dense_attn_naive.py
import os, math, torch, importlib

def test_naive_matches_reference_sdpa():
    os.environ["ATTN_BACKEND"] = "naive"
    import anigen.modules.attention.full_attn as fa
    importlib.reload(fa)
    N, L, H, C = 2, 64, 4, 32
    q = torch.randn(N, L, H, C); k = torch.randn(N, L, H, C); v = torch.randn(N, L, H, C)
    out = fa.scaled_dot_product_attention(q, k, v)
    # reference: explicit fp32 attention
    qr, kr, vr = (t.permute(0, 2, 1, 3) for t in (q, k, v))
    w = torch.softmax(qr @ kr.transpose(-2, -1) / math.sqrt(C), dim=-1)
    ref = (w @ vr).permute(0, 2, 1, 3)
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
```
(If the public function name differs, use the actual exported name from `full_attn.py`'s `__all__`.)

- [ ] **Step 2: Run — confirm it passes on CPU first, then MPS**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_dense_attn_naive.py -v`
Expected: PASS. Then repeat forcing MPS tensors (append `.to('mps')` to q/k/v in a second test) and confirm max abs error stays < 1e-4.

- [ ] **Step 3: Commit**

```bash
git add tests/mps/test_dense_attn_naive.py
git commit -m "test(mps): dense naive attention parity guard"
```

---

### Task 6: Sparse attention — naive fp32 fallback

**Files:**
- Create: `anigen/modules/sparse/attention/fallback_attn.py`
- Modify: `anigen/modules/sparse/attention/full_attn.py`
- Modify: `anigen/modules/sparse/attention/__init__.py` (and/or `anigen/modules/sparse/__init__.py` `set_attn`)
- Test: `tests/mps/test_sparse_attn_naive.py`

AniGen's sparse attention (`anigen/modules/sparse/attention/full_attn.py`) only supports `xformers`/`flash_attn` — neither runs on MPS. We add a `naive` path that reproduces the same variable-length, block-diagonal contract using the per-batch `q_seqlen`/`kv_seqlen` already computed in that function.

- [ ] **Step 1: Write the failing parity test**

```python
# tests/mps/test_sparse_attn_naive.py
import math, torch
from anigen.modules.sparse.attention.fallback_attn import naive_varlen_attention

def _ref_blockdiag(q, k, v, q_seqlen, kv_seqlen):
    # q: [Tq, H, C], k/v: [Tkv, H, C] — independent attention per batch segment.
    outs = []
    qo = ko = 0
    for sq, sk in zip(q_seqlen, kv_seqlen):
        qi = q[qo:qo+sq].permute(1, 0, 2)            # [H, sq, C]
        ki = k[ko:ko+sk].permute(1, 0, 2)            # [H, sk, C]
        vi = v[ko:ko+sk].permute(1, 0, 2)            # [H, sk, C]
        w = torch.softmax(qi @ ki.transpose(-2, -1) / math.sqrt(qi.shape[-1]), dim=-1)
        outs.append((w @ vi).permute(1, 0, 2))       # [sq, H, C]
        qo += sq; ko += sk
    return torch.cat(outs, dim=0)

def test_naive_varlen_matches_reference():
    H, C = 4, 16
    q_seqlen = [30, 50]; kv_seqlen = [30, 50]
    q = torch.randn(sum(q_seqlen), H, C)
    k = torch.randn(sum(kv_seqlen), H, C)
    v = torch.randn(sum(kv_seqlen), H, C)
    out = naive_varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    ref = _ref_blockdiag(q, k, v, q_seqlen, kv_seqlen)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_sparse_attn_naive.py -v`
Expected: FAIL — `ModuleNotFoundError` / `naive_varlen_attention` undefined.

- [ ] **Step 3: Implement `fallback_attn.py`**

```python
# anigen/modules/sparse/attention/fallback_attn.py
"""Naive fp32 variable-length attention for MPS (replaces flash-attn varlen).

Contract matches anigen.modules.sparse.attention.full_attn: q is [Tq, H, C],
k/v are [Tkv, H, C], with block-diagonal (per-batch) attention defined by the
q_seqlen / kv_seqlen segment lengths. Computed segment-by-segment in fp32 to
avoid the MPS fused-SDPA cliff. Output is [Tq, H, Cv].
"""
import math
from typing import List
import torch


def naive_varlen_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    q_seqlen: List[int], kv_seqlen: List[int],
) -> torch.Tensor:
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    H, C = q.shape[1], q.shape[2]
    Cv = v.shape[2]
    scale = 1.0 / math.sqrt(C)
    out = torch.empty(q.shape[0], H, Cv, device=q.device, dtype=q.dtype)
    qo = ko = 0
    for sq, sk in zip(q_seqlen, kv_seqlen):
        qi = q[qo:qo + sq].permute(1, 0, 2).float()          # [H, sq, C]
        ki = k[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, C]
        vi = v[ko:ko + sk].permute(1, 0, 2).float()          # [H, sk, Cv]
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        seg = torch.matmul(weights, vi).permute(1, 0, 2)     # [sq, H, Cv]
        out[qo:qo + sq] = seg.to(out.dtype)
        qo += sq; ko += sk
    return out
```

- [ ] **Step 4: Wire `naive` into the sparse dispatcher**

In `anigen/modules/sparse/attention/full_attn.py`, relax the import guard (lines 6-11) so `naive` does not raise:
```python
# before:
if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
else:
    raise ValueError(f"Unknown attention module: {ATTN}")
# after:
if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
elif ATTN == 'naive':
    from .fallback_attn import naive_varlen_attention
else:
    raise ValueError(f"Unknown attention module: {ATTN}")
```
Then add a `naive` branch in the backend-dispatch block (after the `flash_attn` branch, before the final `else`). It must unpack the packed `qkv`/`kv` forms exactly like the existing branches:
```python
    elif ATTN == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)          # each [T, H, C]
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        # num_all_args == 3: q, k, v already separated above
        out = naive_varlen_attention(q, k, v, q_seqlen, kv_seqlen)
```

- [ ] **Step 5: Accept `naive` as a valid sparse backend value**

In `anigen/modules/sparse/__init__.py`, extend `set_attn` and the env parsing so `'naive'` is accepted (currently only `'xformers'`/`'flash_attn'`):
```python
def set_attn(attn):
    global ATTN
    assert attn in ('xformers', 'flash_attn', 'naive'), f"Unknown sparse attn backend: {attn}"
    ATTN = attn
```

- [ ] **Step 6: Run the parity test**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_sparse_attn_naive.py -v`
Expected: PASS.

- [ ] **Step 7: High-token correctness check (the SDPA-cliff regression)**

Add to the test file and run:
```python
def test_naive_varlen_large_token_count():
    H, C = 8, 64
    q_seqlen = kv_seqlen = [21504]   # exceeds the ~18-20k MPS SDPA cliff
    q = torch.randn(21504, H, C); k = torch.randn(21504, H, C); v = torch.randn(21504, H, C)
    out = naive_varlen_attention(q, k, v, q_seqlen, kv_seqlen)
    assert torch.isfinite(out).all()
    assert out.abs().mean() > 0  # not a variance-collapsed near-zero blob
```
Expected: PASS (and finite). If memory-bound on MPS, add query-chunk tiling inside `naive_varlen_attention` (mirror Pixal3D's `_NA2D_QUERY_CHUNK=2048` loop) and re-run.

- [ ] **Step 8: Commit**

```bash
git add anigen/modules/sparse/attention/fallback_attn.py anigen/modules/sparse/attention/full_attn.py anigen/modules/sparse/__init__.py tests/mps/test_sparse_attn_naive.py
git commit -m "feat(mps): naive fp32 fallback for sparse attention (bans MPS SDPA)"
```

---

## Phase 3 — Sparse Convolution (highest risk)

### Task 7: Census which sparse-conv ops AniGen invokes at inference (decision gate)

flex_gemm exposes **only** submanifold conv (`sparse_submanifold_conv3d`, stride=1). If AniGen's inference path uses strided `SparseConv3d` or `SparseInverseConv3d` (down/upsampling), flex_gemm does not cover them and we need a decision (CPU spconv is unavailable on Mac; options: Metal strided conv, or restructure). This task finds out before we build the adapter.

**Files:**
- Create: `scripts/probe_sparse_ops.py`

- [ ] **Step 1: Write the probe (instruments the conv classes, runs a tiny forward)**

```python
# scripts/probe_sparse_ops.py
"""Census of sparse-conv ops actually constructed/invoked at inference.
Run on a CUDA box OR any box where the models can be instantiated on CPU meta
(we only need construction, not a full forward). Prints the distinct conv types.
"""
import anigen_mps  # noqa: F401  (sets env)
import collections
import anigen.modules.sparse.conv.conv_spconv as cs

seen = collections.Counter()
_orig_subm = cs.SparseConv3d.__init__
def _patched(self, in_c, out_c, kernel_size, stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    submanifold = (stride == 1 and padding is None)
    seen['SparseConv3d:submanifold' if submanifold else 'SparseConv3d:strided'] += 1
    return _orig_subm(self, in_c, out_c, kernel_size, stride, dilation, padding, bias, indice_key)
cs.SparseConv3d.__init__ = _patched

if hasattr(cs, 'SparseInverseConv3d'):
    _orig_inv = cs.SparseInverseConv3d.__init__
    def _patched_inv(self, *a, **k):
        seen['SparseInverseConv3d'] += 1
        return _orig_inv(self, *a, **k)
    cs.SparseInverseConv3d.__init__ = _patched_inv

# Instantiate the decoders used at inference for each variant.
from anigen.utils import model_utils  # noqa
# NOTE: fill in the actual decoder load calls for ss_dae + slat_dae here, e.g.:
#   model_utils.load_decoder("ckpts/anigen/slat_dae", "...", device="cpu")
#   model_utils.load_decoder("ckpts/anigen/ss_dae", "...", device="cpu")
print("Sparse conv op census:", dict(seen))
```

- [ ] **Step 2: Run the probe**

Run: `.venv-mac/bin/python scripts/probe_sparse_ops.py`
Expected: prints a census dict. **Decision gate:**
- If only `SparseConv3d:submanifold` appears → Task 8 (flex_gemm) fully covers it. Proceed.
- If `SparseConv3d:strided` or `SparseInverseConv3d` appear → STOP and record counts in the spec's Risks section. The flex_gemm submanifold API does not cover these; resolve before continuing (consult whether Pixal3D's `conv_flex_gemm.py` grew a strided path, or whether these convs are encoder-only/training-only and absent from the inference decoders).

- [ ] **Step 3: Commit the probe + findings**

```bash
git add scripts/probe_sparse_ops.py
git commit -m "test(mps): sparse-conv op census probe (flex_gemm coverage gate)"
```

---

### Task 8: `conv_flex_gemm.py` submanifold backend

**Files:**
- Create: `anigen/modules/sparse/conv/conv_flex_gemm.py`
- Modify: `anigen/modules/sparse/conv/__init__.py` (route `SPARSE_CONV_BACKEND=flex_gemm`)
- Test: `tests/mps/test_flex_gemm_conv.py`

Template: Pixal3D's `pixal3d/modules/sparse/conv/conv_flex_gemm.py`. Adapt to AniGen's `SparseTensor` (`x.feats`→`self.data.features`, `x.coords`→`self.data.indices`, `x.replace(feats)` swaps `_features`).

- [ ] **Step 1: Write the failing parity test (submanifold conv, CPU reference)**

```python
# tests/mps/test_flex_gemm_conv.py
import os, torch, pytest
os.environ.setdefault("SPARSE_BACKEND", "spconv")

def _make_sparse(feats, coords):
    from anigen.modules.sparse.basic import SparseTensor
    return SparseTensor(feats=feats, coords=coords)

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_flex_gemm_submanifold_runs_and_is_finite():
    from anigen.modules.sparse.conv.conv_flex_gemm import SparseConv3d as FGConv
    N, Ci, Co = 128, 16, 32
    coords = torch.zeros(N, 4, dtype=torch.int32)
    coords[:, 1:] = torch.randint(0, 8, (N, 3))
    feats = torch.randn(N, Ci)
    x = _make_sparse(feats, coords).to("mps")
    conv = FGConv(Ci, Co, kernel_size=3).to("mps")
    out = conv(x)
    assert out.feats.shape == (N, Co)
    assert torch.isfinite(out.feats).all()
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_flex_gemm_conv.py -v`
Expected: FAIL — module/class not found.

- [ ] **Step 3: Implement `conv_flex_gemm.py`**

```python
# anigen/modules/sparse/conv/conv_flex_gemm.py
"""flex_gemm (Metal) submanifold sparse-conv backend, drop-in for conv_spconv.

Matches conv_spconv.SparseConv3d's nn.Module interface but routes the kernel to
flex_gemm.ops.spconv.sparse_submanifold_conv3d. Only stride==1 + padding is None
(submanifold) is supported; anything else raises (see Task 7 decision gate).
"""
import torch
import torch.nn as nn
import flex_gemm
from flex_gemm.ops.spconv import sparse_submanifold_conv3d, set_algorithm, set_hashmap_ratio
from .. import SparseTensor

__all__ = ['SparseConv3d', 'SparseInverseConv3d']


class SparseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, padding=None, bias=True, indice_key=None):
        super().__init__()
        if not (stride == 1 and padding is None):
            raise NotImplementedError(
                "flex_gemm backend supports submanifold conv only (stride=1, no padding). "
                "See Task 7 decision gate.")
        ks = (kernel_size,) * 3 if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = ks
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else tuple(dilation)
        # flex_gemm weight layout: [Co, Kd, Kh, Kw, Ci]
        self.weight = nn.Parameter(torch.empty(out_channels, *ks, in_channels))
        nn.init.kaiming_uniform_(self.weight.view(out_channels, -1), a=5 ** 0.5)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: SparseTensor) -> SparseTensor:
        set_algorithm(flex_gemm.ops.spconv.Algorithm.IMPLICIT_GEMM)
        set_hashmap_ratio(2.0)
        Kd, Kh, Kw = self.kernel_size
        cache_key = f"flexgemm_subm_{Kw}x{Kh}x{Kd}_d{self.dilation}"
        neighbor_cache = x.get_spatial_cache(cache_key) if hasattr(x, "get_spatial_cache") else None
        shape = torch.Size([x.shape[0], self.in_channels, *x.data.spatial_shape])
        out_feats, cache_ = sparse_submanifold_conv3d(
            x.feats, x.coords, shape, self.weight, self.bias, neighbor_cache, self.dilation)
        if neighbor_cache is None and hasattr(x, "register_spatial_cache"):
            x.register_spatial_cache(cache_key, cache_)
        return x.replace(out_feats)


class SparseInverseConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "flex_gemm backend has no inverse/strided conv. See Task 7 decision gate.")
```
(If Task 7 shows `get_spatial_cache`/`register_spatial_cache` do not exist on AniGen's `SparseTensor`, pass `neighbor_cache=None` every call — correct but slower — and add the cache methods in a follow-up.)

- [ ] **Step 4: Route the backend in `conv/__init__.py`**

In `anigen/modules/sparse/conv/__init__.py`, after the existing `BACKEND` switch, add:
```python
import os
if os.environ.get("SPARSE_CONV_BACKEND") == "flex_gemm":
    from .conv_flex_gemm import *
elif BACKEND == 'torchsparse':
    from .conv_torchsparse import *
elif BACKEND == 'spconv':
    from .conv_spconv import *
```
(Place the flex_gemm check first so it wins on Mac; CUDA boxes leave `SPARSE_CONV_BACKEND` unset and fall through to spconv.)

- [ ] **Step 5: Run the parity test**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_flex_gemm_conv.py -v`
Expected: PASS (finite output, correct shape).

- [ ] **Step 6: Commit**

```bash
git add anigen/modules/sparse/conv/conv_flex_gemm.py anigen/modules/sparse/conv/__init__.py tests/mps/test_flex_gemm_conv.py
git commit -m "feat(mps): flex_gemm submanifold sparse-conv backend"
```

---

## Phase 4 — KNN & Rasterization Fallbacks

### Task 9: scipy cKDTree shim for `pytorch3d` KNN / ball_query

**Files:**
- Create: `anigen_mps/knn_cpu.py`
- Test: `tests/mps/test_knn_cpu.py`

Call sites (all `[1, P, C]` batched, K small): `cube2mesh_skeleton.py`, `skeleton/grouping.py:59,72,105`, `anigen_decoder.py`, `anigen_encoder.py`. We provide drop-ins matching `knn_points(...).idx` and `ball_query(...)` return contracts and monkeypatch `pytorch3d.ops` at bootstrap.

- [ ] **Step 1: Write the failing parity test (vs brute force)**

```python
# tests/mps/test_knn_cpu.py
import torch
from anigen_mps.knn_cpu import knn_points, ball_query

def _brute_knn(q, ref, K):
    d = torch.cdist(q, ref)                       # [1, P1, P2]
    dist, idx = d.topk(K, largest=False, dim=-1)
    return dist ** 2, idx

def test_knn_points_matches_brute_force():
    q = torch.randn(1, 50, 3); ref = torch.randn(1, 80, 3)
    out = knn_points(q, ref, K=3)
    bd, bi = _brute_knn(q, ref, 3)
    assert out.idx.shape == (1, 50, 3)
    assert torch.equal(out.idx, bi)
    assert torch.allclose(out.dists, bd, atol=1e-4)

def test_ball_query_radius_filtering():
    q = torch.randn(1, 40, 3); ref = torch.randn(1, 60, 3)
    out_dists, out_idx, _ = ball_query(q, ref, K=5, radius=0.5)
    assert out_idx.shape == (1, 40, 5)
    # entries beyond radius are marked -1 (pytorch3d convention)
    assert (out_idx == -1).any() or out_idx.min() >= 0
```

- [ ] **Step 2: Run to confirm failure**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_knn_cpu.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `knn_cpu.py`**

```python
# anigen_mps/knn_cpu.py
"""CPU scipy.spatial.cKDTree drop-ins for pytorch3d.ops.knn_points / ball_query.

Matches the pytorch3d return contracts used in AniGen:
  knn_points(p1, p2, K, ...) -> namedtuple(dists[B,P1,K] (squared), idx[B,P1,K], knn=None)
  ball_query(p1, p2, K, radius, ...) -> (dists[B,P1,K], idx[B,P1,K] (-1 padded), knn=None)
Inputs are [B, P, D] tensors on any device; compute happens on CPU then returns
on the input device.
"""
from collections import namedtuple
import torch
from scipy.spatial import cKDTree

_KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def knn_points(p1, p2, K=1, return_nn=False, norm=2, **kwargs):
    dev = p1.device
    B = p1.shape[0]
    dists_all, idx_all = [], []
    for b in range(B):
        a = p1[b].detach().cpu().numpy()
        ref = p2[b].detach().cpu().numpy()
        tree = cKDTree(ref)
        d, i = tree.query(a, k=K)
        if K == 1:
            d = d[:, None]; i = i[:, None]
        dists_all.append(torch.from_numpy(d).to(dev).float() ** 2)  # pytorch3d returns squared dists
        idx_all.append(torch.from_numpy(i).to(dev).long())
    dists = torch.stack(dists_all, 0)
    idx = torch.stack(idx_all, 0)
    knn = None
    if return_nn:
        knn = torch.gather(p2, 1, idx.unsqueeze(-1).expand(-1, -1, -1, p2.shape[-1]))
    return _KNN(dists, idx, knn)


def ball_query(p1, p2, K=500, radius=1.0, return_nn=False, **kwargs):
    dev = p1.device
    B = p1.shape[0]
    dists_all, idx_all = [], []
    for b in range(B):
        a = p1[b].detach().cpu().numpy()
        ref = p2[b].detach().cpu().numpy()
        tree = cKDTree(ref)
        d, i = tree.query(a, k=K, distance_upper_bound=radius)
        if K == 1:
            d = d[:, None]; i = i[:, None]
        # cKDTree pads out-of-range with index == n and dist == inf; pytorch3d uses -1 / 0.
        n = ref.shape[0]
        i_t = torch.from_numpy(i).to(dev).long()
        d_t = torch.from_numpy(d).to(dev).float()
        oob = i_t >= n
        i_t[oob] = -1
        d_t[oob] = 0.0
        idx_all.append(i_t)
        dists_all.append(d_t ** 2)
    dists = torch.stack(dists_all, 0)
    idx = torch.stack(idx_all, 0)
    return dists, idx, None
```

- [ ] **Step 4: Monkeypatch `pytorch3d.ops` in the bootstrap**

Append to `anigen_mps/__init__.py`:
```python
def install_knn_shim() -> None:
    """Replace pytorch3d.ops.knn_points/ball_query with CPU cKDTree drop-ins.

    pytorch3d's MPS backend for these ops is broken/absent. Install a fake
    pytorch3d.ops module BEFORE anigen imports it, so call sites bind to ours.
    """
    import sys, types
    from anigen_mps import knn_cpu
    try:
        import pytorch3d.ops as _ops  # real package present (CPU build) -> patch in place
        _ops.knn_points = knn_cpu.knn_points
        _ops.ball_query = knn_cpu.ball_query
    except Exception:
        pkg = types.ModuleType("pytorch3d"); ops = types.ModuleType("pytorch3d.ops")
        ops.knn_points = knn_cpu.knn_points
        ops.ball_query = knn_cpu.ball_query
        pkg.ops = ops
        sys.modules.setdefault("pytorch3d", pkg)
        sys.modules["pytorch3d.ops"] = ops


install_knn_shim()
```

- [ ] **Step 5: Run the parity tests**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_knn_cpu.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add anigen_mps/knn_cpu.py anigen_mps/__init__.py tests/mps/test_knn_cpu.py
git commit -m "feat(mps): scipy cKDTree shim for pytorch3d knn_points/ball_query"
```

---

### Task 10: Rasterization fallbacks (mtldiffrast + utils3d CPU)

**Files:**
- Modify: `anigen/renderers/mesh_renderer.py` (already device-gated in Task 4; confirm mtldiffrast API)
- Verify: `anigen/utils/postprocessing_utils.py` (utils3d CPU backend from Task 4)
- Test: `tests/mps/test_rasterizer_smoke.py`

- [ ] **Step 1: Confirm the exact mtldiffrast context + rasterize API**

Run:
```bash
.venv-mac/bin/python -c "import mtldiffrast.torch as m; print([n for n in dir(m) if 'Rast' in n or 'rasterize' in n])"
```
Expected: lists the context class + `rasterize` (+ `interpolate`, `antialias`, `texture` if present). Update the Task 4 `mesh_renderer.py` edit to use the exact names printed here.

- [ ] **Step 2: Write a rasterizer smoke test (single triangle)**

```python
# tests/mps/test_rasterizer_smoke.py
import torch, pytest

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mesh_renderer_constructs_on_mps():
    from anigen.renderers.mesh_renderer import MeshRenderer  # actual class name in file
    r = MeshRenderer(device="mps")
    assert r.glctx is not None
```
(Use the real renderer class name from `mesh_renderer.py`.)

- [ ] **Step 3: Run; fix the renderer edit until it constructs**

Run: `.venv-mac/bin/python -m pytest tests/mps/test_rasterizer_smoke.py -v`
Expected: PASS. If mtldiffrast's context constructor signature differs, correct the Task 4 edit in `mesh_renderer.py:52`.

- [ ] **Step 4: Confirm utils3d CPU rasterization produces a non-empty buffer**

Run a minimal call through `postprocessing_utils._rast_backend()` path with a tiny mesh (add a tmp script or extend the test) and assert the visibility/rasterize buffer is finite and non-empty. If the utils3d CPU backend cannot rasterize at all, record it as the top spec risk and gate texture-bake behind a `--no_texture` flag for Phase 1 (mesh + skeleton + skin still export).

- [ ] **Step 5: Commit**

```bash
git add anigen/renderers/mesh_renderer.py tests/mps/test_rasterizer_smoke.py
git commit -m "feat(mps): mtldiffrast renderer + utils3d CPU rasterization fallback"
```

---

## Phase 5 — Integration & Variants

### Task 11: `example_mps.py` entrypoint + first end-to-end run (default combo)

**Files:**
- Create: `example_mps.py`

- [ ] **Step 1: Write `example_mps.py`**

```python
# example_mps.py
"""Apple Silicon entrypoint. Bootstraps device shims, then runs example.main()."""
import anigen_mps  # noqa: F401  — configures env + installs knn shim at import
import sys

if __name__ == "__main__":
    if "--device" not in sys.argv:
        sys.argv += ["--device", "mps"]
    import example  # AniGen's stock CLI; device is now honored (Task 4)
    example.main() if hasattr(example, "main") else None
```
(If `example.py` runs at import time rather than via `main()`, wrap its body in `def main():` as a one-line Task 4 follow-up so it is callable.)

- [ ] **Step 2: Ensure checkpoints are present**

Run:
```bash
.venv-mac/bin/python -c "import anigen_mps; from anigen.utils.ckpt_utils import ensure_ckpts; ensure_ckpts('.')"
```
Expected: `ckpts/` populated (DINOv2, DSINE, VGG, ss_dae, slat_dae, ss_flow_duet, slat_flow_auto).

- [ ] **Step 3: First end-to-end inference (default ss_flow_duet + slat_flow_auto)**

Run:
```bash
.venv-mac/bin/python example_mps.py --image_path assets/cond_images/trex.png --output_dir results_mac
```
Expected: completes without CUDA/SDPA/device errors; produces `results_mac/trex/mesh.glb`, `skeleton.glb`, `processed_image.png`. Fix errors at their root (most will route back to Task 4/6/8/9/10 shims), not by disabling stages.

- [ ] **Step 4: Commit**

```bash
git add example_mps.py
git commit -m "feat(mps): example_mps.py entrypoint; first end-to-end run"
```

---

### Task 12: Structural validation of the output

**Files:**
- Create: `scripts/validate_glb.py`
- Test: `tests/mps/test_output_structural.py`

- [ ] **Step 1: Write `scripts/validate_glb.py`**

```python
# scripts/validate_glb.py
"""Structural sanity for a rigged AniGen GLB: watertight-ish mesh, sane skeleton,
skin weights ~sum to 1. Exits non-zero on failure."""
import sys, trimesh

def main(path):
    scene = trimesh.load(path)
    geoms = scene.geometry.values() if hasattr(scene, "geometry") else [scene]
    mesh = next(iter(geoms))
    assert len(mesh.vertices) > 100, f"too few vertices: {len(mesh.vertices)}"
    assert len(mesh.faces) > 100, f"too few faces: {len(mesh.faces)}"
    assert mesh.is_watertight or mesh.fill_holes(), "mesh not watertight and unfillable"
    print(f"OK: V={len(mesh.vertices)} F={len(mesh.faces)} watertight={mesh.is_watertight}")

if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: Run validation on the trex output**

Run: `.venv-mac/bin/python scripts/validate_glb.py results_mac/trex/mesh.glb`
Expected: prints `OK: V=... F=... watertight=...`.

- [ ] **Step 3: Write a skin-weights sanity test**

```python
# tests/mps/test_output_structural.py
import os, numpy as np, pytest, trimesh

GLB = "results_mac/trex/mesh.glb"

@pytest.mark.skipif(not os.path.exists(GLB), reason="run example_mps.py first")
def test_skin_weights_sum_to_one():
    # AniGen embeds skin weights in the glb; load and check rows ~sum to 1.
    scene = trimesh.load(GLB)
    # Skin weights live in mesh.visual / vertex attributes depending on exporter;
    # assert presence + per-vertex normalization with tolerance.
    # (Wire to the actual attribute key produced by AniGen's GLB export.)
    assert scene is not None
```
(Bind the assertion to the real skin-weight attribute key once observed in Task 11's output.)

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_glb.py tests/mps/test_output_structural.py
git commit -m "test(mps): structural validation of rigged GLB output"
```

---

### Task 13: Extend to all model variants

**Files:**
- Create: `scripts/run_all_variants_mac.sh`

- [ ] **Step 1: Write the variant matrix runner**

```bash
#!/usr/bin/env bash
set -euo pipefail
IMG="assets/cond_images/trex.png"
PY=".venv-mac/bin/python"

# SS flow variants with the default SLAT
for ss in ss_flow_duet ss_flow_solo ss_flow_epic; do
  $PY example_mps.py --image_path "$IMG" --output_dir "results_mac/$ss" \
      --ss_flow_path "ckpts/anigen/$ss" --slat_flow_path "ckpts/anigen/slat_flow_auto"
done

# Controllable SLAT across joint-density levels 0..4
for jd in 0 1 2 3 4; do
  $PY example_mps.py --image_path "$IMG" --output_dir "results_mac/slat_control_jd$jd" \
      --ss_flow_path "ckpts/anigen/ss_flow_duet" \
      --slat_flow_path "ckpts/anigen/slat_flow_control" --joints_density "$jd"
done
echo "all variants complete"
```

- [ ] **Step 2: Run the full matrix**

Run: `bash scripts/run_all_variants_mac.sh`
Expected: every variant produces a `mesh.glb` + `skeleton.glb` under its `results_mac/<variant>` dir, no errors.

- [ ] **Step 3: Validate every output**

Run:
```bash
for f in results_mac/*/*/mesh.glb; do .venv-mac/bin/python scripts/validate_glb.py "$f"; done
```
Expected: every GLB prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_all_variants_mac.sh
git commit -m "test(mps): all-variant inference matrix runner + validation"
```

---

## Phase 2 (separate plan) — Optimization

Out of scope for this plan. After Phase 1 is green, profile per-image latency and write a follow-up plan to port the proven CPU bottlenecks to Metal — likely (1) the cKDTree KNN paths and (2) the utils3d CPU rasterizer — only where measurement justifies it.

---

## Self-Review Notes

- **Spec coverage:** §2 success target → Tasks 11/13; §3 shim strategy → Tasks 3/4; §4 vendoring → Tasks 1/2; §5 component plan → Tasks 4–10; §6 MPS-SDPA-ban → Tasks 5/6; §7 phasing → Phase split; §8 validation → Tasks 12/13; §9 risks → Task 7 gate + Task 10 texture fallback. Covered.
- **Top risk made explicit:** flex_gemm covers submanifold conv only — Task 7 is a hard decision gate before building the adapter, and `SparseInverseConv3d` raises rather than silently producing garbage.
- **Names consistent:** `naive_varlen_attention`, `knn_points`/`ball_query`, `configure_mps_environment`/`resolve_device`/`install_knn_shim`, `SPARSE_CONV_BACKEND=flex_gemm` used identically across tasks.
- **Bind-to-reality flags:** exact public symbol names in `full_attn.py`, `mesh_renderer.py`, the GLB skin-weight attribute key, and mtldiffrast's context class are confirmed in their respective tasks before code depends on them.
