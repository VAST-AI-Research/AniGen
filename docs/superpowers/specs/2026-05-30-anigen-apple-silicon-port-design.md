# AniGen → Apple Silicon Port — Design Spec

**Date:** 2026-05-30
**Status:** Approved for planning
**Scope:** Inference-only port of AniGen to Apple Silicon (MPS/Metal), reusing the
Metal toolkit developed for the Pixal3D Apple Silicon port.

---

## 1. Background & Key Finding

AniGen generates a fully rigged 3D asset (mesh + articulated skeleton + skinning
weights) from a single image, via a two-stage flow-matching pipeline over *S^3
Fields* (Shape, Skeleton, Skin) in a shared structured-latent (SLAT) space.

**Critical architectural fact (drove the project scope):** AniGen's skeleton and
skinning weights are **co-generated jointly with geometry inside AniGen's own SLAT
latent space** — same sparse voxel coordinates, cross-attention between
vertex-skin and joint features, trained end-to-end. They are **not** a
post-process bolted onto a finished mesh. Evidence:
`anigen/models/structured_latent_vae/anigen_decoder.py` exposes four decoder heads
(geometry, skeleton, joint-skin, vertex-skin) off the *same* latent
(`x_skin = x0.feats[:, self.latent_channels:]`, ~line 635), feeding a
`TreeTransformerSkinDecoder`.

**Consequence:** The original idea — swap AniGen's generator for Pixal3D's
(TRELLIS.2/Direct3D-S2) geometry backbone and bolt AniGen's rig on top — is **not
viable** without retraining the rig decoder against Pixal3D's feature space (a
research effort, explicitly out of scope). Pixal3D's SLAT encodes shape + texture
only; it is not articulation-aware.

**What we reuse instead:** the Apple Silicon *porting infrastructure* built for
Pixal3D. AniGen has the **same CUDA-only dependency stack** (spconv, nvdiffrast,
flash-attn), so we run AniGen's real, unmodified pipeline on the Metal substrate
already proven in Pixal3D.

---

## 2. Goal & Success Criteria

Run AniGen's **unmodified inference pipeline** (its own generator + entangled
skeleton/skin heads) natively on Apple Silicon.

**Done =**
- `python example.py --image_path assets/cond_images/trex.png` produces valid
  `mesh.glb` + `skeleton.glb` + `processed_image.png` on MPS.
- Works across **all model variants**: `ss_flow_duet`, `ss_flow_solo`,
  `ss_flow_epic`; `slat_flow_auto`, `slat_flow_control` (incl. joint-density
  levels 0–4).
- Output is structurally sane: watertight mesh, plausible joint count/hierarchy,
  skin weights summing to ~1.

**Out of scope:** training, the CUBVH extension, the Pixal3D-as-generator hybrid,
`natten-mps`, `o_voxel`, and the Gradio `app.py` demo.

---

## 3. Strategy: Shim Layer Over a Clean Clone

Mirror the Pixal3D pattern: a thin **`anigen_mps` bootstrap** plus a CLI entrypoint
(`example_mps.py`) that applies all device shims **before** importing the pipeline,
keeping the AniGen clone close to upstream and re-syncable.

The bootstrap:
- Sets `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Forces the attention backend to `naive` (see §6 — MPS SDPA is banned).
- Applies fp32 upcasting for DiTs/VAEs (the Pixal3D `PIXAL3D_FP32_MODELS` playbook)
  to avoid bf16/fp16 precision cliffs on MPS.
- Monkeypatches CUDA-only imports at their boundaries (spconv→flex_gemm,
  nvdiffrast→mtldiffrast, flash_attn→naive, pytorch3d KNN→CPU shim, utils3d
  rasterizer→CPU fallback).

Make small **targeted in-tree edits** only where monkeypatching is fragile (e.g.
the sparse-conv backend selector). Default to import-time shims so upstream stays
diffable.

---

## 4. Dependency Reuse & Vendoring

AniGen gets its **own Python 3.10 venv** on Mac (separate from Pixal3D's).

**Vendored Metal packages** — vendored into `AniGen/extern/`, sourced from the
user's personal forks (which carry fixes not yet upstreamed to Pedro Naugusto's
originals). Each fork is its own GitHub repo at
`github.com/pawel-mazurkiewicz/<pkg>`:

| Vendored pkg | Import name | Replaces | AniGen use |
|---|---|---|---|
| `mtlgemm` | `flex_gemm` | spconv sparse conv | sparse VAE/flow ops |
| `mtldiffrast` | `mtldiffrast` | nvdiffrast | renderer + texture bake |
| `mtlmesh` | `cumesh` | cumesh mesh ops | mesh post-processing |
| `mtlbvh` | `mtlbvh` | CUDA BVH | used internally by mtlmesh |

Install editable: `pip install -e extern/<pkg> --no-build-isolation` for each.
**Not vendored:** `natten-mps` (AniGen uses flash-attn/xformers, not neighborhood
attention) and `o_voxel` (unused in AniGen inference).

**Provenance note** (record in `extern/README.md`): these are forks of Pedro
Naugusto's Metal libraries with our fixes; track upstream and drop the fork pin
if/when our fixes merge.

**Prerequisites:**
- Xcode Metal toolchain (`xcrun`) — needed to compile `.metal` → `.metallib`.
- **`torch==2.12.0`** on Darwin. `flex_gemm` requires `torch>=2.11` for
  `at::mps::dispatch_sync_with_rethrow`. This conflicts with AniGen upstream's
  stated torch 2.4–2.5 target; we pin to 2.12 (matching Pixal3D's verified env)
  and **validate AniGen's code against it** as an explicit risk (see §8).
- A Mac requirements file that **drops** the CUDA-only wheels (`spconv-cuXX`,
  `nvdiffrast`, `flash-attn`, the CUDA `pytorch3d` build).

---

## 5. Component Port Plan

Derived from the inference-path CUDA op-gap analysis.

| Component | Call site(s) | Action | Effort |
|---|---|---|---|
| spconv `SparseConv3d` / `SparseInverseConv3d` | `anigen/modules/sparse/conv/conv_spconv.py` | Route AniGen `SparseTensor` backend → `flex_gemm` | Low |
| nvdiffrast | `anigen/renderers/mesh_renderer.py`, `anigen/utils/postprocessing_utils.py` | Swap → `mtldiffrast` | Low |
| Attention (dense + sparse) | `anigen/modules/attention/full_attn.py:116-120`, `anigen/modules/sparse/attention/full_attn.py` | Force `naive` backend, hardened to Pixal3D fp32 chunked matmul→softmax→matmul. **MPS SDPA banned.** | Medium |
| FlexiCubes mesh extraction | `anigen/representations/mesh/flexicubes/flexicubes.py` | None — pure PyTorch, runs on MPS | None |
| pytorch3d `knn_points` / `ball_query` | `anigen/representations/mesh/cube2mesh_skeleton.py`, `anigen/representations/skeleton/grouping.py:59,72`, `anigen/models/structured_latent_vae/anigen_decoder.py` | CPU shim via `scipy.spatial.cKDTree` | Medium |
| `utils3d.torch` CUDA rasterizer | `anigen/utils/postprocessing_utils.py:78,480` | CPU fallback path (texture-bake visibility) | Medium-High |
| Mesh post (trimesh/pymeshfix/xatlas/igraph/pyvista) | `anigen/utils/postprocessing_utils.py` | None — CPU libs already | None |
| DINOv2 / DSINE | `anigen/pipelines/anigen_image_to_3d.py:66,71` | None — plain PyTorch on MPS | None |

---

## 6. Attention: MPS SDPA Is Banned

**First-class constraint, not a footnote.** Pixal3D proved that MPS's fused
`scaled_dot_product_attention` is bugged: above ~18–20k tokens it returns
catastrophically wrong output (mae 0.05–0.10 — the "SDPA cliff"). The fix was a
custom **naive fp32 matmul→softmax→matmul kernel on the MPS device**
(`SPARSE_ATTN_BACKEND=naive` in Pixal3D).

AniGen's HR sparse-attention stages exceed that token count, so the same trap
applies. We therefore:
- Force AniGen's existing `naive` attention backend everywhere at inference.
- Harden/verify that `naive` path matches Pixal3D's chunked-fp32 implementation
  (and does **not** internally fall through to SDPA).
- Treat MPS SDPA as untrusted for the entire pipeline.

(Phase 2 may selectively re-enable SDPA only for stages provably below the cliff
and validated faithful — not assumed.)

---

## 7. Phasing (Correctness-First)

**Phase 1 — Correctness, all variants.** Wire everything with CPU fallbacks on the
two hard gaps (KNN, utils3d rasterizer) + naive attention + fp32 upcasts. Get every
model variant emitting a valid rigged GLB. No optimization.

**Phase 2 — Targeted optimization.** Profile; port to Metal **only** the CPU paths
that measurement proves are real bottlenecks (likely KNN first, rasterizer second).
This mirrors how the Pixal3D port itself evolved.

---

## 8. Numerical Validation

- Per stage, capture intermediate tensors (sparse-structure coords, SLAT feats,
  joints, skin weights); sanity-check shapes and value ranges.
- Where a known-good CUDA reference output exists, compare mesh/skeleton/skin
  within tolerance. Otherwise validate structurally (watertight mesh, sane joint
  hierarchy, skin weights ≈ 1) + visual inspection of the GLB.
- The naive-attention path gets a dedicated high-token correctness check — that is
  exactly where MPS SDPA fails.

---

## 9. Top Risks & Open Questions

1. **`utils3d` CPU rasterizer** (riskiest): confirm a CPU path exists and produces
   usable visibility masks; otherwise texture bake degrades. Verify during
   planning.
2. **torch 2.12 vs AniGen upstream (2.4–2.5):** AniGen code may use APIs that
   shifted across torch versions. Validate the pipeline imports and runs under
   2.12 early.
3. **KNN parity:** `cKDTree` vs `pytorch3d.knn_points` ordering/tie-breaking could
   subtly shift skeleton grouping — validate joint output.
4. **Skin weight quality on CPU paths:** the entangled rig is the whole point;
   watch skin weights closely through the fallbacks.

---

## 10. Out of Scope

Training, CUBVH, the Pixal3D-as-generator hybrid, `natten-mps`, `o_voxel`,
Gradio `app.py`.
