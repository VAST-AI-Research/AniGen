# Repository Guidelines

`AGENTS.md` is the shared, canonical instruction file for coding agents in this repository.
Keep common project context, commands, conventions, and gotchas here.
If `CLAUDE.md` or `CODEX.md` exists, they should stay lightweight and only contain tool-specific addenda to avoid duplicated guidance drifting out of sync.

## Project Overview

**AniGen** is a unified framework that generates animate-ready 3D assets from a single image. It produces coherent meshes with articulated skeletons and skinning weights, ready for immediate use in 3D pipelines and motion-capture-driven animation. The core representation is *S^3 Fields* (Shape, Skeleton, Skin) defined over a shared spatial domain. Built on a two-stage flow-matching pipeline, AniGen first synthesizes a sparse structural scaffold (skeleton + coarse shape) and then generates dense geometry and articulation in a structured latent space.

- **Paper**: ACM SIGGRAPH 2026 ([arXiv:2604.08746](https://arxiv.org/pdf/2604.08746))
- **License**: MIT (source code). `extensions/CUBVH/` is non-commercial / research only (NVIDIA instant-ngp derived). See `THIRD_PARTY_LICENSES.md`.

## Project Structure & Module Organization

```
.
├── anigen/                        # Main Python package
│   ├── models/                    # Model architectures (VAE, flow matching)
│   │   ├── anigen_sparse_structure_vae.py
│   │   ├── anigen_sparse_structure_flow.py
│   │   ├── anigen_structured_latent_flow.py
│   │   └── structured_latent_vae/
│   ├── modules/                   # Neural network building blocks
│   │   ├── sparse/                # Sparse 3D convolution & attention (spconv)
│   │   ├── attention/             # Standard attention mechanisms
│   │   └── transformer/           # Transformer blocks with modulation
│   ├── pipelines/                 # Inference pipelines
│   │   ├── anigen_image_to_3d.py  # Main end-to-end inference pipeline
│   │   └── samplers/              # Flow-matching samplers
│   ├── datasets/                  # Training dataset loaders
│   ├── trainers/                  # Training orchestration
│   │   ├── vae/                   # VAE trainers (skin AE, SS VAE, SLAT VAE)
│   │   └── flow_matching/         # Flow matching trainers with CFG
│   ├── representations/           # 3D data structures
│   │   ├── mesh/                  # FlexiCubes implicit surface extraction
│   │   ├── skeleton/              # Skeleton with bone grouping
│   │   ├── gaussian/              # Gaussian splatting
│   │   └── octree/                # Spatial indexing
│   ├── renderers/                 # Rendering utilities
│   └── utils/                     # Utilities (checkpoint, mesh, image, skin, etc.)
├── configs/                       # Training configuration files (JSON)
│   ├── anigen_skin_ae.json        # Stage 1: Skin AutoEncoder
│   ├── ss_dae.json                # Stage 2: Sparse Structure DAE
│   ├── slat_dae.json              # Stage 3: Structured Latent DAE
│   ├── ss_flow_duet.json          # Stage 4: SS Flow Matching
│   └── slat_flow_auto.json        # Stage 5: SLAT Flow Matching
├── extensions/
│   └── CUBVH/                     # CUDA BVH extension (training only)
├── ckpts/                         # Pretrained model weights
│   ├── anigen/                    # AniGen models (ss_flow_*, slat_flow_*, *_dae)
│   ├── dinov2/                    # DINOv2 vision encoder
│   ├── dsine/                     # Surface normal estimator
│   └── vgg/                       # VGG backbone
├── assets/                        # Demo images and example outputs
├── third_parties/                 # External dependencies (dsine)
├── example.py                     # CLI inference entrypoint
├── app.py                         # Gradio web demo
├── train.py                       # Training entrypoint
├── setup.sh                       # Smart installation script
└── requirements.txt               # Python dependencies (non-CUDA)
```

## Environment Setup

### Prerequisites
- **OS**: Linux only.
- **GPU**: NVIDIA GPU with >= 18 GB VRAM. Tested on A800, RTX 3090.
- **CUDA Toolkit**: 11.8 or 12.2 (needed to compile extensions).
- **Python**: 3.10+.

### Installation Commands

Full install (creates a new virtual environment):
```bash
source ./setup.sh --new-env --all
```

With Tsinghua PyPI mirror (for users in China):
```bash
source ./setup.sh --new-env --all --tsinghua
```

Into an existing environment with PyTorch already installed:
```bash
source ./setup.sh --basic
```

Add Gradio demo dependencies:
```bash
source ./setup.sh --demo
```

Add training dependencies (builds CUBVH extension):
```bash
source ./setup.sh --train
```

**Important**: `setup.sh` must be **sourced** (`. ./setup.sh` or `source ./setup.sh`), not executed directly (`bash setup.sh` will fail to activate the environment).

The script auto-detects CUDA version and installs matching wheels for PyTorch, spconv, pytorch3d, and nvdiffrast. It prefers `uv` for fast installs and falls back to `pip`.

## Pretrained Model Weights

Weights are stored under `ckpts/`. The `ensure_ckpts()` function in `anigen/utils/ckpt_utils.py` auto-downloads missing weights from Hugging Face on first run.

Required weight directories:
- `ckpts/dinov2/` — DINOv2 vision encoder
- `ckpts/dsine/` — DSINE surface normal prediction
- `ckpts/vgg/` — VGG feature backbone
- `ckpts/anigen/ss_dae/` — Sparse Structure encoder/decoder
- `ckpts/anigen/slat_dae/` — Structured Latent encoder/decoder
- `ckpts/anigen/ss_flow_solo/` — SS Flow model (default)
- `ckpts/anigen/slat_flow_auto/` — SLAT Flow model (default)

**Default model combination**: `ss_flow_solo` + `slat_flow_auto` (defaults in `example.py`; accurate geometry, frozen geo weights).

Other variants:
- `ss_flow_duet` — more detailed skeleton (full-FT geo)
- `ss_flow_epic` — balanced geometry & skeleton (LoRA-FT geo)
- `slat_flow_control` — controllable joint density (levels 0–4)

## Inference Commands

### CLI (primary recommended test)
```bash
# Single image
python example.py --image_path assets/cond_images/trex.png

# Directory of images (batch)
python example.py --image_path assets/cond_images/

# With custom model variants
python example.py --image_path assets/cond_images/trex.png \
    --ss_flow_path ckpts/anigen/ss_flow_solo \
    --slat_flow_path ckpts/anigen/slat_flow_control \
    --joints_density 2
```

Key arguments:
- `--seed` (default: 42) — random seed
- `--cfg_scale_ss` (default: 7.5) — classifier-free guidance scale for SS stage
- `--cfg_scale` (default: 3.0) — guidance scale for SLAT stage
- `--joints_density` (default: 1) — joint density level 0–4 (only for `slat_flow_control`)
- `--output_dir` (default: `results/`)
- `--use_ema` — use EMA checkpoint if available

Output files per image: `mesh.glb` (rigged mesh), `skeleton.glb` (skeleton viz), `processed_image.png`.

### Web Demo (primary smoke test)
```bash
python app.py
```
Opens a Gradio interface. Requires demo dependencies (`source ./setup.sh --demo`).

## Training Commands

Training has 5 sequential stages. Later stages depend on earlier ones.

```bash
# Stage 1: Skin AutoEncoder
python train.py --config configs/anigen_skin_ae.json --output_dir outputs/anigen_skin_ae

# Stage 2: Sparse Structure DAE
python train.py --config configs/ss_dae.json --output_dir outputs/ss_dae

# Stage 3: Structured Latent DAE
python train.py --config configs/slat_dae.json --output_dir outputs/slat_dae

# Stage 4: SS Flow Matching (image-conditioned generation)
python train.py --config configs/ss_flow_duet.json --output_dir outputs/ss_flow_duet

# Stage 5: SLAT Flow Matching (image-conditioned generation)
python train.py --config configs/slat_flow_auto.json --output_dir outputs/slat_flow_auto
```

### Multi-Node / Multi-GPU
```bash
python train.py --config configs/<config>.json --output_dir outputs/<output> \
    --num_nodes XX --node_rank XX --master_addr XX --master_port XX
```

### Resume / Restart
- Training automatically resumes from the latest checkpoint in `--output_dir`.
- To start fresh: `--ckpt none`.
- CUBVH extension (`extensions/CUBVH/`) is required for training but **not** for inference.

### Training Data
Sample data: [VAST-AI/AniGen_sample_data](https://huggingface.co/datasets/VAST-AI/AniGen_sample_data) on Hugging Face.

## Key Dependencies & Gotchas

| Dependency | Notes |
|---|---|
| PyTorch 2.4–2.5 | Auto-selected by `setup.sh` based on Python version |
| spconv | CUDA-version-matched (`spconv-cu118` or `spconv-cu121`) |
| pytorch3d | Pre-built wheel or source build; requires `--no-build-isolation` |
| nvdiffrast | Differentiable rasterization; compiled extension |
| CUBVH | Local CUDA extension in `extensions/CUBVH/`; training only |
| flash-attn | Optional, improves speed; falls back to SDPA if absent |
| xformers | Optional, improves memory efficiency |
| DSINE | Loaded at runtime via `torch.hub`; no manual install needed |
| numpy | Must be `< 2` (`numpy>=1.24,<2` in requirements.txt) |
| transformers | HuggingFace Transformers for DINOv2 encoder |

Common issues:
- CUDA compilation of pytorch3d/nvdiffrast/CUBVH can take 10–30 minutes. Be patient.
- If `spconv` import fails, ensure the CUDA version matches your PyTorch CUDA version.
- `torch.hub` requires internet access on first run to download DSINE weights.

## Coding Style & Conventions

- **Python**: 3.10+, 4-space indentation.
- **Naming**: `snake_case` for functions/variables/files, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- **Config-driven training**: All hyperparameters live in JSON config files under `configs/`. Model architecture, dataset, trainer, optimizer, loss schedules, and sampling strategies are all specified in config.
- **Mixin pattern**: Model components use mixin classes (e.g., classifier-free guidance mixin in flow matching trainers).
- **Imports**: Use `from anigen import models, datasets, trainers` pattern. Models/datasets/trainers are registered by name and instantiated via `getattr()`.

## Testing

No dedicated test suite (research codebase). Verify correctness with:

1. **Primary smoke test** — launch the Gradio web demo:
   ```bash
   python app.py
   ```
   This loads all models and verifies the full pipeline is operational.

2. **CLI smoke test** — run inference on a sample image:
   ```bash
   python example.py --image_path assets/cond_images/trex.png
   ```
   Check that `results/trex/mesh.glb` is generated successfully.

## Commit & Pull Request Guidelines

- Short, descriptive subject line.
- Reference issue/PR numbers when applicable.
- Keep commits focused by concern (feature, refactor, fix).
- PRs should include a clear problem/solution summary and test evidence.
