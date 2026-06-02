import os
import torch
import argparse
from PIL import Image

# Add current directory to path to allow imports
import sys
sys.path.append(os.getcwd())

from anigen.pipelines import AnigenImageTo3DPipeline
from anigen.utils.random_utils import set_random_seed
from anigen.utils.image_utils import _expand_image_inputs
from anigen.utils.ckpt_utils import ensure_ckpts


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image or a folder of images')
    parser.add_argument('--ss_flow_path', type=str, required=False, default='ckpts/anigen/ss_flow_duet', help='Path to SS Flow model directory')
    parser.add_argument('--slat_flow_path', type=str, required=False, default='ckpts/anigen/slat_flow_auto', help='Path to SLat Flow model directory (e.g. slat_flow_auto, slat_flow_control)')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cfg_scale_ss', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--cfg_scale', type=float, default=3.0, help='Classifier-free guidance scale')
    parser.add_argument('--deterministic', action='store_true', help='Enable mostly-deterministic torch behavior (may be slower)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA checkpoint if available')

    parser.add_argument(
        '--output_name',
        type=str,
        default=None,
        help='Optional subfolder name to save outputs under `--output_dir`. If not provided, the image filename stem is used.',
    )

    parser.add_argument('--simplify_ratio', type=float, default=0.95,
                        help='Fraction of faces to remove in mesh simplification (0-1). '
                             'Lower keeps more detail (e.g. 0.9 or 0.8) at the cost of a larger mesh. Default 0.95.')
    parser.add_argument('--texture_size', type=int, default=1024,
                        help='Baked texture resolution in pixels. Higher is sharper but slower/larger. '
                             'Set 0 to skip texturing. Default 1024.')
    parser.add_argument('--bake_mode', type=str, default=None, choices=['fast', 'opt'],
                        help="Texture baking: 'fast' = projective single-pass; 'opt' = differentiable "
                             "multi-view optimization (sharper, slower). Default: opt on CUDA, fast on MPS. "
                             "opt now works on Apple Silicon (no mip backward).")

    parser.add_argument('--no_smooth_skin_weights', action='store_true', help='Disable skin-weight smoothing')
    parser.add_argument('--smooth_skin_weights_iters', type=int, default=100, help='Number of smoothing iterations (default: 100)')
    parser.add_argument('--smooth_skin_weights_alpha', type=float, default=1.0, help='Smoothing alpha (default: 1.0)')

    parser.add_argument(
        '--no_filter_skin_weights',
        action='store_true',
        help='Use geodesic distribution to filter mesh skinning weights.',
    )

    parser.add_argument(
        '--joints_density', '--joint_density',
        type=int,
        default=1,
        help='Optional joint density level for Slat flow (from 0 to 4, higher means more joints)',
    )
    args = parser.parse_args()

    base_output_dir = args.output_dir
    input_image_paths, is_dir = _expand_image_inputs(args.image_path)
    if is_dir and len(input_image_paths) == 0:
        raise ValueError(f"No supported images found under directory: {args.image_path}")

    # For directory input, group outputs under a batch folder.
    # For single-image input, keep original behavior: output under `<base_output_dir>/<output_name|image_stem>`.
    batch_folder_name = None
    if is_dir:
        batch_folder_name = args.output_name if (args.output_name is not None and str(args.output_name).strip() != '') else os.path.basename(os.path.normpath(args.image_path))
    set_random_seed(args.seed, deterministic=args.deterministic)

    ensure_ckpts()

    print("Loading models...")
    pipeline = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path=args.ss_flow_path,
        slat_flow_path=args.slat_flow_path,
        device=args.device,
        use_ema=args.use_ema
    )
    pipeline.to(args.device)

    for idx, cur_image_path in enumerate(input_image_paths):
        # Per-image output directory.
        image_stem = os.path.splitext(os.path.basename(cur_image_path))[0]
        if is_dir:
            args.output_dir = os.path.join(base_output_dir, str(batch_folder_name), image_stem)
        else:
            # Allow user to override the saved folder name via --output_name. Fallback to image stem.
            folder_name = args.output_name if (args.output_name is not None and str(args.output_name).strip() != '') else image_stem
            args.output_dir = os.path.join(base_output_dir, folder_name)
        os.makedirs(args.output_dir, exist_ok=True)

        # Keep args.image_path aligned for any downstream logging/debug usage.
        args.image_path = cur_image_path
        print(f"Processing image {idx + 1}/{len(input_image_paths)}: {cur_image_path}")
        image = Image.open(cur_image_path)
        
        # Run pipeline
        output_glb_path = os.path.join(args.output_dir, 'mesh.glb')
        outputs = pipeline.run(
            image,
            seed=args.seed,
            cfg_scale_ss=args.cfg_scale_ss,
            cfg_scale_slat=args.cfg_scale,
            joints_density=args.joints_density,
            simplify_ratio=args.simplify_ratio,
            texture_size=args.texture_size,
            bake_mode=args.bake_mode,
            no_smooth_skin_weights=args.no_smooth_skin_weights,
            no_filter_skin_weights=args.no_filter_skin_weights,
            smooth_skin_weights_iters=args.smooth_skin_weights_iters,
            smooth_skin_weights_alpha=args.smooth_skin_weights_alpha,
            output_glb=output_glb_path
        )
        
        # Save processed images
        outputs['processed_image'].save(os.path.join(args.output_dir, 'processed_image.png'))

    
if __name__ == '__main__':
    main()
