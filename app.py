import gradio as gr
from gradio_litmodel3d import LitModel3D

import os
import uuid
import shutil
import gc
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image

import sys
sys.path.append(os.getcwd())

from anigen.pipelines import AnigenImageTo3DPipeline
from anigen.utils.random_utils import set_random_seed
from anigen.utils.ckpt_utils import ensure_ckpts

MAX_SEED = 100
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

SS_MODEL_CHOICES = ["ss_flow_duet", "ss_flow_solo", "ss_flow_epic"]
SLAT_MODEL_CHOICES = ["slat_flow_auto", "slat_flow_control"]
DEFAULT_SS_MODEL = "ss_flow_duet"
DEFAULT_SLAT_MODEL = "slat_flow_auto"

current_ss_model_name = DEFAULT_SS_MODEL
current_slat_model_name = DEFAULT_SLAT_MODEL


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image, _ = pipeline.preprocess_image(image)
    return processed_image


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def on_slat_model_change(slat_model_name: str):
    is_control = (slat_model_name == "slat_flow_control")
    return (
        gr.update(interactive=is_control),
        gr.update(visible=not is_control),
    )


def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_model_name: str,
    slat_model_name: str,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    joints_density: int,
    texture_size: int,
    req: gr.Request = None,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[str, str, Image.Image]:
    """
    Convert an image to a 3D model.
    """
    global current_ss_model_name, current_slat_model_name

    no_smooth_skin_weights = False
    no_filter_skin_weights = False
    smooth_skin_weights_iters = 100
    smooth_skin_weights_alpha = 1.0

    if ss_model_name != current_ss_model_name:
        progress(0, desc=f"Loading SS model: {ss_model_name}...")
        pipeline.load_ss_flow_model(f'ckpts/anigen/{ss_model_name}', device='cuda', use_ema=False)
        current_ss_model_name = ss_model_name

    if slat_model_name != current_slat_model_name:
        progress(0, desc=f"Loading SLAT model: {slat_model_name}...")
        pipeline.load_slat_flow_model(f'ckpts/anigen/{slat_model_name}', device='cuda', use_ema=False)
        current_slat_model_name = slat_model_name

    session_id = req.session_hash if req else uuid.uuid4().hex
    user_dir = os.path.join(TMP_DIR, session_id)
    os.makedirs(user_dir, exist_ok=True)

    run_id = uuid.uuid4().hex
    run_dir = os.path.join(user_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    output_glb_path = os.path.join(run_dir, 'mesh.glb')
    skeleton_glb_path = os.path.join(run_dir, 'skeleton.glb')

    progress(0, desc="Preprocessing image...")

    def ss_progress_callback(step, total):
        frac = (step + 1) / total
        progress(frac * 0.45, desc=f"SS Sampling: {step + 1}/{total}")

    def slat_progress_callback(step, total):
        frac = (step + 1) / total
        progress(0.45 + frac * 0.45, desc=f"SLat Sampling: {step + 1}/{total}")

    def postprocess_progress_callback(frac, desc):
        progress(0.90 + frac * 0.10, desc=desc)

    outputs = pipeline.run(
        image,
        seed=seed,
        cfg_scale_ss=ss_guidance_strength,
        cfg_scale_slat=slat_guidance_strength,
        ss_steps=ss_sampling_steps,
        slat_steps=slat_sampling_steps,
        joints_density=joints_density,
        no_smooth_skin_weights=no_smooth_skin_weights,
        no_filter_skin_weights=no_filter_skin_weights,
        smooth_skin_weights_iters=smooth_skin_weights_iters,
        smooth_skin_weights_alpha=smooth_skin_weights_alpha,
        texture_size=int(texture_size),
        output_glb=output_glb_path,
        ss_progress_callback=ss_progress_callback,
        slat_progress_callback=slat_progress_callback,
        postprocess_progress_callback=postprocess_progress_callback,
    )

    processed_image = outputs['processed_image']
    del outputs
    gc.collect()
    
    torch.cuda.empty_cache()
    
    if not os.path.exists(skeleton_glb_path):
        skeleton_glb_path = None
        
    return output_glb_path, skeleton_glb_path, processed_image


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [AniGen]
    * Upload an image and click "Generate" to create a 3D asset with skeleton.
    """)
    
    gr.HTML("""
<style>
@keyframes gentle-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.35; }
}
</style>
<div style="text-align:left; color:#888; font-size:1em; line-height:1.6; margin-bottom:-8px;">
    <span style="animation: gentle-pulse 3s ease-in-out infinite; display:inline-block;">&#128161; <b>Tip</b></span>&ensp;
    Not satisfied with the geometry or skeleton?
    Try switching the SS Model to <code>ss_flow_solo</code> or <code>ss_flow_duet</code> in Generation Settings.
</div>
""")

    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)

            with gr.Accordion(label="Generation Settings", open=True):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)

                gr.Markdown("**Model Selection**")
                with gr.Row():
                    ss_model_dropdown = gr.Dropdown(
                        choices=SS_MODEL_CHOICES,
                        value=DEFAULT_SS_MODEL,
                        label="SS Model (Sparse Structure)",
                    )
                    slat_model_dropdown = gr.Dropdown(
                        choices=SLAT_MODEL_CHOICES,
                        value=DEFAULT_SLAT_MODEL,
                        label="SLAT Model (Structured Latent)",
                    )

                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 15.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                
                gr.Markdown("Skeleton & Skinning Settings")
                joints_density = gr.Slider(0, 4, label="Joints Density", value=1, step=1, interactive=False)
                density_hint = gr.Markdown(
                    "*Switch `SLAT Model` to `slat_flow_control` to enable joint density control.*",
                    visible=True,
                )
                no_smooth_skin_weights = False  # gr.Checkbox(label="Disable Skin Weight Smoothing", value=False)
                no_filter_skin_weights = False  # gr.Checkbox(label="Disable Skin Weight Filtering", value=False)
                smooth_skin_weights_iters = 100  # gr.Slider(0, 200, label="Smoothing Iterations", value=100, step=1)
                smooth_skin_weights_alpha = 1.0  # gr.Slider(0.0, 1.0, label="Smoothing Alpha", value=1.0, step=0.1)

                gr.Markdown("Texture Settings")
                texture_size = gr.Slider(256, 2048, label="Texture Resolution", value=1024, step=256)

            generate_btn = gr.Button("Generate")

        with gr.Column():
            mesh_output = gr.Model3D(label="Generated Mesh", height=300, interactive=False)
            download_mesh = gr.DownloadButton(label="Download Mesh GLB", interactive=False)
            skeleton_output = LitModel3D(label="Generated Skeleton", exposure=5.0, height=300, interactive=False)
            download_skeleton = gr.DownloadButton(label="Download Skeleton GLB", interactive=False)
            processed_image_output = gr.Image(label="Processed Image", type="pil", height=300)
    
    # Example images at the bottom of the page
    with gr.Row() as single_image_example:
        examples = gr.Examples(
            examples=[
                f'assets/cond_images/{image}'
                for image in os.listdir("assets/cond_images")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    
    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    slat_model_dropdown.change(
        on_slat_model_change,
        inputs=[slat_model_dropdown],
        outputs=[joints_density, density_hint],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, ss_model_dropdown, slat_model_dropdown,
            ss_guidance_strength, ss_sampling_steps, 
            slat_guidance_strength, slat_sampling_steps, joints_density,
            texture_size,
        ],
        outputs=[mesh_output, skeleton_output, processed_image_output],
    ).then(
        lambda mesh_path, skel_path: tuple([
            gr.DownloadButton(value=mesh_path, interactive=True) if mesh_path else gr.DownloadButton(interactive=False),
            gr.DownloadButton(value=skel_path, interactive=True) if skel_path else gr.DownloadButton(interactive=False)
        ]),
        inputs=[mesh_output, skeleton_output],
        outputs=[download_mesh, download_skeleton],
    )

# Launch the Gradio app
if __name__ == "__main__":
    ensure_ckpts()
    pipeline = AnigenImageTo3DPipeline.from_pretrained(
        ss_flow_path=f'ckpts/anigen/{DEFAULT_SS_MODEL}',
        slat_flow_path=f'ckpts/anigen/{DEFAULT_SLAT_MODEL}',
        device='cuda',
        use_ema=False
    )
    pipeline.cuda()
    enable_share = os.environ.get("ANIGEN_GRADIO_SHARE", "0") == "1"
    demo.launch(server_name="0.0.0.0", share=enable_share)
