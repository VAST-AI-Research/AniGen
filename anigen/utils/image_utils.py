import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import rembg

_SUPPORTED_IMAGE_EXTS = {
    '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'
}

def _expand_image_inputs(image_path: str) -> tuple[list[str], bool]:
    """Return (image_paths, is_directory).

    If image_path is a directory, returns all supported images under it (non-recursive),
    sorted by filename. Otherwise returns [image_path].
    """
    if image_path is None:
        raise ValueError('image_path is None')

    image_path = str(image_path)
    if os.path.isdir(image_path):
        entries = []
        for name in sorted(os.listdir(image_path)):
            full = os.path.join(image_path, name)
            if not os.path.isfile(full):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in _SUPPORTED_IMAGE_EXTS:
                entries.append(full)
        return entries, True

    return [image_path], False

def load_dsine(device='cuda'):
    ckpt_path = 'ckpts/dsine/dsine.pt'
    hub_kwargs = {}
    if os.path.exists(ckpt_path):
        hub_kwargs['local_file_path'] = ckpt_path

    # Prefer cached hub repo to avoid network access
    hub_dir = torch.hub.get_dir()
    cached_repo = os.path.join(hub_dir, 'hugoycj_DSINE-hub_main')

    if os.path.isdir(cached_repo):
        print(f"Loading DSINE from cached hub repo: {cached_repo}")
        predictor = torch.hub.load(cached_repo, "DSINE", source='local',
                                   trust_repo=True, **hub_kwargs)
    else:
        print("Loading DSINE via torch.hub (requires network)...")
        predictor = torch.hub.load("hugoycj/DSINE-hub:main", "DSINE",
                                   trust_repo=True, **hub_kwargs)
    return predictor

def estimate_normal(image, predictor, device='cuda'):
    # image: PIL Image RGB
    # predictor: DSINE Predictor from torch.hub (handles padding, intrinsics, inference)
    with torch.no_grad():
        pred_norm = predictor.infer_pil(image)

    # Revert the X axis
    pred_norm[:, 0, :, :] = -pred_norm[:, 0, :, :]

    # Convert to [0, 1]
    pred_norm = (pred_norm + 1) / 2.0

    return pred_norm # (1, 3, H, W)

def preprocess_image(input_image, dsine_model=None, device='cuda'):
    # 1. DSINE Normal Estimation on Original Image
    input_rgb = input_image.convert('RGB')
    if dsine_model is not None:
        normal_tensor = estimate_normal(input_rgb, dsine_model, device) # (1, 3, H, W)
        normal_np = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # (H, W, 3)
        normal_image = Image.fromarray((normal_np * 255).astype(np.uint8))
    else:
        normal_image = Image.new('RGB', input_image.size, (128, 128, 255))

    has_alpha = False
    if input_image.mode == 'RGBA':
        alpha = np.array(input_image)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input_image
    else:
        input_image = input_image.convert('RGB')
        max_size = max(input_image.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input_image = input_image.resize((int(input_image.width * scale), int(input_image.height * scale)), Image.Resampling.LANCZOS)
            # Also resize normal image if we resized input
            normal_image = normal_image.resize((int(normal_image.width * scale), int(normal_image.height * scale)), Image.Resampling.LANCZOS)
        
        session = rembg.new_session('birefnet-general')
        output = rembg.remove(input_image, session=session)
        
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    if len(bbox) == 0:
        bbox = [0, 0, output.height, output.width]
        bbox_crop = (0, 0, output.width, output.height)
    else:
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox_crop = (int(center[0] - size // 2), int(center[1] - size // 2), int(center[0] + size // 2), int(center[1] + size // 2))
    
    output = output.crop(bbox_crop)
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))

    # Process Normal
    normal_rgba = normal_image.convert('RGBA')
    
    # Create alpha mask image
    alpha_img = Image.fromarray(alpha)
    normal_rgba.putalpha(alpha_img)
    
    normal_crop = normal_rgba.crop(bbox_crop)
    normal_crop = normal_crop.resize((518, 518), Image.Resampling.LANCZOS)
    
    normal_np = np.array(normal_crop).astype(np.float32) / 255
    normal_np = normal_np[:, :, :3] * normal_np[:, :, 3:4]
    normal_output = Image.fromarray((normal_np * 255).astype(np.uint8))

    return output, normal_output

def encode_image(image, image_cond_model, device):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = np.array(image.convert('RGB')).astype(np.float32) / 255
    image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image_tensor = transform(image_tensor)
    
    with torch.no_grad():
        features = image_cond_model(image_tensor, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
    return patchtokens
