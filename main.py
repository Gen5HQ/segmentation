import modal
from typing import Dict, Any
import json

# Modal app configuration
app = modal.App("sam-segmentation")

# Define the image for the Modal function
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev", "libgomp1")
    .pip_install(
        "torch",
        "torchvision", 
        "transformers",
        "segment-anything",
        "opencv-python-headless",
        "Pillow",
        "numpy",
    )
)

@app.function(
    image=image,
    gpu="H100",
    memory=8192,
    timeout=600,
    enable_memory_snapshot=True,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def get_first_mask(image_bytes: bytes) -> Dict[str, Any]:
    """
    Get the first segmentation mask from an uploaded image
    
    Args:
        image_bytes: Input image as bytes
    
    Returns:
        Dictionary with the first mask as base64
    """
    import torch
    import numpy as np
    from PIL import Image
    import io
    import cv2
    import base64
    import os
    import urllib.request
    
    print("Processing image with SAM...")
    
    # Use the original SAM library
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    
    # Load SAM model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM on: {device}")
    
    # Use cached checkpoint
    checkpoint_path = "/tmp/sam_vit_h_4b8939.pth"
    
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM checkpoint...")
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print("✓ Download complete")
    else:
        print("✓ Using cached SAM checkpoint")
    
    # Load model
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Load and process image
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image_pil)
    
    print("Generating masks...")
    masks = mask_generator.generate(image_array)
    print(f"Generated {len(masks)} masks")
    
    if not masks:
        return {"error": "No masks generated"}
    
    # Get the first mask (usually the largest/most prominent)
    first_mask = masks[0]['segmentation']
    mask_np = first_mask.astype(np.uint8) * 255
    
    # Convert mask to base64
    _, buffer = cv2.imencode('.png', mask_np)
    mask_base64 = base64.b64encode(buffer).decode()
    
    return {
        "mask_base64": mask_base64,
        "total_masks": len(masks),
        "mask_area": masks[0].get('area', 0),
        "stability_score": masks[0].get('stability_score', 0)
    }

@app.local_entrypoint()
def main():
    """
    Simple test entrypoint
    """
    print("SAM First Mask Generator")
    print("Usage: Call get_first_mask.remote(image_bytes)")