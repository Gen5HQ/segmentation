import modal
import pathlib
import base64
import json
from PIL import Image
import io

SamMask = modal.Cls.from_name("sam-first-mask-fast", "SamMask")
fn = SamMask()

img_path = "sample.JPG"
img_bytes = pathlib.Path(img_path).read_bytes()
result = fn.get_first_mask.remote(img_bytes)

if "error" in result:
    print(f"Error: {result['error']}")
else:
    mask_data = base64.b64decode(result["mask_base64"])
    with open("mask_output.png", "wb") as f:
        f.write(mask_data)
    
    print(f"Mask saved to mask_output.png")
    print(f"Total masks found: {result['total_masks']}")
    print(f"Mask area: {result['mask_area']}")
    print(f"Stability score: {result['stability_score']:.3f}")