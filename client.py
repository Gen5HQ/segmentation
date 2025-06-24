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
result = fn.get_all_masks.remote(img_bytes)

if "error" in result:
    print(f"Error: {result['error']}")
else:
    masks = result["masks"]
    for i, mask in enumerate(masks):
        mask_data = base64.b64decode(mask["mask_base64"])
        filename = f"mask_{i+1}.png"
        with open(filename, "wb") as f:
            f.write(mask_data)
        print(f"✓ {filename} saved (area: {mask['area']}, score: {mask['stability_score']:.3f})")
    
    print(f"\nTotal masks found: {result['total_masks_found']}")
    print(f"Top masks saved: {result['returned_masks']}")