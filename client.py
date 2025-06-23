#!/usr/bin/env python3
"""
Simple client for SAM first mask generation
"""
import modal
import base64
import sys

def get_first_mask_from_image(image_path: str):
    """
    Get the first segmentation mask from an image file
    
    Args:
        image_path: Path to the image file
    """
    # Connect to deployed app
    get_first_mask = modal.Function.from_name("sam-segmentation", "get_first_mask")
    
    print(f"Processing {image_path}...")
    
    # Read image file
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Call remote function
    result = get_first_mask.remote(image_bytes)
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Success! Generated mask from {result['total_masks']} total masks")
    print(f"Mask area: {result['mask_area']}")
    print(f"Stability score: {result['stability_score']:.3f}")
    
    # Save the mask
    mask_data = base64.b64decode(result['mask_base64'])
    output_path = "first_mask.png"
    
    with open(output_path, "wb") as f:
        f.write(mask_data)
    
    print(f"First mask saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_client.py <image_path>")
        print("Example: python simple_client.py sample.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    get_first_mask_from_image(image_path)