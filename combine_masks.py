import cv2
import numpy as np
import os

masks_dir = "masks"
output_path = "combined_mask.png"

combined_mask = None

for i in range(1, 10):
    mask_path = os.path.join(masks_dir, f"{i}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if combined_mask is None:
        combined_mask = np.zeros_like(mask)
    
    combined_mask = cv2.bitwise_or(combined_mask, mask)

cv2.imwrite(output_path, combined_mask)
print(f"Combined mask saved to {output_path}")