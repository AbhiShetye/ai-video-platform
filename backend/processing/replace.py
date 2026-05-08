import cv2
import numpy as np
import os

def replace_region(frame_path, mask, replacement_color, output_path):
    frame = cv2.imread(frame_path)
    
    # Apply replacement color to masked region
    result = frame.copy()
    result[mask == 255] = replacement_color
    
    cv2.imwrite(output_path, result)
    return output_path

def replace_frames(frame_paths, masks, output_folder, replacement_color=(0, 255, 0)):
    os.makedirs(output_folder, exist_ok=True)
    output_paths = []

    for i, (frame_path, mask_data) in enumerate(zip(frame_paths, masks)):
        output_path = os.path.join(output_folder, f"replaced_{i:04d}.jpg")
        replace_region(frame_path, mask_data["mask"], replacement_color, output_path)
        output_paths.append(output_path)

    return output_paths