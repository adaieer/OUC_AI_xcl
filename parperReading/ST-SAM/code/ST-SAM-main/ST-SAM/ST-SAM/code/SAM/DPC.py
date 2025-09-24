import cv2
import numpy as np
import time
from tqdm import tqdm  
import torch
import os
import shutil 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from SAM.segment_anything import SamPredictor, sam_model_registry

def is_point_inside_mask(mask, point):
    x, y = int(point[0]), int(point[1])
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x] == 255
    return False

def find_nearest_inner_point(contour, mask, max_search_steps=20):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    if is_point_inside_mask(mask, (cx, cy)):
        return (cx, cy)
    
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    angle_rad = np.deg2rad(angle)
    
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)
    
    for step in range(1, max_search_steps):
        for sign in [-1, 1]:
            x = int(cx + sign * step * dx)
            y = int(cy + sign * step * dy)
            if is_point_inside_mask(mask, (x, y)):
                return (x, y)
    return None

def load_mask_and_safe_centers(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    safe_centers = []
    for contour in contours:
        if len(contour) < 5:  
            continue
        point = find_nearest_inner_point(contour, binary_mask)
        if point is not None:
            safe_centers.append(point)
    
    return binary_mask, np.array(safe_centers)

import shutil  

def process_image_mask_pair(image_path, mask_path, output_path, predictor):
    image = cv2.imread(image_path)
    binary_mask, safe_centers = load_mask_and_safe_centers(mask_path)
    
    predictor.set_image(image)
    
    x, y, w, h = cv2.boundingRect(binary_mask)
    input_box = np.array([x, y, x + w, y + h])
    
    point_coords = safe_centers.astype(np.float32)
    point_labels = np.ones(len(safe_centers), dtype=np.int32)
    
    if len(point_coords) == 0:
        shutil.copy(mask_path, output_path)
        print(f"No valid points found for {image_path}. Mask copied to {output_path}.")
        return
    
    masks, scores, _ = predictor.predict(
        box=input_box[None, :],
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    
    best_mask = masks[0].astype(np.uint8) * 255
    cv2.imwrite(output_path, best_mask)
    print(f"Mask saved as {output_path}")


def process_selected_samples(image_dir, mask_dir, output_dir, selected_files, sam_checkpoint="/sam_vit_h_4b8939.pth"):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda')
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
    predictor = SamPredictor(sam)
    
    os.makedirs(output_dir, exist_ok=True)
    sam_mask_dir = './sam-mask'  
    os.makedirs(sam_mask_dir, exist_ok=True)
    
    for filename in tqdm(selected_files, desc="Processing selected samples"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, base_name + '.jpg')
        mask_path = os.path.join(mask_dir, filename)
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Skipping {filename}: image or mask not found.")
            continue
        
        image = cv2.imread(image_path)       
        predictor.set_image(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        _, safe_centers = load_mask_and_safe_centers(mask_path)  
        x, y, w, h = cv2.boundingRect(binary_mask)
        input_box = np.array([x, y, x + w, y + h])
        point_coords = safe_centers.astype(np.float32) if len(safe_centers) > 0 else None
        point_labels = np.ones(len(safe_centers), dtype=np.int32) if len(safe_centers) > 0 else None
        if point_coords is not None:
            masks, _, _ = predictor.predict(
                box=input_box[None, :],
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False
            )
        else:
            masks, _, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=False
            )
        sam_mask = masks[0].astype(np.uint8) * 255

        sam_mask_path = os.path.join(sam_mask_dir, filename)
        cv2.imwrite(sam_mask_path, sam_mask)

        weighted_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        fuse_mask = cv2.addWeighted(sam_mask, 0.5, weighted_mask, 0.5, 0)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, fuse_mask)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda')
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
    predictor = SamPredictor(sam)
    
    image_dir = "./image"
    mask_dir = "./mask"
    output_dir = "./output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    total_images = len(image_files)
    
    processed_count = 0  
    copied_count = 0  
    
    start_time = time.time()
    
    for image_name in tqdm(image_files, desc="Processing images"):
        base_name = os.path.splitext(image_name)[0]
        mask_name = base_name + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            print(f"Mask file {mask_name} not found for image {image_name}. Skipping.")
            continue

        image_path = os.path.join(image_dir, image_name)
        output_path = os.path.join(output_dir, mask_name)

        image = cv2.imread(image_path)
        binary_mask, safe_centers = load_mask_and_safe_centers(mask_path)

        predictor.set_image(image)

        x, y, w, h = cv2.boundingRect(binary_mask)
        input_box = np.array([x, y, x + w, y + h])

        point_coords = safe_centers.astype(np.float32)
        point_labels = np.ones(len(safe_centers), dtype=np.int32)

        if len(point_coords) == 0:
            shutil.copy(mask_path, output_path)
            copied_count += 1
            continue

        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False
        )

        best_mask = masks[0].astype(np.uint8) * 255
        cv2.imwrite(output_path, best_mask)
        processed_count += 1
    
    total_time = time.time() - start_time

    print("\nProcessing completed!")
    print(f"Total images: {total_images}")
    print(f"Processed images: {processed_count}")
    print(f"Copied masks: {copied_count}")
    print(f"Total time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()