import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import uniform_filter, gaussian_filter

def calculate_local_entropy(gray_mask, window_size=7, sigma=1):
    eps = 1e-8
    gray_normalized = gray_mask.astype(np.float32) / 255.0
    mean = uniform_filter(gray_normalized, size=window_size)
    p_foreground = mean
    p_background = 1 - p_foreground
    entropy = -p_foreground * np.log(p_foreground + eps) - p_background * np.log(p_background + eps)
    entropy = gaussian_filter(entropy, sigma=sigma)
    entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy) + eps)
    return entropy
    #return 1 - entropy

def calculate_entropy(mask):
    eps = 1e-8
    p_foreground = np.mean(mask)
    p_background = 1 - p_foreground
    entropy = -p_foreground * np.log(p_foreground + eps) - p_background * np.log(p_background + eps)
    return entropy

def image_level_filter(mask, tau_a=0.3, min_foreground_ratio=0.05):
    foreground_ratio = np.mean(mask)
    if foreground_ratio < min_foreground_ratio:
        return False
    global_entropy = calculate_entropy(mask)
    high_uncertainty_threshold = 0.5 * global_entropy
    local_entropy = calculate_local_entropy(mask)
    high_uncertainty = (local_entropy > high_uncertainty_threshold).astype(np.float32)
    u_a = np.mean(high_uncertainty)
    return u_a < tau_a
    #return u_a >= tau_a

def process_folder(input_dir, output_dir_b, output_dir_c, tau_a=0.3, min_foreground_ratio=0.05, k=1):
    os.makedirs(output_dir_b, exist_ok=True)
    os.makedirs(output_dir_c, exist_ok=True)
    total_entropy = 0
    valid_count = 0
    rejected_count = 0
    entropies = []
    
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.endswith(".png"): 
            continue
        path = os.path.join(input_dir, filename)
        gray_mask = np.array(Image.open(path))
        binary_mask = (gray_mask > 128).astype(np.float32)
        
        if image_level_filter(binary_mask, tau_a, min_foreground_ratio):
            cv2.imwrite(os.path.join(output_dir_b, filename), (binary_mask * 255).astype(np.uint8))
            local_entropy = calculate_local_entropy(gray_mask)
            #weight = 0.1 + 0.9 * (1 - local_entropy) ** k
            weight = 0.5 + 0.5 * (1 - local_entropy) ** k  
            weighted_mask = weight * gray_mask
            min_val, max_val = np.min(weighted_mask), np.max(weighted_mask)
            if max_val > min_val:
                weighted_mask = (weighted_mask - min_val) / (max_val - min_val) * 255
            weighted_mask = np.clip(weighted_mask, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir_c, filename), weighted_mask)
            current_entropy = np.mean(calculate_local_entropy(weighted_mask))
            entropies.append((filename, current_entropy))
            total_entropy += current_entropy
            valid_count += 1
        else:
            rejected_count += 1
    
    print(f"Valid/Total: {valid_count}/{valid_count + rejected_count}")
    print(f"Final Average Entropy: {total_entropy / valid_count if valid_count > 0 else 0:.4f}")
    return entropies