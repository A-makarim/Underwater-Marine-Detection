"""Underwater image enhancement using CLAHE in LAB color space"""

import cv2
import numpy as np


def apply_gray_world_white_balance(image):
    """Apply Gray World white balance to correct color cast"""
    avg_b = np.mean(image[:, :, 0]) 
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    scale_b = min(avg_gray / avg_b, 4.0)
    scale_g = min(avg_gray / avg_g, 4.0)
    scale_r = min(avg_gray / avg_r, 4.0)
    
    balanced = image.astype(np.float32)
    balanced[:, :, 0] = np.clip(balanced[:, :, 0] * scale_b, 0, 255)
    balanced[:, :, 1] = np.clip(balanced[:, :, 1] * scale_g, 0, 255)
    balanced[:, :, 2] = np.clip(balanced[:, :, 2] * scale_r, 0, 255)
    
    return balanced.astype(np.uint8)


def preprocess_underwater_image(image):
    # Split into Blue, Green, Red
    b, g, r = cv2.split(image)

    # decrease blue channel intensity to reduce blue cast

    # === STEP 1: PHYSICS-BASED RED CHANNEL COMPENSATION ===
    # Instead of multiplying everything by a huge number (which makes pink noise),
    # we use the Green channel to "guess" where the Red should be.
    # Formula: New_Red = Red + (Green - Red) * (1 - Red) * Green
    # This only boosts Red where Green is present (the coral), ignoring the background.
    
    # Work in float to prevent overflow
    r_f = r.astype(float) / 255.0
    g_f = g.astype(float) / 255.0
    
    # The Ancuti Formula for red compensation
    # We add red signal based on the difference between G and R
    compensated_r = r_f + (g_f - r_f) * (1.0 - r_f) * g_f
    
    # Convert back to uint8
    r_new = np.clip(compensated_r * 255, 0, 255).astype(np.uint8)

    # Re-merge with the new Red channel
    image_compensated = cv2.merge([b, g, r_new])

    # === STEP 2: GRAY WORLD WHITE BALANCE (With Clamping) ===
    # Now that we have a decent Red channel, we balance the colors safely.
    result = cv2.cvtColor(image_compensated, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    # Shift the A and B channels towards 128 (Neutral Gray)
    # But limit the shift to avoid over-correction
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.2)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.2)

    # === STEP 3: CLAHE (Luminance Only) ===
    # Fix the contrast in the L channel
    l, a, b_lab = cv2.split(result)
    
    # Clip Limit 2.0 is safer. 3.0 might be too gritty.
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    l = clahe.apply(l)

    # Merge and convert back
    final_lab = cv2.merge((l, a, b_lab))
    final = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)

    return final

def visualize_preprocessing_comparison(original, enhanced, window_name="Preprocessing"):
    """Create side-by-side comparison of original and enhanced images"""
    h, w = original.shape[:2]
    comparison = np.hstack([original, enhanced])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Enhanced (CLAHE)", (w + 10, 30), font, 1, (0, 255, 0), 2)
    
    return comparison


def get_image_statistics(image, name="Image"):
    """Calculate and print image statistics"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print(f"\n{name} Statistics:")
    print(f"  Mean: {np.mean(gray):.2f}")
    print(f"  Std: {np.std(gray):.2f}")
    print(f"  Min: {np.min(gray)}, Max: {np.max(gray)}")
    print(f"  Contrast: {np.std(gray)/np.mean(gray):.3f}")
