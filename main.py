"""
Underwater Image Enhancement Pipeline
Author: Computer Vision Engineer
Project: Underwater Image Preprocessing & CLAHE Enhancement

This script demonstrates underwater image enhancement using CLAHE
(Contrast Limited Adaptive Histogram Equalization) in LAB color space.

Usage:
    python main.py --input path/to/image.jpg
    python main.py --input path/to/image.jpg --save-output
    python main.py --input path/to/image.jpg --save-output --output-dir results/
"""

import cv2
import argparse
import os
from underwater_preprocessing import (
    preprocess_underwater_image,
    visualize_preprocessing_comparison,
    get_image_statistics
)


def process_image(image_path, save_output=False, output_dir='output'):
    """
    Complete preprocessing pipeline for underwater image.
    
    Args:
        image_path: Path to input underwater image
        save_output: Save images to disk instead of displaying (useful for WSL/headless)
        output_dir: Directory to save output images
    """
    # Load image
    print(f"\nProcessing: {image_path}")
    original = cv2.imread(image_path)
    
    if original is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image size: {original.shape[1]}x{original.shape[0]}")
    
    # Analyze original image statistics
    get_image_statistics(original, "Original")
    
    # Apply CLAHE enhancement
    enhanced = preprocess_underwater_image(original)
    
    # Analyze enhanced image statistics
    get_image_statistics(enhanced, "Enhanced")
    
    if save_output:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save comparison
        comparison = visualize_preprocessing_comparison(original, enhanced)
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")
        cv2.imwrite(comparison_path, comparison)
        print(f"\n✓ Saved comparison: {comparison_path}")
        
        # Save individual images
        original_path = os.path.join(output_dir, f"{base_name}_original.jpg")
        enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
        cv2.imwrite(original_path, original)
        cv2.imwrite(enhanced_path, enhanced)
        print(f"✓ Saved original: {original_path}")
        print(f"✓ Saved enhanced: {enhanced_path}")
        
        print(f"\n✓ Processing complete! All images saved to '{output_dir}/' directory")
    else:
        # Display results
        comparison = visualize_preprocessing_comparison(original, enhanced)
        cv2.imshow("Underwater Image Enhancement (CLAHE)", comparison)
        
        print("\n✓ Processing complete! Press any key to close window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Underwater Image Enhancement using CLAHE in LAB Color Space"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input underwater image'
    )
    parser.add_argument(
        '--save-output',
        action='store_true',
        help='Save output images to disk instead of displaying (useful for WSL/headless)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output images (default: output/)'
    )
    
    args = parser.parse_args()
    
    # Process image
    process_image(args.input, save_output=args.save_output, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
