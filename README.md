# Underwater Image Enhancement

Image preprocessing pipeline using CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space for improved visibility in degraded underwater imagery for YOLO and SAM3 detection and segmentation.

---

**Technologies:** Python, OpenCV, CLAHE  
**Use Cases:** Underwater robotics, marine biology, offshore inspection, autonomous vehicles

---

## Features

- **CLAHE Enhancement**: Adaptive histogram equalization in LAB color space
- **Color Correction**: Gray world white balance algorithm
- **Statistics Analysis**: Mean, standard deviation, and contrast metrics
- **Comparison View**: Side-by-side original vs enhanced visualization

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process single image
python main.py --input test.jpg

# Process and save output (for WSL/headless)
python main.py --input test.jpg --save-output

# Custom output directory
python main.py --input test.jpg --save-output --output-dir results/
```

---

## Technical Details

### Why CLAHE in LAB Color Space?

**Problem**: Underwater images suffer from:
- Low contrast (light absorption)
- Color cast (wavelength-dependent attenuation)
- Backscatter (suspended particles)

**Solution**:
1. **LAB Color Space**: Separates luminance (L) from chrominance (A, B)
2. **CLAHE on L-channel**: Enhances contrast without amplifying color noise
3. **Gray World White Balance**: Corrects blue/green color cast
4. **Preserve Color**: A and B channels enhanced without distortion

**Result**: Improved visibility while maintaining natural color appearance.

### CLAHE Parameters
- **clip_limit=2.0**: Prevents over-amplification of noise
- **tileGridSize=(8,8)**: Local enhancement regions for adaptive equalization

### Enhancement Pipeline
1. **Gray World White Balance**: Corrects color cast (max gain: 4.0)
2. **BGR → LAB Conversion**: Separates luminance from color
3. **CLAHE on L-channel**: Adaptive histogram equalization
4. **LAB → BGR Conversion**: Back to standard color space

---

## Project Structure

```
├── main.py                      # Main processing script
├── underwater_preprocessing.py  # CLAHE enhancement module
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── output/                      # Saved results
```

---

## Example Output

```
Processing: underwater.jpg
Image size: 1920x1080

Original Statistics:
  Mean: 65.2, Std: 28.4, Contrast: 0.436

Enhanced Statistics:
  Mean: 127.8, Std: 45.6, Contrast: 0.357

✓ Saved comparison: output/underwater_comparison.jpg
✓ Saved original: output/underwater_original.jpg
✓ Saved enhanced: output/underwater_enhanced.jpg

✓ Processing complete! All images saved to 'output/' directory
```
