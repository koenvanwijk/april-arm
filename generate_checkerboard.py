#!/usr/bin/env python3
"""Generate a checkerboard calibration pattern for camera calibration."""

import cv2
import numpy as np

# Checkerboard dimensions (internal corners)
ROWS = 6  # Number of internal corners vertically
COLS = 9  # Number of internal corners horizontally

# Square size in pixels (for printing on A4: ~50mm squares at 300 DPI)
SQUARE_SIZE_PX = 590  # ~50mm at 300 DPI

# Calculate image size
img_width = (COLS + 1) * SQUARE_SIZE_PX
img_height = (ROWS + 1) * SQUARE_SIZE_PX

# Create checkerboard
checkerboard = np.zeros((img_height, img_width), dtype=np.uint8)

for i in range(ROWS + 1):
    for j in range(COLS + 1):
        if (i + j) % 2 == 0:
            y1 = i * SQUARE_SIZE_PX
            y2 = (i + 1) * SQUARE_SIZE_PX
            x1 = j * SQUARE_SIZE_PX
            x2 = (j + 1) * SQUARE_SIZE_PX
            checkerboard[y1:y2, x1:x2] = 255

# Save as high-res PNG
cv2.imwrite("checkerboard_9x6.png", checkerboard)
print(f"✓ Checkerboard saved: checkerboard_9x6.png")
print(f"  Size: {COLS+1}x{ROWS+1} squares ({COLS}x{ROWS} internal corners)")
print(f"  Image: {img_width}x{img_height} pixels")
print(f"  Square size: ~50mm when printed on A4")
print()
print("To print:")
print("  1. Open checkerboard_9x6.png")
print("  2. Print on A4 paper (landscape)")
print("  3. Measure actual square size after printing")
print("  4. Update MARKER_SIZE in calibrate.py if needed")
