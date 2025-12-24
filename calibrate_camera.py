#!/usr/bin/env python3
"""
Interactive camera calibration with live capture.
Hold checkerboard in different positions and press SPACE to capture.
"""

import cv2
import numpy as np
import os
from datetime import datetime

# Checkerboard dimensions (internal corners: 9x6 board has 8x5 corners)
CHECKERBOARD = (9, 6)  # Generated checkerboard: 10x7 squares = 9x6 internal corners
SQUARE_SIZE = 25.8  # mm - measure your printed squares!

# Camera
CAM_ID = 0

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Storage
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
images_captured = []

# Define 3D coordinates for checkerboard corners
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

print("=" * 70)
print("CAMERA CALIBRATION - Interactive Capture")
print("=" * 70)
print(f"Checkerboard: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} internal corners")
print(f"Square size: {SQUARE_SIZE}mm")
print()
print("Instructions:")
print("  1. Hold checkerboard in view")
print("  2. Move it to different positions, angles, distances")
print("  3. Cover all areas of the image (edges, corners, center)")
print("  4. Press SPACE when corners are detected (green)")
print("  5. Collect 15-30 good images")
print("  6. Press C to calibrate")
print("  7. Press Q to quit")
print("=" * 70)

cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

# Get camera resolution
ret, frame = cap.read()
if not ret:
    print("❌ Cannot read from camera")
    exit(1)

img_shape = frame.shape[:2]
print(f"Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    # Visualization
    display = frame.copy()
    
    if ret_corners:
        # Refine corners
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Draw corners
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners_refined, ret_corners)
        
        # Status: GREEN = ready to capture
        status_text = "READY - Press SPACE to capture"
        status_color = (0, 255, 0)
    else:
        # Status: RED = not detected
        status_text = "Move checkerboard into view"
        status_color = (0, 0, 255)
    
    # Draw status bar
    cv2.rectangle(display, (0, 0), (display.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Draw capture count
    info_text = f"Captured: {len(objpoints)}/15-30"
    cv2.putText(display, info_text, (10, display.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show calibration button hint
    if len(objpoints) >= 15:
        cal_text = "Press C to calibrate!"
        cv2.putText(display, cal_text, (10, display.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Camera Calibration", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' ') and ret_corners:
        # Capture this frame
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        images_captured.append(frame.copy())
        print(f"✓ Captured image {len(objpoints)}")
        
    elif key == ord('c') and len(objpoints) >= 15:
        # Start calibration
        print()
        print("=" * 70)
        print(f"Calibrating with {len(objpoints)} images...")
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
            mean_error /= len(objpoints)
            
            print(f"✓ Calibration successful!")
            print(f"  Reprojection error: {mean_error:.3f} pixels")
            print(f"  Focal length: fx={mtx[0,0]:.1f}, fy={mtx[1,1]:.1f}")
            print(f"  Principal point: cx={mtx[0,2]:.1f}, cy={mtx[1,2]:.1f}")
            
            # Save
            np.savez("calib.npz", mtx=mtx, dist=dist)
            print(f"✓ Saved to calib.npz")
            
            # Save example images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("photos", exist_ok=True)
            for i, img in enumerate(images_captured):
                filename = f"photos/calib_{timestamp}_{i:03d}.jpg"
                cv2.imwrite(filename, img)
            print(f"✓ Saved {len(images_captured)} images to photos/")
            print("=" * 70)
        else:
            print("❌ Calibration failed!")
        
        print()
        print("Press Q to quit or continue capturing...")
        
    elif key == ord('r'):
        # Reset
        objpoints.clear()
        imgpoints.clear()
        images_captured.clear()
        print("✗ Reset - all captures cleared")
        
    elif key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
