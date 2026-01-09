#!/usr/bin/env python3
"""
Complete cube calibration: optimize both rotation and translation offsets
so all visible tags produce the same cube pose.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import sys
from collections import defaultdict
import threading

from april import TAG_FAMILY, MARKER_SIZE, CAM_ID, estimate_pose

# Load calibration
try:
    calib = np.load("camera.npz")
    mtx, dist = calib["mtx"], calib["dist"]
except FileNotFoundError:
    print("❌ Camera calibration not found!")
    print("   Run: python calibrate_camera.py")
    sys.exit(1)

# Store observations: list of (frame_data)
# where frame_data = {tid: (tvec, rvec)}
observations = []

print("=" * 70)
print("COMPLETE CUBE CALIBRATION")
print("=" * 70)
print()
print("Show the cube with MULTIPLE tags visible.")
print("Move it around to different angles.")
print("Frames with 2+ tags are automatically captured.")
print("Collect 50+ frames, then compute offsets.")
print()
print("Controls:")
print("  C     = Compute offsets and save")
print("  R     = Reset/clear observations")
print("  Q/ESC = Quit")
print("=" * 70)

def rotation_matrix_to_params(R_mat):
    """Convert rotation matrix to 3 parameters (axis-angle)."""
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return rotvec

def params_to_rotation_matrix(params):
    """Convert 3 parameters to rotation matrix."""
    return R.from_rotvec(params).as_matrix()

def compute_offsets_optimized(observations, optimize_camera=True):
    """
    Optimize rot_offset and trans_offset for all tags simultaneously.
    Optionally also refine camera calibration parameters.
    
    Objective: minimize reprojection error + pose consistency.
    """
    # Get all tag IDs
    all_tids = set()
    for frame in observations:
        all_tids.update(frame.keys())
    all_tids = sorted(all_tids)
    
    if len(all_tids) < 2:
        print("Need at least 2 different tags")
        return None, None, None, None
    
    # Force tag 0 to be in the list as reference
    if 0 not in all_tids:
        print("⚠️  Tag 0 must be visible for reference!")
        return None, None, None, None
    
    # Reorder so tag 0 is first
    if all_tids[0] != 0:
        all_tids.remove(0)
        all_tids = [0] + all_tids
    
    print(f"\nOptimizing for tags: {all_tids}")
    print(f"Using {len(observations)} observations")
    print(f"Reference: Tag 0 = identity")
    if optimize_camera:
        print("Also refining camera parameters")
    
    # Initial guess: tag 0 is identity, others need to be found
    # Each tag has: 3 rotation params + 3 translation params = 6 params
    # Tag 0 is reference: rot=[0,0,0], trans=[0,0,-0.025]
    n_tags = len(all_tids)
    n_params = (n_tags - 1) * 6  # Tag 0 is fixed
    
    # Camera parameters: fx, fy, cx, cy, k1, k2, p1, p2, k3 (9 params)
    if optimize_camera:
        n_params += 9
    
    # Initial guess for other tags - use simple rotations
    # Tags typically on cube faces: 0°, 90°, 180°, 270° rotations
    x0 = []
    for i, tid in enumerate(all_tids[1:]):  # Skip tag 0
        # Start with assumption of 90° rotations between faces
        if tid == 1:  # Opposite face: 180° around X
            x0.extend([np.pi, 0, 0])
        elif tid == 2:  # Right face: 90° around Y
            x0.extend([0, np.pi/2, 0])
        elif tid == 3:  # Left face: -90° around Y
            x0.extend([0, -np.pi/2, 0])
        elif tid == 4:  # Top face: 90° around X
            x0.extend([np.pi/2, 0, 0])
        elif tid == 5:  # Bottom face: -90° around X
            x0.extend([-np.pi/2, 0, 0])
        else:
            x0.extend([0, 0, 0])
        
        x0.extend([0, 0, -0.025])  # translation
    
    # Camera parameters initial values (from current calibration)
    if optimize_camera:
        x0.extend([mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2]])  # fx, fy, cx, cy
        x0.extend([dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3], dist[0, 4]])  # k1, k2, p1, p2, k3
    
    x0 = np.array(x0)
    
    # Store initial camera params for regularization
    initial_camera_params = x0[-(9 if optimize_camera else 0):] if optimize_camera else None
    
    def unpack_params(x):
        """Convert flat parameter vector to rot_offset, trans_offset, and camera params."""
        rot_offset = {0: np.eye(3)}
        trans_offset = {0: np.array([0, 0, -0.025])}
        
        idx = 0
        for tid in all_tids[1:]:
            rot_params = x[idx:idx+3]
            trans_params = x[idx+3:idx+6]
            rot_offset[tid] = params_to_rotation_matrix(rot_params)
            trans_offset[tid] = trans_params
            idx += 6
        
        if optimize_camera:
            camera_params = x[idx:idx+9]
            K = np.array([
                [camera_params[0], 0, camera_params[2]],
                [0, camera_params[1], camera_params[3]],
                [0, 0, 1]
            ])
            D = camera_params[4:9].reshape(1, 5)
            return rot_offset, trans_offset, K, D
        else:
            return rot_offset, trans_offset, mtx, dist
    
    def objective(x):
        """
        Cost function: pose consistency + camera regularization.
        """
        rot_offset, trans_offset, K, D = unpack_params(x)
        
        total_cost = 0
        
        for frame in observations:
            # Compute cube pose from each tag in this frame
            cube_positions = []
            cube_rotations = []
            
            for tid, (tvec, rvec) in frame.items():
                if tid not in rot_offset:
                    continue
                
                R_tag, _ = cv2.Rodrigues(rvec)
                t_tag = tvec.flatten()
                
                # Cube pose from this tag
                R_cube = R_tag @ rot_offset[tid]
                t_cube = t_tag + R_tag @ trans_offset[tid]
                
                cube_positions.append(t_cube)
                cube_rotations.append(R_cube)
            
            if len(cube_positions) < 2:
                continue
            
            # Cost: variance of positions and rotations
            positions = np.array(cube_positions)
            pos_variance = np.sum(np.var(positions, axis=0))
            
            # For rotations: measure pairwise angular differences
            rot_cost = 0
            for i in range(len(cube_rotations)):
                for j in range(i+1, len(cube_rotations)):
                    R_diff = cube_rotations[i].T @ cube_rotations[j]
                    trace = np.trace(R_diff)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    rot_cost += angle ** 2
            
            total_cost += pos_variance * 1000 + rot_cost * 10  # Scale position cost
        
        # Regularization: keep camera params close to initial values
        if optimize_camera and initial_camera_params is not None:
            camera_params = x[-(9 if optimize_camera else 0):]
        # Regularization: keep camera params close to initial values
        if optimize_camera and initial_camera_params is not None:
            camera_params = x[-(9 if optimize_camera else 0):]
            camera_regularization = np.sum((camera_params - initial_camera_params) ** 2) * 100
            total_cost += camera_regularization
        
        return total_cost
    
    print("\nOptimizing offsets...")
    result = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        print("\n✓ Optimization converged")
        rot_offset, trans_offset, K_opt, D_opt = unpack_params(result.x)
        
        # Check for unusual rotation angles (should be near 90° multiples for cube)
        print("\n⚠️  Checking rotation offset quality...")
        warnings = []
        for tid in sorted(rot_offset.keys()):
            if tid == 0:
                continue  # Tag 0 is reference (identity)
            euler = R.from_matrix(rot_offset[tid]).as_euler('xyz', degrees=True)
            # Check each angle - should be near 0, ±90, ±180
            max_deviation = 0
            for i, angle in enumerate(euler):
                # Find nearest 90° multiple
                nearest_90 = round(angle / 90) * 90
                deviation = abs(angle - nearest_90)
                # Wrap to [-180, 180]
                if deviation > 180:
                    deviation = 360 - deviation
                max_deviation = max(max_deviation, deviation)
            
            if max_deviation > 15:
                warnings.append(f"  Tag {tid}: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]° (max deviation: {max_deviation:.1f}°)")
        
        if warnings:
            print("  ⚠️  WARNING: Some tags have unusual rotation offsets:")
            for w in warnings:
                print(w)
            print("  This may indicate:")
            print("    - Insufficient observations for those tags")
            print("    - Poor optimizer convergence")
            print("    - Tags physically misaligned on cube")
            print("  Consider recalibrating with more observations.")
        else:
            print("  ✓ All rotation offsets look good (near 90° multiples)")
        
        if optimize_camera:
            print("\nRefined camera parameters:")
            print(f"  fx: {mtx[0,0]:.1f} → {K_opt[0,0]:.1f}")
            print(f"  fy: {mtx[1,1]:.1f} → {K_opt[1,1]:.1f}")
            print(f"  cx: {mtx[0,2]:.1f} → {K_opt[0,2]:.1f}")
            print(f"  cy: {mtx[1,2]:.1f} → {K_opt[1,2]:.1f}")
        
        return rot_offset, trans_offset, K_opt, D_opt
    else:
        print("\n⚠️  Optimization failed")
        return None, None, None, None

def save_and_update(rot_offset, trans_offset, K_new=None, D_new=None):
    """Save cube offsets and optionally update camera calibration."""
    
    # Save camera calibration if provided
    if K_new is not None and D_new is not None:
        np.savez("camera.npz", mtx=K_new, dist=D_new)
        print("💾 Updated camera.npz with refined camera parameters")
    
    # Save cube offsets to file (secure format without pickle)
    save_data = {}
    for tag_id in trans_offset.keys():
        save_data[f'trans_{tag_id}'] = trans_offset[tag_id]
        save_data[f'rot_{tag_id}'] = rot_offset[tag_id]
    np.savez("cube_offsets.npz", **save_data)
    print("💾 Saved cube offsets to cube_offsets.npz (secure format)")
    
    # Print summary
    print("\nOffset summary:")
    for tid in sorted(trans_offset.keys()):
        print(f"  Tag {tid}:")
        euler = R.from_matrix(rot_offset[tid]).as_euler('xyz', degrees=True)
        print(f"    Rotation: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")
        print(f"    Translation: [{trans_offset[tid][0]*1000:.2f}, {trans_offset[tid][1]*1000:.2f}, {trans_offset[tid][2]*1000:.2f}] mm")
    
    return True

# Setup
cap = cv2.VideoCapture(CAM_ID)

if not cap.isOpened():
    sys.exit("No webcam")

detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(TAG_FAMILY),
    cv2.aruco.DetectorParameters()
)

computed_rot = None
computed_trans = None
computed_K = None
computed_D = None
frame_skip = 0  # Skip frames to avoid too many similar frames
CAPTURE_EVERY_N = 5  # Capture every Nth frame when tags visible
computing = False  # Flag to indicate optimization in progress
compute_thread = None

def compute_in_background():
    """Run optimization in background thread."""
    global computed_rot, computed_trans, computed_K, computed_D, computing
    computing = True
    try:
        computed_rot, computed_trans, computed_K, computed_D = compute_offsets_optimized(observations, optimize_camera=True)
        if computed_rot:
            print("\n✓ Computed offsets:")
            for tid in sorted(computed_rot.keys()):
                print(f"  Tag {tid}:")
                print(f"    rot: {R.from_matrix(computed_rot[tid]).as_euler('xyz', degrees=True)}")
                print(f"    trans: {computed_trans[tid]}")
            # Auto-save
            save_and_update(computed_rot, computed_trans, computed_K, computed_D)
    except Exception as e:
        print(f"\n⚠️  Optimization error: {e}")
    finally:
        computing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Single camera detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    h, w = frame.shape[:2]
    
    # Auto-capture frames with 2+ good tags
    auto_captured = False
    if ids is not None and len(ids) >= 2:
        frame_skip += 1
        if frame_skip >= CAPTURE_EVERY_N:
            frame_skip = 0
            
            # Build frame data, checking quality
            frame_data = {}
            QUALITY_THRESHOLD = 5.0  # pixels - increase if too many good tags are rejected
            
            for corner, id_arr in zip(corners, ids):
                tid = int(id_arr[0])
                rvec, tvec, error = estimate_pose([corner], MARKER_SIZE, mtx, dist)
                
                # Only include high-quality detections
                if error[0] <= QUALITY_THRESHOLD:
                    frame_data[tid] = (tvec[0].flatten(), rvec[0])
            
            # Only save if we still have 2+ good tags
            if len(frame_data) >= 2:
                observations.append(frame_data)
                auto_captured = True
    
    # Visualization
    display_frame = frame
    h_display, w_display = h, w
    if ids is not None:
        # Draw with quality feedback
        for corner, id_arr in zip(corners, ids):
            _, _, error = estimate_pose([corner], MARKER_SIZE, mtx, dist)
            color = (0, 255, 0) if error[0] <= 5.0 else (0, 0, 255)
            cv2.aruco.drawDetectedMarkers(display_frame, [corner], np.array([[id_arr]]), borderColor=color)
    
    if ids is not None:
        n_tags = len(ids)
        
        if n_tags >= 2:
            color = (0, 255, 0) if auto_captured else (0, 255, 255)
            msg = f"{n_tags} tags visible" + (" - CAPTURED" if auto_captured else "")
        else:
            color = (150, 150, 150)
            msg = f"Only {n_tags} tag - need 2+"
        
        cv2.putText(display_frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw tag axes
        for corner, id_arr in zip(corners, ids):
            rvec, tvec, _ = estimate_pose([corner], MARKER_SIZE, mtx, dist)
            cv2.drawFrameAxes(display_frame, mtx, dist, rvec[0], tvec[0], MARKER_SIZE * 0.3)
    
    # Count observations per tag
    tag_counts = {}
    for frame_data in observations:
        for tid in frame_data.keys():
            tag_counts[tid] = tag_counts.get(tid, 0) + 1
    
    # Show observation count
    cv2.putText(display_frame, f"Frames captured: {len(observations)}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show per-tag counts
    y = 100
    cv2.putText(display_frame, "Tag observations:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    for tid in sorted(tag_counts.keys()):
        count = tag_counts[tid]
        color = (0, 255, 0) if count >= 20 else (150, 150, 150)
        cv2.putText(display_frame, f"  Tag {tid}: {count}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 20
    
    # Show computing status
    if computing:
        cv2.putText(display_frame, "Computing and saving... Please wait", 
                   (10, h_display-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif computed_rot and computed_trans:
        cv2.putText(display_frame, "Offsets saved! Press R to recalibrate", 
                   (10, h_display-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display_frame, "C=Compute&Save | R=Reset | Q=Quit",
               (10, h_display-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Complete Calibration", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r') or key == ord('R'):
        observations = []
        computed_rot = None
        computed_trans = None
        computed_K = None
        computed_D = None
        computing = False
        print("\n🔄 Reset observations")
    
    elif key == ord('c') or key == ord('C'):
        if not computing and len(observations) >= 20:
            # Start optimization in background thread
            compute_thread = threading.Thread(target=compute_in_background, daemon=True)
            compute_thread.start()
            print("\n🔧 Starting optimization in background...")
        elif computing:
            print("⚠️  Already computing...")
        else:
            print(f"⚠️  Need 20+ observations (have {len(observations)})")
    
    elif key == ord('q') or key == ord('Q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
