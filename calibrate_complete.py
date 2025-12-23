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

from april import TAG_FAMILY, MARKER_SIZE, CAM_ID, estimate_pose

# Load calibration
calib = np.load("calib.npz")
mtx, dist = calib["mtx"], calib["dist"]

# Store observations: list of (frame_data)
# where frame_data = {tid: (tvec, rvec)}
observations = []

print("=" * 70)
print("COMPLETE CUBE CALIBRATION")
print("=" * 70)
print()
print("Show the cube with MULTIPLE tags visible.")
print("Move it around to different angles.")
print("Collect 50+ frames with 2+ tags visible.")
print()
print("Controls:")
print("  SPACE = Capture frame")
print("  C     = Compute offsets")
print("  S     = Save and update april.py")
print("  Q/ESC = Quit")
print("=" * 70)

def rotation_matrix_to_params(R_mat):
    """Convert rotation matrix to 3 parameters (axis-angle)."""
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return rotvec

def params_to_rotation_matrix(params):
    """Convert 3 parameters to rotation matrix."""
    return R.from_rotvec(params).as_matrix()

def compute_offsets_optimized(observations):
    """
    Optimize rot_offset and trans_offset for all tags simultaneously.
    
    Objective: minimize variance of cube poses computed from different tags
    in the same frame.
    """
    # Get all tag IDs
    all_tids = set()
    for frame in observations:
        all_tids.update(frame.keys())
    all_tids = sorted(all_tids)
    
    if len(all_tids) < 2:
        print("Need at least 2 different tags")
        return None, None
    
    print(f"\nOptimizing for tags: {all_tids}")
    print(f"Using {len(observations)} observations")
    
    # Initial guess: tag 0 is identity, others need to be found
    # Each tag has: 3 rotation params + 3 translation params = 6 params
    # Tag 0 is reference: rot=[0,0,0], trans=[0,0,-0.025]
    n_tags = len(all_tids)
    n_params = (n_tags - 1) * 6  # Tag 0 is fixed
    
    # Initial guess for other tags
    x0 = []
    for tid in all_tids[1:]:  # Skip tag 0
        x0.extend([0, 0, 0])  # rotation
        x0.extend([0, 0, -0.025])  # translation
    
    x0 = np.array(x0)
    
    def unpack_params(x):
        """Convert flat parameter vector to rot_offset and trans_offset dicts."""
        rot_offset = {all_tids[0]: np.eye(3)}
        trans_offset = {all_tids[0]: np.array([0, 0, -0.025])}
        
        idx = 0
        for tid in all_tids[1:]:
            rot_params = x[idx:idx+3]
            trans_params = x[idx+3:idx+6]
            rot_offset[tid] = params_to_rotation_matrix(rot_params)
            trans_offset[tid] = trans_params
            idx += 6
        
        return rot_offset, trans_offset
    
    def objective(x):
        """
        Cost function: variance of cube poses within each frame.
        """
        rot_offset, trans_offset = unpack_params(x)
        
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
        
        return total_cost
    
    print("\nOptimizing offsets...")
    result = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'disp': True})
    
    if result.success:
        print("\n✓ Optimization converged")
        rot_offset, trans_offset = unpack_params(result.x)
        return rot_offset, trans_offset
    else:
        print("\n⚠️  Optimization failed")
        return None, None

def save_and_update(rot_offset, trans_offset):
    """Save and update april.py."""
    # Save to file
    with open("calibrated_offsets.txt", "w") as f:
        f.write("# Optimized offsets\n")
        f.write("from scipy.spatial.transform import Rotation as R\n")
        f.write("import numpy as np\n\n")
        
        f.write("trans_offset = {\n")
        for tid in sorted(trans_offset.keys()):
            t = trans_offset[tid]
            f.write(f"    {tid}: np.array([{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]),\n")
        f.write("}\n\n")
        
        f.write("rot_offset = {\n")
        for tid in sorted(rot_offset.keys()):
            R_mat = rot_offset[tid]
            if np.allclose(R_mat, np.eye(3)):
                f.write(f"    {tid}: np.eye(3),\n")
            else:
                euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                f.write(f"    {tid}: R.from_euler('xyz', [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}], degrees=True).as_matrix(),\n")
        f.write("}\n")
    
    print(f"\n💾 Saved to calibrated_offsets.txt")
    
    # Update april.py
    try:
        with open("april.py", "r") as f:
            content = f.read()
        
        # Replace trans_offset
        trans_lines = ["trans_offset = {  # tag frame → cube-COM translation (metres)\n"]
        for tid in sorted(trans_offset.keys()):
            t = trans_offset[tid]
            trans_lines.append(f"    {tid}: np.array([{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]),\n")
        trans_lines.append("}\n")
        
        # Replace rot_offset
        rot_lines = ["rot_offset = {\n"]
        for tid in sorted(rot_offset.keys()):
            R_mat = rot_offset[tid]
            if np.allclose(R_mat, np.eye(3)):
                rot_lines.append(f"    {tid}: np.eye(3),\n")
            else:
                euler = R.from_matrix(R_mat).as_euler('xyz', degrees=True)
                rot_lines.append(f"    {tid}: R.from_euler('xyz', [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}], degrees=True).as_matrix(),\n")
        rot_lines.append("}\n")
        
        # Find and replace sections
        import re
        
        # Replace trans_offset
        pattern = r'trans_offset = \{[^}]*\}'
        replacement = ''.join(trans_lines).rstrip()
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Replace rot_offset
        pattern = r'rot_offset = \{[^}]*\}'
        replacement = ''.join(rot_lines).rstrip()
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        with open("april.py", "w") as f:
            f.write(content)
        
        print("✓ Updated april.py")
        return True
        
    except Exception as e:
        print(f"⚠️  Could not update april.py: {e}")
        return False

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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    
    h, w = frame.shape[:2]
    
    if ids is not None:
        n_tags = len(ids)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        if n_tags >= 2:
            color = (0, 255, 0)
            msg = f"{n_tags} tags visible - SPACE to capture"
        else:
            color = (0, 255, 255)
            msg = f"Only {n_tags} tag - need 2+"
        
        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw tag axes
        for corner, id_arr in zip(corners, ids):
            rvec, tvec, _ = estimate_pose([corner], MARKER_SIZE, mtx, dist)
            cv2.drawFrameAxes(frame, mtx, dist, rvec[0], tvec[0], MARKER_SIZE * 0.3)
    
    # Show observation count
    cv2.putText(frame, f"Frames captured: {len(observations)}", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if computed_rot and computed_trans:
        cv2.putText(frame, "Offsets computed! Press S to save", 
                   (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, "SPACE=Capture | C=Compute | S=Save | Q=Quit", 
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow("Complete Calibration", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):
        if ids is not None and len(ids) >= 2:
            frame_data = {}
            for corner, id_arr in zip(corners, ids):
                tid = int(id_arr[0])
                rvec, tvec, _ = estimate_pose([corner], MARKER_SIZE, mtx, dist)
                frame_data[tid] = (tvec[0].flatten(), rvec[0])
            
            observations.append(frame_data)
            print(f"✓ Captured frame {len(observations)} with {len(frame_data)} tags")
        else:
            print("⚠️  Need 2+ tags visible")
    
    elif key == ord('c') or key == ord('C'):
        if len(observations) >= 20:
            computed_rot, computed_trans = compute_offsets_optimized(observations)
            if computed_rot:
                print("\n✓ Computed offsets:")
                for tid in sorted(computed_rot.keys()):
                    print(f"  Tag {tid}:")
                    print(f"    rot: {R.from_matrix(computed_rot[tid]).as_euler('xyz', degrees=True)}")
                    print(f"    trans: {computed_trans[tid]}")
        else:
            print(f"⚠️  Need 20+ observations (have {len(observations)})")
    
    elif key == ord('s') or key == ord('S'):
        if computed_rot and computed_trans:
            save_and_update(computed_rot, computed_trans)
        else:
            print("⚠️  Compute offsets first (press C)")
    
    elif key == ord('q') or key == ord('Q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
