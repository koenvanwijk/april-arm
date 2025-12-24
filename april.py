import json
import sys
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
from cv2.typing import MatLike
from scipy.spatial.transform import Rotation as R

# ---------- constants ------------------------------------------------------
TAG_FAMILY = cv2.aruco.DICT_APRILTAG_16h5
MARKER_SIZE = 0.046  # 46 mm
CAM_ID = 0

# Load camera rotation or use default
try:
    CAMERA_ROTATION = np.load('camera_rotation.npz')['rotation_matrix']
    print("✓ Loaded camera rotation from camera_rotation.npz")
except FileNotFoundError:
    # Default: 30° tilt
    CAMERA_TILT_DEGREES = 30.0
    tilt_rad = np.radians(CAMERA_TILT_DEGREES)
    CAMERA_ROTATION = np.array([
        [1, 0, 0],
        [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
        [0, np.sin(tilt_rad), np.cos(tilt_rad)]
    ], dtype=np.float64)
    print(f"⚠️  Using default camera rotation (30° tilt)")
# ---------- cube geometry --------------------------------------------------
cube_length = 0.05
cube_half = cube_length / 2

# Cube offsets: load from calibration file or use defaults
try:
    cube_offsets_data = np.load("cube_offsets.npz", allow_pickle=True)
    trans_offset = cube_offsets_data['trans_offset'].item()
    rot_offset = cube_offsets_data['rot_offset'].item()
    print(f"✓ Loaded cube offsets from cube_offsets.npz")
except FileNotFoundError:
    # Default offsets (tag frame → cube-COM)
    trans_offset = {
        0: np.array([0.000000, 0.000000, -0.025000]),
        1: np.array([-0.000400, -0.001444, -0.024548]),
        2: np.array([-0.001020, -0.001139, -0.027475]),
        3: np.array([0.003198, 0.001266, -0.023672]),
        4: np.array([-0.006244, -0.004858, -0.023419]),
        5: np.array([0.000696, -0.001491, -0.027059]),
    }
    
    rot_offset = {
        0: np.eye(3),
        1: R.from_euler('xyz', [-179.4, 1.4, 179.9], degrees=True).as_matrix(),
        2: R.from_euler('xyz', [177.8, -89.2, -178.6], degrees=True).as_matrix(),
        3: R.from_euler('xyz', [-155.2, 88.0, -155.2], degrees=True).as_matrix(),
        4: R.from_euler('xyz', [89.7, 3.7, 3.0], degrees=True).as_matrix(),
        5: R.from_euler('xyz', [-91.1, 0.2, 179.4], degrees=True).as_matrix(),
    }
    print(f"Using default cube offsets (run calibrate_cube.py to calibrate)")




def estimate_pose(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and reprojection errors
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    rvecs = []
    tvecs = []
    errors = []

    for c in corners:
        success, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if success:
            # Calculate reprojection error as quality metric (average per corner)
            projected, _ = cv2.projectPoints(marker_points, R, t, mtx, distortion)
            # c has shape (1, 4, 2), projected has shape (4, 1, 2)
            error = cv2.norm(c.reshape(4, 2), projected.reshape(4, 2), cv2.NORM_L2) / 4.0
            errors.append(error)
        else:
            errors.append(float('inf'))
        
        rvecs.append(R)
        tvecs.append(t)
    
    return rvecs, tvecs, errors


@dataclass
class Pose:
    pos: npt.NDArray[np.float64]  # (x,y,z) for MuJoCo
    quat: npt.NDArray[np.float64]  # (x,y,z,w) for MuJoCo
    yaw: float = 0.0  # Rotation around world Z-axis in radians


# ---------- pose smoother --------------------------------------------------
class PoseLPF:
    def __init__(self, a_pos=0, a_rot=0):
        self.a_pos = a_pos
        self.a_rot = a_rot
        self.pos = None
        self.quat = None

    def update(self, t: np.ndarray, r: np.ndarray):
        t = t.reshape(3)
        q = R.from_rotvec(r.reshape(3)).as_quat()
        if self.pos is None:
            self.pos = t
        else:
            self.pos = self.a_pos * self.pos + (1 - self.a_pos) * t
        if self.quat is None:
            self.quat = q
        else:
            if np.dot(self.quat, q) < 0:
                q = -q
            self.quat = self.a_rot * self.quat + (1 - self.a_rot) * q
            self.quat /= np.linalg.norm(self.quat)
        return self.pos.copy(), R.from_quat(self.quat).as_rotvec().reshape(1, 1, 3)


cube_filter = PoseLPF(0.7, 0.5)  # smooth final COM pose
tag_filters: dict[int, PoseLPF] = {}  # per-tag smoothing


# ---------- helpers --------------------------------------------------------
def average_rot(mats: list[np.ndarray]) -> np.ndarray:
    """Return the orthonormal mean of rotation matrices."""
    M = sum(mats) / len(mats)
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


def get_tag_filter(tid: int) -> PoseLPF:
    if tid not in tag_filters:
        tag_filters[tid] = PoseLPF(0.5, 0.5)
    return tag_filters[tid]


# ---------- main detection -------------------------------------------------
def detect_and_draw(
    frame: MatLike, mtx: npt.NDArray[np.float64], dist_coeffs: npt.NDArray[np.float64],
    table_plane: tuple[np.ndarray, float] | None = None,
    calibration_samples: list | None = None
) -> tuple[MatLike, Pose | None, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(TAG_FAMILY), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return frame, None, np.zeros(3)

    com_positions = []
    com_rotations = []
    for c, id_arr in zip(corners, ids):
        tid = int(id_arr[0])
        offT = trans_offset.get(tid)
        offR = rot_offset.get(tid)
        if offT is None:
            continue
        rvec, tvec, error = estimate_pose([c], MARKER_SIZE, mtx, dist_coeffs)
        
        # Check detection quality - skip if reprojection error is too high
        QUALITY_THRESHOLD = 5.0  # pixels - increase if too many good tags are rejected
        if error[0] > QUALITY_THRESHOLD:
            # Draw rejected tag in red
            cv2.aruco.drawDetectedMarkers(frame, [c], np.array([[id_arr]]), borderColor=(0, 0, 255))
            cv2.putText(frame, f"Err: {error[0]:.1f}px", 
                       tuple(c[0][0].astype(int)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            continue
        
        # Draw accepted tag with quality info
        cv2.putText(frame, f"{error[0]:.1f}px", 
                   tuple(c[0][0].astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # smooth per-tag pose
        tfilt = get_tag_filter(tid)
        t_s, r_s = tfilt.update(tvec[0], rvec[0])

        # tag frame → cube COM
        R_tag, _ = cv2.Rodrigues(r_s)
        # First apply rotation correction to get cube orientation
        com_rot = R_tag @ offR
        # Translation: tag position + rotation-corrected offset
        # offT is defined in tag frame, but we need to apply rotation first
        # The tag is at distance cube_half from center along cube Z-axis
        # So in cube frame, each tag is at a different position
        # We need: cube_center = tag_pos + R_cube @ (-tag_pos_in_cube_frame)
        com_pos = t_s + R_tag @ offT  # Transform offset from tag frame to camera frame
        com_positions.append(com_pos)
        com_rotations.append(com_rot)

        # draw tag axes
        _ = cv2.drawFrameAxes(frame, mtx, dist_coeffs, r_s, t_s, MARKER_SIZE * 0.4)

    # fuse COM if at least one face visible
    if com_positions:
        pos_avg = np.mean(np.vstack(com_positions), axis=0)
        rot_avg = average_rot(com_rotations)
        pos_sm, rvec_sm = cube_filter.update(pos_avg, cv2.Rodrigues(rot_avg)[0])

        # Draw cube coordinate frame in camera frame (standard)
        _ = cv2.drawFrameAxes(
            frame, mtx, dist_coeffs, rvec_sm, pos_sm, MARKER_SIZE * 0.6
        )
        
        # Calculate height above table plane
        # If we have a calibrated table plane, calculate distance to it
        if table_plane is not None:
            plane_normal, plane_d = table_plane
            # Distance from point to plane: n·p + d
            # Normal points toward camera (negative Y), so as cube lifts (Y decreases), distance increases
            height_above_table = np.dot(plane_normal, pos_sm) + plane_d
        else:
            height_above_table = 0.0
        
        # Transform camera position to world frame using CAMERA_ROTATION
        pos_world_full = CAMERA_ROTATION @ pos_sm
        
        # Replace Z with calibrated height from table plane
        pos_world = np.array([pos_world_full[0], pos_world_full[1], height_above_table])
        
        # Calculate rotation around world Z-axis (yaw/heading)
        # Transform cube rotation to world frame
        R_cube_cam, _ = cv2.Rodrigues(rvec_sm)
        R_cube_world = CAMERA_ROTATION @ R_cube_cam
        
        # Extract yaw angle (rotation around Z-axis)
        # Use atan2 of rotation matrix elements to get Z-axis rotation
        yaw_rad = np.arctan2(R_cube_world[1, 0], R_cube_world[0, 0])
        yaw_deg = np.degrees(yaw_rad)
        
        # Draw world frame axes using CAMERA_ROTATION
        axis_length = MARKER_SIZE * 1.2
        
        # Define world axes
        world_x = np.array([axis_length, 0, 0])
        world_y = np.array([0, axis_length, 0])
        world_z = np.array([0, 0, axis_length])
        
        # Transform back to camera frame for visualization
        cam_x = CAMERA_ROTATION.T @ world_x
        cam_y = CAMERA_ROTATION.T @ world_y
        cam_z = CAMERA_ROTATION.T @ world_z
        
        # World axes endpoints in camera frame
        x_end = pos_sm + cam_x
        y_end = pos_sm + cam_y
        z_end = pos_sm + cam_z
        
        points_3d = np.array([x_end.reshape(3), y_end.reshape(3), z_end.reshape(3)], dtype=np.float32)
        points_2d, _ = cv2.projectPoints(points_3d, np.zeros(3), np.zeros(3), mtx, dist_coeffs)
        
        # Draw world frame axes (thicker, labeled)
        center_2d, _ = cv2.projectPoints(pos_sm.reshape(1, 3), np.zeros(3), np.zeros(3), mtx, dist_coeffs)
        center_pt = tuple(center_2d[0].ravel().astype(int))
        
        # World X (right) - red
        cv2.arrowedLine(frame, center_pt, tuple(points_2d[0].ravel().astype(int)), 
                       (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(frame, "X(world)", tuple(points_2d[0].ravel().astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # World Y (forward) - green
        cv2.arrowedLine(frame, center_pt, tuple(points_2d[1].ravel().astype(int)), 
                       (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(frame, "Y(world)", tuple(points_2d[1].ravel().astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # World Z (up) - blue
        cv2.arrowedLine(frame, center_pt, tuple(points_2d[2].ravel().astype(int)), 
                       (255, 0, 0), 3, tipLength=0.2)
        cv2.putText(frame, "Z(up)", tuple(points_2d[2].ravel().astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display position in mm with white background
        y_text = 30
        # Draw white background boxes for better contrast
        box_height = 170 if table_plane is None else 145
        cv2.rectangle(frame, (5, 5), (380, box_height), (255, 255, 255), -1)
        
        cv2.putText(frame, "World Position:", (10, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_text += 30
        cv2.putText(frame, f"X: {pos_world[0]*1000:6.1f} mm (right+)", (10, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_text += 25
        cv2.putText(frame, f"Y: {pos_world[1]*1000:6.1f} mm (forward+)", (10, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y_text += 25
        
        # Show calibration status
        if table_plane is None:
            cv2.putText(frame, f"Z: {height_above_table*1000:6.1f} mm (UNCALIBRATED)", (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            y_text += 25
            cv2.putText(frame, "Move cube on table, press 'c'", (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
            y_text += 20
            n_samples = len(calibration_samples) if calibration_samples is not None else 0
            cv2.putText(frame, f"(samples: {n_samples})", (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1)
        else:
            cv2.putText(frame, f"Z: {pos_world[2]*1000:6.1f} mm (height)", (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_text += 25
            cv2.putText(frame, f"Yaw: {yaw_deg:6.1f}\u00b0 (Z-axis rotation)", (10, y_text),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        _ = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return frame, Pose(pos_world, R.from_rotvec(rvec_sm.reshape(3)).as_quat(), yaw_rad), pos_sm.copy()
    else:
        _ = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return frame, None, np.zeros(3)


# ---------- live loop ------------------------------------------------------
def vision_loop(q: Queue | None, calib="camera.npz"):
    try:
        data = np.load(calib)
        K, D = data["mtx"], data["dist"]
    except FileNotFoundError:
        print(f"❌ Camera calibration file '{calib}' not found!")
        print("   Run: python calibrate_camera.py")
        sys.exit(1)
    
    # Table plane calibration: collect multiple points to define the table plane
    # Plane equation: ax + by + cz + d = 0, or n·p + d = 0 where n is normal
    table_plane = None  # Will be (normal_vector, d_offset)
    calibration_samples = []  # Collect camera-space positions on table
    
    # Try to load existing table plane calibration
    table_plane_file = "table_plane.npz"
    try:
        data = np.load(table_plane_file)
        table_plane = (data['normal'], float(data['d']))
        print(f"✓ Loaded table plane from {table_plane_file}")
    except FileNotFoundError:
        print(f"No table plane calibration found. Move cube and press 'c' to calibrate.")
    
    # Open camera
    cap = cv2.VideoCapture(CAM_ID)
    
    if not cap.isOpened():
        sys.exit("No webcam")

    prev_pos: npt.NDArray[np.float64] | None = None
    initial_pos: npt.NDArray[np.float64] | None = None
    while True:
        _, frame = cap.read()
        
        # Single camera detection
        (frame, pose, pos_cam) = detect_and_draw(frame, K, D, table_plane, calibration_samples)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and pose is not None:
            # Collect calibration sample - use pos_sm (camera position) not pos_cam
            calibration_samples.append(pos_cam.copy())
            print(f"✓ Sample {len(calibration_samples)} at cam pos: [{pos_cam[0]:.3f}, {pos_cam[1]:.3f}, {pos_cam[2]:.3f}]")
            
            # After 5+ samples, fit a plane
            if len(calibration_samples) >= 5:
                # Fit plane: find normal vector using SVD
                points = np.array(calibration_samples)
                centroid = points.mean(axis=0)
                centered = points - centroid
                
                # Print variance in each direction to debug
                variances = np.var(centered, axis=0)
                print(f"  Variances: X={variances[0]:.6f}, Y={variances[1]:.6f}, Z={variances[2]:.6f}")
                
                _, s, Vt = np.linalg.svd(centered)
                print(f"  Singular values: {s}")
                
                normal = Vt[2, :]  # Last row = normal with smallest variance
                
                # Ensure normal points "up" (toward camera, negative Y direction in cam frame)
                # Camera Y points down, so table normal should have negative Y
                if normal[1] > 0:
                    normal = -normal
                
                # Compute d: n·p + d = 0, so d = -n·centroid
                d = -np.dot(normal, centroid)
                
                table_plane = (normal, d)
                print(f"✓ Table plane calibrated with {len(calibration_samples)} samples")
                print(f"  Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")
                print(f"  d offset: {d:.3f}")
                
                # Save to file
                np.savez(table_plane_file, normal=normal, d=d)
                print(f"✓ Saved to {table_plane_file}")
                
                # Automatically calibrate camera orientation from table plane
                up_cam = -normal / np.linalg.norm(normal)  # Negate because normal points toward camera
                
                # Choose forward direction (Y in world) as projection of camera Z onto table
                forward_cam = np.array([0.0, 0.0, 1.0])  # Camera looks along +Z
                forward_cam_on_table = forward_cam - np.dot(forward_cam, up_cam) * up_cam
                forward_cam_on_table = forward_cam_on_table / np.linalg.norm(forward_cam_on_table)
                
                # Right direction (X) is perpendicular to both
                right_cam = np.cross(forward_cam_on_table, up_cam)
                right_cam = right_cam / np.linalg.norm(right_cam)
                
                # Build rotation matrix: columns are world axes expressed in camera frame
                global CAMERA_ROTATION
                CAMERA_ROTATION = np.column_stack([right_cam, forward_cam_on_table, up_cam])
                
                # Save camera rotation to file
                np.savez('camera_rotation.npz', rotation_matrix=CAMERA_ROTATION)
                print("✓ Camera orientation auto-calibrated from table plane")
                euler = R.from_matrix(CAMERA_ROTATION).as_euler('xyz', degrees=True)
                print(f"  Rotation: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]°")
                
                calibration_samples.clear()
        
        elif key == ord('r'):
            # Reset calibration
            table_plane = None
            calibration_samples.clear()
            print("✗ Calibration reset")
            calibration_samples.clear()
            print("✗ Calibration reset")
        elif key in (27, ord("q")):
            break
        
        if q is not None:
            if pose is not None:
                if initial_pos is None:
                    initial_pos = pose.pos.copy()
                
                if prev_pos is not None:
                    # ----------------  Δ translation  -------------------------
                    # pose.pos is already in world frame (pos_world)
                    d_pos_world = pose.pos - prev_pos  # Already in world frame!
                    
                    # MuJoCo coordinates (from scene.xml): X=right/left, Y=forward/back, Z=up/down
                    # World coordinates: X=right, Y=forward, Z=up
                    # Direct mapping - they're the same!
                    d_pos_mujoco = d_pos_world
                    
                    # Print delta from initial position (total movement)
                    total_delta_world = pose.pos - initial_pos  # Already in world frame!
                    
                    # ----------------  send newest only  ----------------------
                    try:
                        q.get_nowait()  # drop stale message
                    except Empty:
                        pass
                    q.put(Pose(d_pos_mujoco, quat=pose.quat, yaw=pose.yaw))

        if pose is not None:
            prev_pos = pose.pos  # update history
        try:
            cv2.imshow("AprilTag COM demo", frame)
        except cv2.error:
            # Skip display if no GUI available
            time.sleep(0.033)  # ~30 FPS equivalent
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    vision_loop(None)
