# AprilArm 🪽 (🏅 Winner of HuggingFace LeRobot Hackathon 2025)
## _By Susheendhar Vijay, Nidhin Ninan, and Mohammed Rashad_
### Zero-cost leader arm for the SO-100

> Why buy expensive motors just to read joint angles?

Cost-effective robotic teleoperation using **AprilTags** with a **3D printed controller** to control simulated and real SO-100 robots. Replaces expensive leader-follower setups, halving the cost for imitation learning.

### Required hardware:
- SO-100 robot follower arm
- 3D printed handheld controller (just a hollow cube with a grip)
- Webcam

## 🎬 Demo (brain rot version)
[`1 Min Demo Video`](https://github.com/user-attachments/assets/e8480acb-b422-40d0-b512-5118c11b4a0e)


## 💰 Cost Comparison

| Traditional Setup | Our AprilTag System |
|------------------|-------------------|
| Leader Arm: $100+ & controller | 3D Printed Controller: $5 |
| Follower Arm: $100+ & controller | 122$ & controller |
| **Total: $230** | **Total: $127** |

## 🔧 System Architecture
**Hardware**: 3D printed controller with AprilTags, webcam, SO-100 robot  
**Software**: AprilTag detection, MuJoCo simulation

## 🛠 Setup

**Prerequisites**: Python 3.10+, USB webcam, printer for checkerboard

1. **Install dependencies**:
   ```bash
   uv sync  # or pip install mujoco opencv-contrib-python scipy lerobot[feetech]
   ```

2. **3D print controller**: Print cube from `/assets/`, attach 46mm AprilTags (DICT_APRILTAG_16h5, IDs 0-5)

3. **Camera calibration**:
   ```bash
   # Generate checkerboard pattern
   python generate_checkerboard.py
   
   # Print checkerboard_9x6.png on A4 paper
   # Measure actual square size after printing
   
   # Run interactive calibration (collect 15-30 images)
   python calibrate_camera.py
   # Move checkerboard to different positions/angles
   # Press SPACE to capture, C to calibrate, Q to quit
   ```

4. **Table plane calibration** (for stable height detection):
   - Run `python main.py`
   - Place cube flat on table
   - Press **'r'** to reset calibration
   - Move cube to 5+ different positions on table
   - Press **'c'** at each position to collect sample
   - After 5 samples, plane is automatically fitted and saved

5. **[Optional] AprilTag offset calibration** (for better cube pose accuracy):
   ```bash
   python calibrate_cube.py
   # Show cube with multiple tags visible
   # Move to different angles (50+ frames auto-captured)
   # Press C to compute offsets, S to save
   ```

6. **Setup mjpython (macOS)**:
   ```bash
   bash mjpy-init.sh
   ```

## 🚀 Usage

**Main application**:
```bash
python main.py
```

**Individual components**:
```bash
python april.py          # AprilTag detection only
python mujoco_loop.py    # MuJoCo simulation only
```

**Controls**:
- **Hand movement**: Move cube controller to control robot end-effector
- **Cube rotation**: Rotate cube to control gripper wrist orientation
- **MuJoCo UI Control panel**: Use slider to open/close gripper (right panel in viewer)
- **'c' key**: Collect table plane calibration sample (in april.py window)
- **'r' key**: Reset table plane calibration
- **ESC/Q**: Exit

**Live feedback**:
- **April window**: Shows world position (X, Y, Z in mm) and table plane status
- **Terminal**: Prints delta positions from start for both April and MuJoCo
- **MuJoCo viewer**: Control panel on right for gripper control

## 🧠 How It Works

**AprilTag Detection**: 
- Multi-face detection using 6 tags on cube
- Pose fusion with quality filtering (reprojection error threshold)
- Temporal smoothing via low-pass filter
- **Table plane calibration**: SVD-based plane fitting from multiple samples for stable height detection
- Coordinate transformation from camera frame to world frame (accounting for 30° camera tilt)

**Robot Control**: 
- Delta-based position updates (relative movements, not absolute)
- Quaternion orientation tracking for gripper wrist
- Inverse kinematics with Jacobian-based control
- Gravity compensation and joint limit enforcement

**Architecture**: 
- Two parallel threads communicate via queue
- AprilTag detection (`april.py`) runs in vision thread
- MuJoCo simulation (`mujoco_loop.py`) runs in physics thread
- Real-time performance with ~30 FPS tracking

## 📁 Files

**Main application**:
- `main.py` - Main application entry point
- `april.py` - AprilTag detection and table plane calibration
- `mujoco_loop.py` - MuJoCo simulation with IK control
- `scene.xml` - MuJoCo scene definition
- `so_arm100.xml` - SO-100 robot model

**Calibration**:
- `generate_checkerboard.py` - Generate checkerboard pattern for printing
- `calibrate_camera.py` - Interactive camera calibration with live preview
- `calibrate.py` - Batch camera calibration from photos (legacy)
- `calibrate_cube.py` - Optimize AprilTag offsets on cube (optional, for better accuracy)
- `calib.npz` - Camera intrinsics (focal length, distortion)
- `table_plane.npz` - Table plane parameters (normal, offset)

**Assets**:
- `assets/` - 3D printable STL files for cube controller
- `photos/` - Calibration images storage

## 🔬 Technical Details

**AprilTags**: 
- Family: DICT_APRILTAG_16h5
- Size: 46mm
- IDs: 0-5 (one per cube face)
- Quality threshold: 5.0px reprojection error

**Coordinate Frames**:
- Camera frame: X=right, Y=down, Z=forward
- World frame: X=right, Y=forward, Z=up (30° camera tilt correction)
- MuJoCo frame: X=right, Y=forward, Z=up (matches world frame)

**Table Plane Calibration**:
- Collects 5+ samples at different positions on table
- Fits plane using SVD (Singular Value Decomposition)
- Ensures normal points toward camera (negative Y in camera frame)
- Calculates height as distance from point to plane
- Provides stable Z-coordinate independent of horizontal position

**Control System**:
- Delta position updates: `d_pos = current_pos - prev_pos`
- Quaternion orientation tracking for 6DOF control
- Low-pass filtering: position alpha=0.95, rotation alpha=0.9
- Target starts at table level (Z=0.12m) for full up/down range  

## 🎓 Applications

- Robotics education
- Research data collection  
- Behavior prototyping
- Remote teleoperation

## 💡 Hackathon Update
We are happy to share that this project was one of the winners of the [HuggingFace LeRobot Hackathon 2025](https://huggingface.co/spaces/LeRobot-worldwide-hackathon/all-winners), where we placed **#24** from over 250+ submissions worldwide **(Top 10%)**, and won a [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) as a result. We thank [HuggingFace](https://huggingface.co/) and [Seeedstudio](https://www.seeedstudio.com/) for this award.

---

**Built with**: Python, OpenCV, MuJoCo, NumPy, SciPy
