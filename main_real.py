#!/usr/bin/env python3
"""
Control real SO-ARM100 robot using AprilTag cube detection.
Similar to main.py but for real robot instead of MuJoCo simulation.
"""

import sys
import time
import threading
import queue
from multiprocessing import Process, Queue

import numpy as np
import draccus
import mujoco
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

from april import vision_loop


# Joint ranges from so_arm100.xml (in radians)
JOINT_RANGES = {
    'Rotation': (-2.2, 2.2),              # shoulder_pan
    'Pitch': (-3.14158, 0.2),             # shoulder_lift
    'Elbow': (0, 3.14158),                # elbow_flex
    'Wrist_Pitch': (-2.0, 1.8),           # wrist_flex
    'Wrist_Roll': (-3.14158, 3.14158),    # wrist_roll
    'Jaw': (-0.2, 2.0),                   # gripper
}


def rad_to_percent(rad_value, joint_name):
    """Convert joint angle from radians to percentage.
    Most joints: -100% to +100%
    Jaw (gripper): 0% to 100%
    """
    min_rad, max_rad = JOINT_RANGES[joint_name]
    
    if joint_name == 'Jaw':
        # Map [min_rad, max_rad] -> [0, +100]
        percent = ((rad_value - min_rad) / (max_rad - min_rad)) * 100.0
        return np.clip(percent, 0.0, 100.0)
    else:
        # Map [min_rad, max_rad] -> [-100, +100]
        range_rad = max_rad - min_rad
        center_rad = (max_rad + min_rad) / 2.0
        percent = ((rad_value - center_rad) / (range_rad / 2.0)) * 100.0
        return np.clip(percent, -100.0, 100.0)


def percent_to_rad(percent_value, joint_name):
    """Convert joint angle from percentage to radians.
    Most joints: -100% to +100%
    Jaw (gripper): 0% to 100%
    """
    min_rad, max_rad = JOINT_RANGES[joint_name]
    
    if joint_name == 'Jaw':
        # Map [0, +100] -> [min_rad, max_rad]
        rad = min_rad + (percent_value / 100.0) * (max_rad - min_rad)
        return np.clip(rad, min_rad, max_rad)
    else:
        # Map [-100, +100] -> [min_rad, max_rad]
        range_rad = max_rad - min_rad
        center_rad = (max_rad + min_rad) / 2.0
        rad = center_rad + (percent_value / 100.0) * (range_rad / 2.0)
        return np.clip(rad, min_rad, max_rad)


def robot_control_loop(pose_queue, robot_port="/dev/tty_pink_follower_so101", robot_id="pink"):
    """
    Read poses from queue and control real robot.
    Similar to mujoco_loop.py but for real hardware.
    """
    print("🤖 Starting robot control...")
    
    # Connect to robot using LeRobot
    try:
        old_argv = sys.argv
        try:
            sys.argv = [
                'main_real',
                '--robot.type=so101_follower',
                f'--robot.port={robot_port}',
                f'--robot.id={robot_id}',
                '--teleop.type=so101_leader',
                '--teleop.port=/dev/null',
            ]
            cfg = draccus.parse(TeleoperateConfig)
        finally:
            sys.argv = old_argv
        
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        print("✓ Robot connected")
    except Exception as e:
        print(f"❌ Failed to connect to robot: {e}")
        sys.exit(1)
    
    # Load MuJoCo model for IK computation
    print("📦 Loading MuJoCo model for IK...")
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # IK parameters (same as mujoco_loop.py)
    integration_dt = 1.0
    damping = 1e-4
    max_angvel = 0.0
    
    # Get IDs from MuJoCo model
    end_effector = model.site("attachment_site").id
    mocap_id = model.body("target").mocapid[0]
    
    ik_joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch"]
    ik_dof_ids = np.array([model.joint(name).id for name in ik_joint_names])
    
    wrist_roll_dof_id = model.joint("Wrist_Roll").id
    jaw_dof_id = model.joint("Jaw").id
    
    # Pre-allocate arrays
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    error = np.zeros(6)
    error_pos = error[:3]
    error_ori = error[3:]
    ee_quat = np.zeros(4)
    ee_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    
    # Reset to home position
    key_id = model.key("home-scene").id
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    
    print(f"✓ MuJoCo model loaded")
    
    # Save home position as reference for tracking
    initial_robot_pos = data.site(end_effector).xpos.copy()
    print(f"✓ Home position: [{initial_robot_pos[0]:.3f}, {initial_robot_pos[1]:.3f}, {initial_robot_pos[2]:.3f}]")
    
    # Control state
    wrist_roll_target = data.qpos[wrist_roll_dof_id]
    gripper_target = data.qpos[jaw_dof_id]
    
    # Initial reference position from AprilTag
    initial_pose = None
    
    # Get initial position from queue
    latest_pose = None
    
    print("⏳ Waiting for initial AprilTag detection...")
    while initial_pose is None:
        if not pose_queue.empty():
            try:
                initial_pose = pose_queue.get_nowait()
                print(f"✓ Initial pose captured at [{initial_pose.pos[0]:.3f}, {initial_pose.pos[1]:.3f}, {initial_pose.pos[2]:.3f}]")
            except:
                pass
        time.sleep(0.1)
    
    print("\n📋 Controls:")
    print("  9/0 = Open/close gripper")
    print("  The robot follows the AprilTag cube position")
    print()
    
    # Control loop
    dt = 1.0 / 60.0  # 60 Hz
    
    try:
        while True:
            loop_start = time.time()
            
            # Get latest pose from queue (non-blocking)
            while not pose_queue.empty():
                try:
                    latest_pose = pose_queue.get_nowait()
                    print(f"🔍 RAW from queue - pos type: {type(latest_pose.pos)}, shape: {latest_pose.pos.shape if hasattr(latest_pose.pos, 'shape') else 'N/A'}, values: {latest_pose.pos}")
                except:
                    break
            
            if latest_pose is not None:
                # Calculate delta
                cube_delta = latest_pose.pos - initial_pose.pos
                
                # Invert X-axis (left/right) to match robot frame
                cube_delta[0] = -cube_delta[0]
                
                # Update mocap target (EXACT same as MuJoCo)
                data.mocap_pos[mocap_id] = initial_robot_pos + cube_delta
                
                # Update wrist roll from cube yaw
                if latest_pose.yaw is not None:
                    wrist_roll_target = latest_pose.yaw
                
                # Print positions
                print(f"\n📍 Cube pos:      [{latest_pose.pos[0]:+.3f}, {latest_pose.pos[1]:+.3f}, {latest_pose.pos[2]:+.3f}] m")
                print(f"📍 Initial cube:  [{initial_pose.pos[0]:+.3f}, {initial_pose.pos[1]:+.3f}, {initial_pose.pos[2]:+.3f}] m")
                print(f"📊 Cube delta:    [{cube_delta[0]:+.4f}, {cube_delta[1]:+.4f}, {cube_delta[2]:+.4f}] m")
                print(f"🏠 Initial robot: [{initial_robot_pos[0]:+.3f}, {initial_robot_pos[1]:+.3f}, {initial_robot_pos[2]:+.3f}] m")
                print(f"🎯 Target pos:    [{data.mocap_pos[mocap_id][0]:+.3f}, {data.mocap_pos[mocap_id][1]:+.3f}, {data.mocap_pos[mocap_id][2]:+.3f}] m")
                print(f"🤖 Robot pos:     [{data.site(end_effector).xpos[0]:+.3f}, {data.site(end_effector).xpos[1]:+.3f}, {data.site(end_effector).xpos[2]:+.3f}] m")
                
                # EXACT MuJoCo IK computation (using MuJoCo functions!)
                error_pos[:] = data.mocap_pos[mocap_id] - data.site(end_effector).xpos
                print(f"📏 Error:         [{error_pos[0]:+.4f}, {error_pos[1]:+.4f}, {error_pos[2]:+.4f}] m")
                
                mujoco.mju_mat2Quat(ee_quat, data.site(end_effector).xmat)
                mujoco.mju_negQuat(ee_quat_conj, ee_quat)
                mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], ee_quat_conj)
                mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)
                
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], end_effector)
                
                # Solve system of equations: J @ dq = error
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)
                
                # Zero out wrist roll and jaw
                dq[wrist_roll_dof_id] = 0.0
                dq[jaw_dof_id] = 0.0
                
                # Scale down if needed
                if max_angvel > 0:
                    dq_abs_max = np.abs(dq).max()
                    if dq_abs_max > max_angvel:
                        dq *= max_angvel / dq_abs_max
                
                # Integrate joint positions using MuJoCo!
                q = data.qpos.copy()
                mujoco.mj_integratePos(model, q, dq, integration_dt)
                
                # Update MuJoCo data for next iteration
                data.qpos[:] = q
                mujoco.mj_step(model, data)
                
                # Get IK joint angles
                q_ik = q[ik_dof_ids]
                
                # Send to robot
                try:
                    action = robot.get_observation()
                    
                    # Convert from RADIANS to PERCENTAGES using correct joint ranges
                    action['shoulder_pan.pos'] = np.array([rad_to_percent(q_ik[0], 'Rotation')], dtype=np.float32)
                    action['shoulder_lift.pos'] = np.array([rad_to_percent(q_ik[1], 'Pitch')], dtype=np.float32)
                    action['elbow_flex.pos'] = np.array([rad_to_percent(q_ik[2], 'Elbow')], dtype=np.float32)
                    action['wrist_flex.pos'] = np.array([rad_to_percent(q_ik[3], 'Wrist_Pitch')], dtype=np.float32)
                    action['wrist_roll.pos'] = np.array([rad_to_percent(wrist_roll_target, 'Wrist_Roll')], dtype=np.float32)
                    action['gripper.pos'] = np.array([rad_to_percent(gripper_target, 'Jaw')], dtype=np.float32)
                    
                    print(f"⚙️  IK angles (rad): [{q_ik[0]:+.3f}, {q_ik[1]:+.3f}, {q_ik[2]:+.3f}, {q_ik[3]:+.3f}]")
                    print(f"📤 Robot cmd (%):   [{action['shoulder_pan.pos'][0]:+6.1f}, {action['shoulder_lift.pos'][0]:+6.1f}, {action['elbow_flex.pos'][0]:+6.1f}, {action['wrist_flex.pos'][0]:+6.1f}]")
                    
                    robot.send_action(action)
                    
                except Exception as e:
                    print(f"⚠️ Failed to send action: {e}")
            
            # Maintain loop rate
            elapsed = time.time() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping robot control...")
    
    finally:
        # Disconnect robot
        robot.disconnect()
        print("✓ Robot disconnected")


def main():
    print("=" * 70)
    print("Real Robot Control with AprilTag Cube")
    print("=" * 70)
    print()
    
    # Robot configuration
    robot_port = input("Robot port (default: /dev/tty_pink_follower_so101): ").strip() or "/dev/tty_pink_follower_so101"
    robot_id = input("Robot ID (default: pink): ").strip() or "pink"
    print()
    
    # Create queue for poses
    pose_queue = Queue(maxsize=10)
    
    # Start vision process
    print("📷 Starting vision...")
    vision_process = Process(target=vision_loop, args=(pose_queue,), daemon=True)
    vision_process.start()
    
    # Give vision time to initialize
    time.sleep(2)
    
    # Run robot control in main thread
    try:
        robot_control_loop(pose_queue, robot_port, robot_id)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        vision_process.terminate()
        vision_process.join()
    
    print("✓ Done")


if __name__ == "__main__":
    main()
