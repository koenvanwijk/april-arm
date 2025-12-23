import queue
import threading

import mujoco.viewer

from april import vision_loop
from mujoco_loop import sim_loop

if __name__ == "__main__":
    quueue = queue.Queue()
    pose_q = queue.Queue(maxsize=1)
    threading.Thread(target=vision_loop, args=(pose_q,), daemon=True).start()

    # Gripper control disabled - MuJoCo viewer captures all keyboard input
    # To enable: set enable_gripper_control=True (but it won't work with viewer active)
    sim_loop(pose_q, enable_gripper_control=False)
