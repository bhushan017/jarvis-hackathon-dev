"""Quick test of Reachy Mini: antennas, head nod, body turn."""

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
import time

with ReachyMini() as mini:
    print("Connected to Reachy Mini!")

    # Wiggle antennas
    print("Wiggling antennas...")
    mini.goto_target(antennas=np.deg2rad([40, 40]), duration=0.5)
    time.sleep(0.6)
    mini.goto_target(antennas=np.deg2rad([0, 0]), duration=0.5)
    time.sleep(0.6)

    # Nod head
    print("Nodding head...")
    mini.goto_target(head=create_head_pose(pitch=-15), duration=0.5)
    time.sleep(0.6)
    mini.goto_target(head=create_head_pose(pitch=0), duration=0.5)
    time.sleep(0.6)

    # Turn body left then right
    print("Turning body...")
    mini.goto_target(body_yaw=np.deg2rad(20), duration=0.5)
    time.sleep(0.6)
    mini.goto_target(body_yaw=np.deg2rad(-20), duration=0.5)
    time.sleep(0.6)
    mini.goto_target(body_yaw=np.deg2rad(0), duration=0.5)
    time.sleep(0.6)

    print("Done!")
