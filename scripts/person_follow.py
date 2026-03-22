#!/usr/bin/env python3
"""Person follow demo: detect hand raise → enroll → track with robot head.

Two-thread architecture (same pattern as dance_to_music.py):
  1. tracking_loop  — camera + YOLOv8-pose + re-ID  (~15-30 Hz)
  2. control_loop   — robot head movement at 30 Hz

Usage:
  python scripts/person_follow.py --no-robot --debug   # test without robot
  python scripts/person_follow.py --debug               # with robot + overlay
  python scripts/person_follow.py                        # headless with robot
"""

import argparse
import sys
import os
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pipeline.person_tracker import PersonTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOTION_HZ = 30
H_FOV_DEG = 90.0   # ZED horizontal FOV (approximate, after stereo crop)
V_FOV_DEG = 60.0   # ZED vertical FOV (approximate)
HEAD_YAW_MAX = 20.0   # Reachy head yaw limit (degrees)
BODY_YAW_MAX = 45.0   # Reachy body yaw limit (degrees)
PITCH_MAX = 15.0      # Reachy head pitch limit (degrees)
KP = 0.6              # Proportional gain
SMOOTH_ALPHA = 0.4    # EMA smoothing factor
BODY_SMOOTH_ALPHA = 0.2  # Body moves slower than head for stability
CENTER_RETURN_RATE = 0.05  # How fast head drifts back to center when lost


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class FollowState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.target_center = None   # (px_x, px_y) in frame
        self.frame_size = (640, 480)
        self.tracker_state = "scanning"


# ---------------------------------------------------------------------------
# Tracking thread
# ---------------------------------------------------------------------------
def tracking_loop(state, tracker, camera_index=0, debug=False):
    """Camera → YOLOv8-pose → re-ID → update shared state."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[follow] Cannot open camera {camera_index}")
        state.running = False
        return

    print(f"[follow] Camera {camera_index} opened.", flush=True)

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # ZED stereo: left half
        if frame.shape[1] > 2000:
            frame = frame[:, :frame.shape[1] // 2]

        h, w = frame.shape[:2]
        with state.lock:
            state.frame_size = (w, h)

        result = tracker.process_frame(frame)

        with state.lock:
            state.target_center = result["target_center"]
            state.tracker_state = result["state"]

        if debug:
            draw_debug(frame, result)
            cv2.imshow("Person Follow", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                state.running = False
                break
            # Press 'r' to reset target
            if cv2.waitKey(1) & 0xFF == ord('r'):
                tracker.reset()
                print("[follow] Manual reset — scanning for hand raise.")

    cap.release()
    if debug:
        cv2.destroyAllWindows()


def draw_debug(frame, result):
    """Draw bounding boxes, keypoints, and status on the frame."""
    h, w = frame.shape[:2]

    # Draw crosshair at center
    cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (100, 100, 100), 1)
    cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (100, 100, 100), 1)

    for i, p in enumerate(result["all_persons"]):
        x, y, bw, bh = p["bbox"]
        is_target = (result["target_center"] is not None and
                     result["target_bbox"] == p["bbox"])
        is_hand_raised = (result["hand_raised_idx"] == i)

        # Box color: green=target, yellow=hand raised, blue=other
        if is_target:
            color = (0, 255, 0)
            label = "TARGET"
        elif is_hand_raised:
            color = (0, 255, 255)
            label = "HAND RAISED"
        else:
            color = (255, 150, 0)
            label = ""

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        if label:
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # Draw keypoints
        if p["keypoints"] is not None:
            for kp in p["keypoints"]:
                kx, ky = int(kp[0]), int(kp[1])
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (kx, ky), 3, color, -1)

    # Draw target center marker
    if result["target_center"] is not None:
        tx, ty = int(result["target_center"][0]), int(result["target_center"][1])
        cv2.drawMarker(frame, (tx, ty), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    # Status text
    state_colors = {"scanning": (200, 200, 0), "tracking": (0, 255, 0), "lost": (0, 0, 255)}
    state_text = result["state"].upper()
    color = state_colors.get(result["state"], (255, 255, 255))
    cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    if result["state"] == "scanning":
        cv2.putText(frame, "Raise your hand to be tracked!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)


# ---------------------------------------------------------------------------
# Control thread
# ---------------------------------------------------------------------------
def control_loop(mini, state):
    """30Hz head+body movement loop: pixel offset → split between body and head yaw."""
    if mini is not None:
        from reachy_mini.utils import create_head_pose

    curr_head_yaw = 0.0
    curr_body_yaw = 0.0
    curr_pitch = 0.0

    while state.running:
        with state.lock:
            target = state.target_center
            fw, fh = state.frame_size
            tracker_state = state.tracker_state

        if tracker_state == "tracking" and target is not None:
            tx, ty = target
            # Pixel offset from center (normalized to -1..1)
            dx = (tx - fw / 2) / (fw / 2)
            dy = (ty - fh / 2) / (fh / 2)

            # Total yaw needed (from camera FOV)
            # Negative dx → person is left → positive yaw (turn left)
            total_yaw = -dx * (H_FOV_DEG / 2) * KP
            target_pitch = -dy * (V_FOV_DEG / 2) * KP

            # Split yaw between body and head:
            # Body handles the bulk rotation, head does fine adjustment.
            # Body takes everything beyond what the head can handle,
            # plus gradually absorbs head yaw to keep head near center.
            target_body_yaw = curr_body_yaw + total_yaw * 0.3  # Body tracks 30% of error
            target_body_yaw = max(-BODY_YAW_MAX, min(BODY_YAW_MAX, target_body_yaw))

            # Head compensates for the rest (total error minus what body covers)
            target_head_yaw = total_yaw - (target_body_yaw - curr_body_yaw)
            target_head_yaw = max(-HEAD_YAW_MAX, min(HEAD_YAW_MAX, target_head_yaw))

            target_pitch = max(-PITCH_MAX, min(PITCH_MAX, target_pitch))
        else:
            # No target: drift back to center
            target_head_yaw = 0.0
            target_body_yaw = 0.0
            target_pitch = 0.0

        # EMA smoothing — body moves slower for stability
        curr_head_yaw += (target_head_yaw - curr_head_yaw) * SMOOTH_ALPHA
        curr_body_yaw += (target_body_yaw - curr_body_yaw) * BODY_SMOOTH_ALPHA
        curr_pitch += (target_pitch - curr_pitch) * SMOOTH_ALPHA

        # When lost/scanning, drift back slowly
        if tracker_state != "tracking":
            curr_head_yaw += (0.0 - curr_head_yaw) * CENTER_RETURN_RATE
            curr_body_yaw += (0.0 - curr_body_yaw) * CENTER_RETURN_RATE
            curr_pitch += (0.0 - curr_pitch) * CENTER_RETURN_RATE

        # Send to robot
        if mini is not None:
            try:
                mini.set_target(
                    head=create_head_pose(yaw=curr_head_yaw, pitch=curr_pitch)
                )
                mini.set_target(body_yaw=np.deg2rad(curr_body_yaw))
            except Exception:
                pass

        # Print status periodically
        if int(time.time() * 2) % 10 == 0 and tracker_state == "tracking":
            print(f"[control] head_yaw={curr_head_yaw:+.1f}° body_yaw={curr_body_yaw:+.1f}° pitch={curr_pitch:+.1f}°  state={tracker_state}")

        time.sleep(1.0 / MOTION_HZ)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Person Follow — hand raise to enroll, robot tracks")
    parser.add_argument("--no-robot", action="store_true", help="Run without robot connection")
    parser.add_argument("--debug", action="store_true", help="Show debug visualization window")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--reid-threshold", type=float, default=0.7, help="Re-ID cosine similarity threshold")
    parser.add_argument("--lost-timeout", type=float, default=5.0, help="Seconds before lost → scanning")
    args = parser.parse_args()

    print("=" * 50)
    print("  Person Follow — Reachy Mini")
    print("=" * 50)
    print(f"  Robot: {'OFF' if args.no_robot else 'ON'}")
    print(f"  Debug: {'ON' if args.debug else 'OFF'}")
    print(f"  Camera: {args.camera}")
    print(f"  Re-ID threshold: {args.reid_threshold}")
    print("=" * 50)
    print()
    print("Raise your hand to become the target!")
    print("Press 'r' to reset, 'q' to quit (debug mode).")
    print()

    # Load models (before threads start so we see output)
    tracker = PersonTracker(
        reid_threshold=args.reid_threshold,
        lost_timeout=args.lost_timeout,
    )
    tracker.load_models()
    print("[follow] Models loaded. Starting threads...", flush=True)

    # Connect robot
    mini = None
    if not args.no_robot:
        try:
            from reachy_mini import ReachyMini
            mini = ReachyMini()
            mini.__enter__()
            mini.enable_motors()
            print("[follow] Connected to Reachy Mini.")
        except Exception as e:
            print(f"[follow] Robot connection failed: {e}")
            print("[follow] Continuing without robot.")
            mini = None

    state = FollowState()

    # Start threads
    t_track = threading.Thread(
        target=tracking_loop,
        args=(state, tracker, args.camera, args.debug),
        daemon=True, name="tracking",
    )
    t_control = threading.Thread(
        target=control_loop,
        args=(mini, state),
        daemon=True, name="control",
    )

    t_track.start()
    t_control.start()

    try:
        while state.running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[follow] Shutting down...")
        state.running = False

    t_track.join(timeout=2)
    t_control.join(timeout=2)

    # Return head to neutral and disconnect
    if mini is not None:
        try:
            from reachy_mini.utils import create_head_pose
            mini.goto_target(head=create_head_pose(yaw=0, pitch=0), duration=0.5)
            time.sleep(0.6)
            mini.__exit__(None, None, None)
        except Exception:
            pass

    print("[follow] Done.")


if __name__ == "__main__":
    main()
