"""Bridge between Agora voice pipeline and Reachy Mini robot.

Handles:
- Connecting to the robot
- Head wobble during TTS speech (audio-reactive)
- Emotion/movement commands from the LLM
"""

import base64
import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# Emotion name → (antennas_deg, head_pitch_deg, duration)
EMOTION_MAP = {
    "excited": ([45, 45], 5, 0.4),
    "curious": ([20, -20], -10, 0.5),
    "calm": ([0, 0], 0, 0.8),
    "surprised": ([50, 50], 10, 0.3),
    "amused": ([30, 30], 5, 0.4),
    "skeptical": ([-10, 20], -5, 0.5),
    "happy": ([40, 40], 5, 0.4),
    "sad": ([-20, -20], -15, 0.6),
}


class ReachyBridge:
    def __init__(self):
        self._robot = None
        self._connected = False
        self._wobble_thread = None
        self._wobble_running = False
        self._current_level = 0.0
        self._speaking = False
        self._last_audio_time = 0.0
        self._lock = threading.Lock()

    @property
    def connected(self):
        return self._connected

    def connect(self):
        """Connect to Reachy Mini."""
        try:
            from reachy_mini import ReachyMini
            self._robot = ReachyMini()
            self._robot.__enter__()
            self._robot.enable_motors()
            self._connected = True
            logger.info("Connected to Reachy Mini (motors enabled)")

            # Start wobble thread
            self._wobble_running = True
            self._wobble_thread = threading.Thread(target=self._wobble_loop, daemon=True)
            self._wobble_thread.start()

            return True
        except Exception as e:
            logger.error("Failed to connect to Reachy Mini: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from Reachy Mini."""
        self._wobble_running = False
        if self._wobble_thread:
            self._wobble_thread.join(timeout=2)
        if self._robot:
            try:
                self._robot.__exit__(None, None, None)
            except Exception:
                pass
            self._robot = None
        self._connected = False
        logger.info("Disconnected from Reachy Mini")

    def feed_audio_chunk(self, level: float):
        """Feed audio energy level (0-1) for head wobble."""
        with self._lock:
            self._current_level = min(max(float(level), 0.0), 1.0)
            self._last_audio_time = time.time()

    def set_speaking(self, speaking: bool):
        """Set whether the agent is currently speaking."""
        with self._lock:
            self._speaking = speaking

    def play_emotion(self, emotion: str):
        """Play an emotion on the robot."""
        if not self._connected or not self._robot:
            return

        emotion = emotion.lower().strip()
        if emotion in EMOTION_MAP:
            antennas_deg, pitch_deg, duration = EMOTION_MAP[emotion]
            try:
                from reachy_mini.utils import create_head_pose
                self._robot.goto_target(
                    head=create_head_pose(pitch=pitch_deg),
                    antennas=np.deg2rad(antennas_deg),
                    duration=duration,
                    method="minjerk",
                )
                logger.info("Emotion: %s", emotion)
            except Exception as e:
                logger.error("Emotion error: %s", e)
        else:
            logger.warning("Unknown emotion: %s", emotion)

    def move_head(self, direction: str):
        """Move head in a direction."""
        if not self._connected or not self._robot:
            return

        from reachy_mini.utils import create_head_pose
        moves = {
            "left": create_head_pose(yaw=20),
            "right": create_head_pose(yaw=-20),
            "up": create_head_pose(pitch=15),
            "down": create_head_pose(pitch=-15),
            "front": create_head_pose(pitch=0, yaw=0),
            "nod": None,
        }

        try:
            if direction == "nod":
                self._robot.goto_target(head=create_head_pose(pitch=-10), duration=0.3)
                time.sleep(0.35)
                self._robot.goto_target(head=create_head_pose(pitch=5), duration=0.3)
                time.sleep(0.35)
                self._robot.goto_target(head=create_head_pose(pitch=0), duration=0.3)
            elif direction in moves:
                self._robot.goto_target(head=moves[direction], duration=0.5)
            logger.info("Head move: %s", direction)
        except Exception as e:
            logger.error("Head move error: %s", e)

    def wiggle_antennas(self):
        """Quick antenna wiggle."""
        if not self._connected or not self._robot:
            return
        try:
            self._robot.goto_target(antennas=np.deg2rad([40, 40]), duration=0.3)
            time.sleep(0.35)
            self._robot.goto_target(antennas=np.deg2rad([0, 0]), duration=0.3)
        except Exception:
            pass

    def _wobble_loop(self):
        """Background thread: subtle head pitch wobble while speaking."""
        logger.info("Wobble loop started")
        from reachy_mini.utils import create_head_pose
        last_log = 0
        while self._wobble_running:
            with self._lock:
                level = self._current_level
                audio_age = time.time() - self._last_audio_time

            # Wobble if we received audio in the last 0.5 seconds
            if audio_age < 0.5 and level > 0.02 and self._connected and self._robot:
                pitch = np.sin(time.time() * 6) * level * 25  # ±25 degrees max wobble
                try:
                    self._robot.set_target(head=create_head_pose(pitch=pitch))
                    now = time.time()
                    if now - last_log > 2:
                        logger.info("Wobbling: pitch=%.1f level=%.3f", pitch, level)
                        last_log = now
                except Exception as e:
                    logger.error("Wobble error: %s", e)
                time.sleep(0.03)
            else:
                time.sleep(0.05)

        logger.info("Wobble loop stopped")
