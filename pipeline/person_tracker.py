"""Person re-identification and tracking using YOLOv8-pose + ResNet18 embeddings.

State machine:
  SCANNING  — No target. Looking for someone raising their hand.
  TRACKING  — Target enrolled. Following them by re-ID matching.
  LOST      — Target not seen recently. Searching via re-ID. Timeout → SCANNING.
"""

import time
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

# COCO keypoint indices
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10

STATES = ("scanning", "tracking", "lost")


class PersonTracker:
    def __init__(self, device="cuda", reid_threshold=0.7, lost_timeout=5.0,
                 hand_raise_margin=0.15):
        """
        Args:
            device: 'cuda' or 'cpu'
            reid_threshold: cosine similarity threshold for re-ID match
            lost_timeout: seconds before LOST → SCANNING
            hand_raise_margin: wrist must be this fraction of bbox height above shoulder
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.reid_threshold = reid_threshold
        self.lost_timeout = lost_timeout
        self.hand_raise_margin = hand_raise_margin

        self._yolo = None
        self._reid_model = None
        self._reid_transform = None
        self._lock = threading.Lock()

        # State
        self.state = "scanning"
        self._target_embedding = None
        self._last_seen_time = 0.0
        self._lost_since = 0.0

    def load_models(self):
        """Load YOLOv8n-pose and ResNet18 feature extractor."""
        from ultralytics import YOLO

        print("[tracker] Loading YOLOv8n-pose...")
        self._yolo = YOLO("yolov8n-pose.pt")
        print("[tracker] YOLOv8n-pose loaded.")

        print("[tracker] Loading ResNet18 re-ID backbone...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove classification head — keep up to avgpool → 512-d output
        self._reid_model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self._reid_model.eval().to(self.device)
        print("[tracker] ResNet18 re-ID backbone loaded.")

        self._reid_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # Standard re-ID aspect ratio
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def process_frame(self, frame):
        """Run full pipeline on one frame.

        Returns:
            dict with keys:
                state: 'scanning' | 'tracking' | 'lost'
                target_bbox: (x, y, w, h) or None
                target_center: (cx, cy) or None
                all_persons: list of person dicts
                hand_raised_idx: index into all_persons of hand-raiser, or None
        """
        with self._lock:
            persons = self._detect_persons(frame)

            result = {
                "state": self.state,
                "target_bbox": None,
                "target_center": None,
                "all_persons": persons,
                "hand_raised_idx": None,
            }

            if not persons:
                if self.state == "tracking":
                    self.state = "lost"
                    self._lost_since = time.time()
                if self.state == "lost" and (time.time() - self._lost_since) > self.lost_timeout:
                    self._reset_target()
                result["state"] = self.state
                return result

            # Extract embeddings for all detected persons
            embeddings = []
            for p in persons:
                emb = self._extract_embedding(frame, p["bbox"])
                embeddings.append(emb)
                p["embedding"] = emb

            # Check for hand raises
            for i, p in enumerate(persons):
                if self._check_hand_raised(p["keypoints"], p["bbox"]):
                    result["hand_raised_idx"] = i

            if self.state == "scanning":
                # Enroll whoever raises their hand
                if result["hand_raised_idx"] is not None:
                    idx = result["hand_raised_idx"]
                    self._enroll_target(embeddings[idx])
                    self.state = "tracking"
                    print("[tracker] Target enrolled via hand raise!")

            if self.state in ("tracking", "lost"):
                # Find best re-ID match
                best_idx, best_score = self._find_target(embeddings)

                if best_idx is not None and best_score >= self.reid_threshold:
                    p = persons[best_idx]
                    x, y, w, h = p["bbox"]
                    result["target_bbox"] = p["bbox"]
                    result["target_center"] = (x + w / 2, y + h / 2)
                    self._update_target_embedding(embeddings[best_idx])
                    self._last_seen_time = time.time()
                    self.state = "tracking"
                else:
                    # Target not matched
                    if self.state == "tracking":
                        self.state = "lost"
                        self._lost_since = time.time()
                    elif self.state == "lost":
                        if (time.time() - self._lost_since) > self.lost_timeout:
                            self._reset_target()

            result["state"] = self.state
            return result

    def _detect_persons(self, frame):
        """Run YOLOv8-pose, return list of person dicts."""
        results = self._yolo(frame, conf=0.5, verbose=False, classes=[0])  # class 0 = person

        persons = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            boxes = r.boxes.xywh.cpu().numpy()  # (cx, cy, w, h)
            confs = r.boxes.conf.cpu().numpy()
            keypoints = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

            for i in range(len(boxes)):
                cx, cy, w, h = boxes[i]
                # Convert center-format to top-left format
                x = cx - w / 2
                y = cy - h / 2
                kps = keypoints[i] if keypoints is not None else None
                persons.append({
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": float(confs[i]),
                    "keypoints": kps,
                })

        return persons

    def _check_hand_raised(self, keypoints, bbox):
        """Check if either wrist is above its corresponding shoulder by a margin."""
        if keypoints is None:
            return False

        _, _, _, bh = bbox
        margin = bh * self.hand_raise_margin

        for wrist_idx, shoulder_idx in [(LEFT_WRIST, LEFT_SHOULDER),
                                         (RIGHT_WRIST, RIGHT_SHOULDER)]:
            wrist = keypoints[wrist_idx]
            shoulder = keypoints[shoulder_idx]
            # Skip if keypoints not detected (0,0)
            if wrist[0] == 0 and wrist[1] == 0:
                continue
            if shoulder[0] == 0 and shoulder[1] == 0:
                continue
            # y increases downward, so wrist.y < shoulder.y means hand is above shoulder
            if wrist[1] < shoulder[1] - margin:
                return True

        return False

    def _extract_embedding(self, frame, bbox):
        """Crop person from frame and extract 512-d L2-normalized embedding."""
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        # Clamp to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return torch.zeros(512, device=self.device)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._reid_transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self._reid_model(tensor)  # (1, 512, 1, 1)

        feat = feat.flatten()
        feat = F.normalize(feat, dim=0)
        return feat

    def _enroll_target(self, embedding):
        """Store the target's appearance embedding."""
        self._target_embedding = embedding.clone()

    def _find_target(self, embeddings):
        """Find the best match to the enrolled target. Returns (index, score) or (None, 0)."""
        if self._target_embedding is None:
            return None, 0.0

        best_idx = None
        best_score = -1.0

        for i, emb in enumerate(embeddings):
            score = torch.dot(self._target_embedding, emb).item()
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx, best_score

    def _update_target_embedding(self, embedding, alpha=0.1):
        """EMA update of target embedding for drift tolerance."""
        if self._target_embedding is None:
            self._target_embedding = embedding.clone()
            return
        self._target_embedding = (1 - alpha) * self._target_embedding + alpha * embedding
        self._target_embedding = F.normalize(self._target_embedding, dim=0)

    def _reset_target(self):
        """Clear target, return to SCANNING."""
        self._target_embedding = None
        self.state = "scanning"
        print("[tracker] Target lost — returning to scanning.")

    def reset(self):
        """Public reset: clear target and go back to scanning."""
        with self._lock:
            self._reset_target()
