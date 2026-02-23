"""
inference.py — Real-time security surveillance inference pipeline.

Combines:
  1. YOLOv8 person detection
  2. DeepSORT person tracking
  3. SlowFast action recognition on accumulated clip buffers
  4. Rule-based loitering/prolonged-stay detection

Usage:
    python inference.py                                    # Webcam
    python inference.py --source video.mp4                 # Video file
    python inference.py --source 0 --checkpoint best.pth   # Webcam + custom model
"""

import os
import sys
import argparse
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    InferenceConfig, ACTION_CLASSES, IDX_TO_CLASS, NUM_CLASSES,
)
from models.slowfast_model import load_model_for_inference
from data.dataset import pack_frames_slowfast
from utils.loitering_detector import LoiteringDetector


# ══════════════════════════════════════════════════════════════════════
#  Alert severity levels and colors
# ══════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    "normal":               {"color": (0, 200, 0),   "severity": 0, "label": "Normal"},
    "vandalism":            {"color": (0, 0, 255),    "severity": 3, "label": "⚠ VANDALISM"},
    "harassment":           {"color": (0, 100, 255),  "severity": 3, "label": "⚠ HARASSMENT"},
    "fighting":             {"color": (0, 0, 255),    "severity": 4, "label": "🚨 FIGHTING"},
    "dangerous_object":     {"color": (0, 50, 255),   "severity": 3, "label": "⚠ DANGEROUS OBJECT"},
    "faint":                {"color": (255, 0, 255),  "severity": 4, "label": "🚨 PERSON DOWN"},
    "suspicious_gathering": {"color": (0, 165, 255),  "severity": 2, "label": "⚠ SUSPICIOUS GATHERING"},
    "loitering":            {"color": (0, 255, 255),  "severity": 1, "label": "⚠ LOITERING"},
}


class SecurityInferencePipeline:
    """
    End-to-end real-time inference pipeline for security monitoring.
    """

    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() else "cpu"

        # ── Load the SlowFast action recognition model ──
        print("Loading SlowFast model...")
        if os.path.isfile(cfg.checkpoint_path):
            self.action_model = load_model_for_inference(
                cfg.checkpoint_path,
                num_classes=NUM_CLASSES,
                device=self.device,
            )
        else:
            print(f"⚠ Checkpoint not found: {cfg.checkpoint_path}")
            print("  Running without action recognition (detection + tracking only)")
            self.action_model = None

        # ── Load YOLOv8 for person detection ──
        print("Loading YOLOv8 for person detection...")
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(cfg.yolo_model)
        except ImportError:
            print("⚠ ultralytics not installed. Install with: pip install ultralytics")
            self.yolo = None

        # ── Initialize DeepSORT tracker ──
        print("Initializing DeepSORT tracker...")
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            self.tracker = DeepSort(
                max_age=30,
                n_init=3,
                max_cosine_distance=0.3,
                nn_budget=100,
            )
        except ImportError:
            print("⚠ deep-sort-realtime not installed. Install with: pip install deep-sort-realtime")
            self.tracker = None

        # ── Loitering detector ──
        self.loitering_detector = LoiteringDetector(
            time_threshold=cfg.loiter_time_threshold_sec,
            zone_polygon=cfg.loiter_zone,
        )

        # ── Frame buffer for action recognition ──
        # We accumulate frames and run action recognition every N frames
        self.frame_buffer = deque(maxlen=64)
        self.clip_interval_frames = 15   # Run action recognition every 15 frames
        self.frame_count = 0

        # ── Current predictions ──
        self.current_action = "normal"
        self.current_confidence = 0.0
        self.alert_history = deque(maxlen=100)

        print(f"✓ Pipeline initialized (device={self.device})")

    def detect_persons(self, frame: np.ndarray) -> list:
        """Run YOLOv8 person detection. Returns list of [x1,y1,x2,y2,conf]."""
        if self.yolo is None:
            return []

        results = self.yolo(frame, classes=[0], conf=self.cfg.person_conf_threshold, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf])

        return detections

    def track_persons(self, detections: list, frame: np.ndarray) -> list:
        """Run DeepSORT tracking on detections. Returns track objects."""
        if self.tracker is None or not detections:
            return []

        # DeepSORT expects detections as [[x1,y1,w,h], conf, class]
        dsort_dets = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w, h = x2 - x1, y2 - y1
            dsort_dets.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(dsort_dets, frame=frame)
        return [t for t in tracks if t.is_confirmed()]

    def recognize_action(self, frame: np.ndarray) -> tuple:
        """
        Add frame to buffer and periodically run SlowFast action recognition.
        Returns (action_label, confidence).
        """
        self.frame_buffer.append(frame.copy())
        self.frame_count += 1

        if (self.action_model is None or
                self.frame_count % self.clip_interval_frames != 0 or
                len(self.frame_buffer) < 32):
            return self.current_action, self.current_confidence

        # Build clip from buffer
        frames_list = list(self.frame_buffer)
        indices = np.linspace(0, len(frames_list) - 1, 64, dtype=np.int64)
        clip_frames = np.stack([frames_list[i] for i in indices])  # (64, H, W, 3)

        # Convert BGR → RGB
        clip_frames = clip_frames[:, :, :, ::-1].copy()

        # Pack for SlowFast
        slow_tensor, fast_tensor = pack_frames_slowfast(
            clip_frames,
            num_frames_slow=8,
            num_frames_fast=32,
            crop_size=self.cfg.crop_size,
        )

        # Add batch dimension
        slow_tensor = slow_tensor.unsqueeze(0).to(self.device)
        fast_tensor = fast_tensor.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.action_model([slow_tensor, fast_tensor])
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        action_label = IDX_TO_CLASS[pred_idx.item()]
        confidence = conf.item()

        # Only update if confidence exceeds threshold
        if confidence >= self.cfg.confidence_threshold:
            self.current_action = action_label
            self.current_confidence = confidence

            if action_label != "normal":
                self.alert_history.append({
                    "time": time.time(),
                    "action": action_label,
                    "confidence": confidence,
                })

        return self.current_action, self.current_confidence

    def draw_hud(
        self,
        frame: np.ndarray,
        action: str,
        confidence: float,
        tracks: list,
        loiter_alerts: list,
    ) -> np.ndarray:
        """Draw heads-up display with all detection info."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # ── Draw person bounding boxes and track IDs ──
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            tid = track.track_id

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(overlay, f"Person #{tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        # ── Draw loitering overlay ──
        overlay = self.loitering_detector.draw_overlay(overlay)

        # ── Action recognition status bar ──
        alert_cfg = ALERT_CONFIG.get(action, ALERT_CONFIG["normal"])
        bar_color = alert_cfg["color"]
        label = alert_cfg["label"]

        # Top status bar
        cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.putText(overlay, f"{label} ({confidence:.0%})", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bar_color, 2)

        # Severity indicator
        severity = alert_cfg["severity"]
        for i in range(5):
            color = bar_color if i < severity else (80, 80, 80)
            cv2.rectangle(overlay, (w - 180 + i * 30, 15), (w - 155 + i * 30, 45), color, -1)

        # ── Loitering alerts ──
        for alert_person in loiter_alerts:
            ax1, ay1, ax2, ay2 = alert_person.bbox
            mins = int(alert_person.total_time_in_zone) // 60
            secs = int(alert_person.total_time_in_zone) % 60
            cv2.putText(overlay, f"LOITERING {mins}m{secs:02d}s", (ax1, ay2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── Bottom info bar ──
        cv2.rectangle(overlay, (0, h - 30), (w, h), (30, 30, 30), -1)
        info_text = (f"Persons: {len(tracks)} | "
                     f"Buffer: {len(self.frame_buffer)}/64 | "
                     f"Device: {self.device}")
        cv2.putText(overlay, info_text, (15, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return overlay

    def run(self, source=0):
        """
        Main inference loop.

        Args:
            source: Camera index (int), video file path (str), or RTSP URL.
        """
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"✗ Cannot open video source: {source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.display_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.display_height)

        print(f"\n{'='*60}")
        print(f"  Security Monitoring Active")
        print(f"  Source: {source}")
        print(f"  Press 'q' to quit | 'z' to set zone | 's' to screenshot")
        print(f"{'='*60}\n")

        fps_counter = deque(maxlen=30)

        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Person detection
            detections = self.detect_persons(frame)

            # 2. Person tracking
            tracks = self.track_persons(detections, frame)

            # 3. Loitering detection (tracking-based)
            loiter_alerts = self.loitering_detector.update(tracks, frame)

            # 4. Action recognition (SlowFast, periodic)
            action, confidence = self.recognize_action(frame)

            # 5. Draw HUD
            display_frame = self.draw_hud(frame, action, confidence, tracks, loiter_alerts)

            # FPS
            fps_counter.append(time.time() - t_start)
            fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            cv2.putText(display_frame, f"FPS: {fps:.1f}",
                        (display_frame.shape[1] - 130, display_frame.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Security Monitor", display_frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fname, display_frame)
                print(f"📸 Screenshot saved: {fname}")

        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Monitoring stopped.")


# ══════════════════════════════════════════════════════════════════════
#  Demo mode (no trained model, just YOLO + tracking + loitering)
# ══════════════════════════════════════════════════════════════════════

class DemoPipeline:
    """
    Simplified pipeline for testing WITHOUT a trained SlowFast model.
    Only does person detection, tracking, and loitering detection.
    Useful for verifying your setup before training.
    """

    def __init__(self):
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.yolo = YOLO("yolov8n.pt")
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.loitering = LoiteringDetector(time_threshold=30.0)  # 30s for demo

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        print("Demo mode: Person detection + tracking + loitering (30s threshold)")
        print("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            results = self.yolo(frame, classes=[0], conf=0.5, verbose=False)
            dets = []
            for r in results:
                for box in (r.boxes or []):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1
                    dets.append(([x1, y1, w, h], conf, "person"))

            # Track
            tracks = self.tracker.update_tracks(dets, frame=frame)
            confirmed = [t for t in tracks if t.is_confirmed()]

            # Loitering
            alerts = self.loitering.update(confirmed, frame)

            # Draw
            frame = self.loitering.draw_overlay(frame)
            for t in confirmed:
                ltrb = t.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"#{t.track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

            for a in alerts:
                print(f"🚨 LOITERING ALERT: Person #{a.track_id} "
                      f"({a.total_time_in_zone:.0f}s in zone)")

            cv2.putText(frame, f"Persons: {len(confirmed)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Security Surveillance Inference")
    parser.add_argument("--source", default="0",
                        help="Video source: camera index, video file, or RTSP URL")
    parser.add_argument("--checkpoint", default="./output/checkpoints/best_model.pth",
                        help="Path to trained SlowFast model checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for action alerts")
    parser.add_argument("--loiter_time", type=float, default=120.0,
                        help="Loitering time threshold in seconds")
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode (no trained model needed)")
    args = parser.parse_args()

    if args.demo:
        pipeline = DemoPipeline()
        pipeline.run(int(args.source) if args.source.isdigit() else args.source)
        return

    cfg = InferenceConfig()
    cfg.checkpoint_path = args.checkpoint
    cfg.confidence_threshold = args.threshold
    cfg.loiter_time_threshold_sec = args.loiter_time
    if args.device:
        cfg.device = args.device

    pipeline = SecurityInferencePipeline(cfg)
    pipeline.run(int(args.source) if args.source.isdigit() else args.source)


if __name__ == "__main__":
    main()
