"""
Loitering / Prolonged Stay Detector
────────────────────────────────────
Uses YOLOv8 for person detection and DeepSORT for tracking.
If a tracked person stays in a defined zone longer than a threshold,
an alert is triggered.

This does NOT use action recognition — it is purely:
    Person Detection → Tracking → Time-in-zone > threshold → Alert
"""

import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TrackedPerson:
    """State for a single tracked person."""
    track_id: int
    first_seen: float           # timestamp
    last_seen: float            # timestamp
    in_zone_since: Optional[float] = None
    total_time_in_zone: float = 0.0
    alerted: bool = False
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1,y1,x2,y2


class LoiteringDetector:
    """
    Detect loitering (prolonged stay) in a specified zone.

    Usage:
        detector = LoiteringDetector(time_threshold=120.0)
        # Optional: define a polygon zone
        detector.set_zone([(100, 100), (500, 100), (500, 400), (100, 400)])

        for frame in video_stream:
            detections = yolo_model(frame)  # list of (x1,y1,x2,y2,conf,cls)
            tracks = tracker.update(detections)
            alerts = detector.update(tracks, frame)
    """

    def __init__(
        self,
        time_threshold: float = 120.0,   # seconds
        zone_polygon: Optional[List[Tuple[int, int]]] = None,
        cooldown_after_alert: float = 60.0,
    ):
        self.time_threshold = time_threshold
        self.zone_polygon = zone_polygon
        self.cooldown_after_alert = cooldown_after_alert
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self._zone_mask: Optional[np.ndarray] = None

    def set_zone(self, polygon: List[Tuple[int, int]], frame_shape: Optional[tuple] = None):
        """
        Define the monitoring zone as a polygon.
        If no zone is set, the entire frame is the zone.
        """
        self.zone_polygon = polygon
        if frame_shape is not None:
            self._precompute_zone_mask(frame_shape)

    def _precompute_zone_mask(self, frame_shape: tuple):
        """Create a binary mask for the zone polygon."""
        if self.zone_polygon is None:
            self._zone_mask = None
            return
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(self.zone_polygon, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        self._zone_mask = mask

    def _is_in_zone(self, bbox: Tuple[int, int, int, int], frame_shape: tuple) -> bool:
        """Check if the center of a bounding box is inside the zone."""
        if self.zone_polygon is None:
            return True  # No zone = entire frame

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Use mask if available
        if self._zone_mask is not None:
            h, w = self._zone_mask.shape
            if 0 <= cx < w and 0 <= cy < h:
                return self._zone_mask[cy, cx] > 0
            return False

        # Fallback: point-in-polygon test
        pts = np.array(self.zone_polygon, dtype=np.float32)
        result = cv2.pointPolygonTest(pts, (float(cx), float(cy)), False)
        return result >= 0

    def update(
        self,
        tracks: list,
        frame: np.ndarray,
    ) -> List[TrackedPerson]:
        """
        Update tracker state and return list of loitering alerts.

        Args:
            tracks: List of tracked objects. Each track should have:
                    - track.track_id (int)
                    - track.to_ltrb() → (x1, y1, x2, y2)
                    OR a simple tuple (track_id, x1, y1, x2, y2)
            frame: Current frame (for zone mask computation)

        Returns:
            List of TrackedPerson objects that have exceeded the time threshold.
        """
        now = time.time()
        frame_shape = frame.shape
        alerts = []

        # Precompute zone mask if needed
        if self.zone_polygon and self._zone_mask is None:
            self._precompute_zone_mask(frame_shape)

        # Parse tracks into (track_id, bbox) pairs
        current_ids = set()
        for track in tracks:
            if hasattr(track, "track_id"):
                tid = track.track_id
                if hasattr(track, "to_ltrb"):
                    bbox = tuple(map(int, track.to_ltrb()))
                elif hasattr(track, "to_tlbr"):
                    bbox = tuple(map(int, track.to_tlbr()))
                else:
                    continue
            elif isinstance(track, (list, tuple)) and len(track) >= 5:
                tid = int(track[0])
                bbox = tuple(map(int, track[1:5]))
            else:
                continue

            current_ids.add(tid)
            in_zone = self._is_in_zone(bbox, frame_shape)

            if tid not in self.tracked_persons:
                self.tracked_persons[tid] = TrackedPerson(
                    track_id=tid,
                    first_seen=now,
                    last_seen=now,
                    bbox=bbox,
                )

            person = self.tracked_persons[tid]
            person.last_seen = now
            person.bbox = bbox

            if in_zone:
                if person.in_zone_since is None:
                    person.in_zone_since = now
                person.total_time_in_zone = now - person.in_zone_since

                if (person.total_time_in_zone >= self.time_threshold and not person.alerted):
                    person.alerted = True
                    alerts.append(person)
            else:
                # Left the zone — reset
                person.in_zone_since = None
                person.total_time_in_zone = 0.0

        # Clean up stale tracks (not seen for > 30 seconds)
        stale_ids = [
            tid for tid, p in self.tracked_persons.items()
            if tid not in current_ids and (now - p.last_seen) > 30.0
        ]
        for tid in stale_ids:
            del self.tracked_persons[tid]

        return alerts

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone, tracked persons, and time info on the frame."""
        overlay = frame.copy()

        # Draw zone polygon
        if self.zone_polygon:
            pts = np.array(self.zone_polygon, dtype=np.int32)
            cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            # Semi-transparent fill
            zone_overlay = frame.copy()
            cv2.fillPoly(zone_overlay, [pts], (0, 255, 255))
            overlay = cv2.addWeighted(overlay, 0.85, zone_overlay, 0.15, 0)

        # Draw tracked persons
        for person in self.tracked_persons.values():
            x1, y1, x2, y2 = person.bbox
            time_in_zone = person.total_time_in_zone

            # Color: green → yellow → red based on time
            ratio = min(time_in_zone / self.time_threshold, 1.0)
            if ratio < 0.5:
                color = (0, 255, int(255 * ratio * 2))  # green → yellow
            else:
                color = (0, int(255 * (1 - ratio) * 2), 255)  # yellow → red

            if person.alerted:
                color = (0, 0, 255)  # solid red
                cv2.putText(overlay, "⚠ LOITERING", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Time label
            if time_in_zone > 0:
                mins, secs = divmod(int(time_in_zone), 60)
                time_str = f"ID:{person.track_id} {mins}m{secs:02d}s"
                cv2.putText(overlay, time_str, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return overlay

    def get_status(self) -> Dict:
        """Return current status of all tracked persons."""
        return {
            tid: {
                "track_id": p.track_id,
                "time_in_zone": p.total_time_in_zone,
                "alerted": p.alerted,
                "bbox": p.bbox,
            }
            for tid, p in self.tracked_persons.items()
        }
