

import cv2
import time
import threading
import requests
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Configuration — edit these to match your network / firmware
# ------------------------------------------------------------------

ESP32_S3_IP = "192.168.1.50"          # <-- set to your ESP32-S3's IP address
BASE_URL = f"http://{ESP32_S3_IP}"

FRAME1_URL = f"{BASE_URL}/frame1"      # CAM1 - sentry sweep
FRAME2_URL = f"{BASE_URL}/frame2"      # CAM2 - detection focus
TRIGGER_URL = f"{BASE_URL}/trigger"    # IR + laser trigger

POLL_INTERVAL_S = 0.15                 # ~6-7 fps polling for each camera
HTTP_TIMEOUT_S = 2.0

# Decision logic thresholds
DETECTION_CONFIDENCE_MIN = 0.35        # HOG detector weight threshold
CONSEC_FRAMES_TO_TRIGGER = 3           # avoid firing on a single noisy frame
COOLDOWN_S = 6.0                       # minimum gap between triggers

LOG_DIR = Path("antipiracy_logs")
LOG_DIR.mkdir(exist_ok=True)
EVENT_LOG_PATH = LOG_DIR / "events.log"


# ------------------------------------------------------------------
# Frame source — pulls the latest JPEG from an ESP32-CAM endpoint
# ------------------------------------------------------------------

class FrameSource:
    """Polls a single-shot JPEG endpoint on its own thread and keeps
    the most recent decoded frame available for reading."""

    def __init__(self, url: str, name: str):
        self.url = url
        self.name = name
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.last_error = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)

    def _loop(self):
        while self._running:
            try:
                # If your ESP32-S3 instead serves MJPEG at e.g. /stream1,
                # replace this whole loop body with:
                #   cap = cv2.VideoCapture(self.url)
                #   ok, frame = cap.read()
                resp = requests.get(self.url, timeout=HTTP_TIMEOUT_S)
                resp.raise_for_status()
                arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                    self.last_error = None
                else:
                    self.last_error = "Decoded frame was empty (bad JPEG data)"
            except Exception as exc:  # network hiccups shouldn't crash the app
                self.last_error = str(exc)
            time.sleep(POLL_INTERVAL_S)

    def is_stale(self) -> bool:
        """True if we currently have no usable frame to show."""
        return self.last_error is not None and self._frame is None

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


# ------------------------------------------------------------------
# Detection engine — lightweight person detector (runs on the PC)
# ------------------------------------------------------------------

class DetectionEngine:
    """Wraps OpenCV's built-in HOG + SVM person detector. This is the
    'lightweight person/object model' box from the architecture diagram —
    swap this class out for a TFLite/ONNX model later without touching
    the decision logic below."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        """Returns (detected: bool, boxes: list, best_confidence: float)."""
        if frame is None:
            return False, [], 0.0

        small = cv2.resize(frame, (320, 240))
        boxes, weights = self.hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        if len(weights) == 0:
            return False, [], 0.0

        best_conf = float(max(weights))
        detected = best_conf >= DETECTION_CONFIDENCE_MIN
        return detected, boxes.tolist(), best_conf


# ------------------------------------------------------------------
# Decision logic — debounces detections and fires the actuator trigger
# ------------------------------------------------------------------

@dataclass
class DecisionState:
    consec_hits: int = 0
    last_trigger_time: float = field(default_factory=lambda: 0.0)
    armed: bool = True


class DecisionLogic:
    def __init__(self, event_logger):
        self.state = DecisionState()
        self.log = event_logger

    def update(self, detected: bool, confidence: float):
        now = time.time()

        # Re-arm automatically once the cooldown window has elapsed.
        if not self.state.armed and (now - self.state.last_trigger_time) >= COOLDOWN_S:
            self.state.armed = True

        if not detected:
            self.state.consec_hits = 0
            return False

        if not self.state.armed:
            # Still cooling down from the last trigger — count the hit
            # so a sustained detection fires immediately once re-armed,
            # but don't trigger again yet.
            return False

        self.state.consec_hits += 1

        if self.state.consec_hits >= CONSEC_FRAMES_TO_TRIGGER:
            self.state.last_trigger_time = now
            self.state.consec_hits = 0
            self.state.armed = False
            self.log(f"TRIGGER fired (confidence={confidence:.2f})")
            return True

        return False


# ------------------------------------------------------------------
# Actuator + telemetry — fires IR/laser, logs to file (WiFi telemetry
# analog) instead of a real dashboard backend
# ------------------------------------------------------------------

def fire_trigger():
    try:
        requests.post(TRIGGER_URL, timeout=HTTP_TIMEOUT_S)
        return True
    except Exception as exc:
        log_event(f"Trigger request failed: {exc}")
        return False


def log_event(message: str):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line)
    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------
# Operator dashboard — live dual-feed window with overlay + event log
# ------------------------------------------------------------------

def draw_dashboard(frame1, frame2, detected, confidence, boxes, triggered, last_events, cam1_error=None, cam2_error=None):
    h, w = 240, 320
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    f1 = cv2.resize(frame1, (w, h)) if frame1 is not None else blank.copy()
    f2 = cv2.resize(frame2, (w, h)) if frame2 is not None else blank.copy()

    if frame1 is None and cam1_error:
        cv2.putText(f1, "CAM1 unreachable", (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if frame2 is None and cam2_error:
        cv2.putText(f2, "CAM2 unreachable", (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for (x, y, bw, bh) in boxes:
        scale_x, scale_y = w / 320, h / 240
        cv2.rectangle(
            f2,
            (int(x * scale_x), int(y * scale_y)),
            (int((x + bw) * scale_x), int((y + bh) * scale_y)),
            (0, 0, 255) if detected else (0, 255, 0),
            2,
        )

    cv2.putText(f1, "CAM 1 - sentry sweep", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(f2, "CAM 2 - detection focus", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    top_row = np.hstack([f1, f2])

    panel = np.zeros((160, top_row.shape[1], 3), dtype=np.uint8)
    status_color = (0, 0, 255) if triggered else ((0, 200, 255) if detected else (0, 200, 0))
    status_text = "TRIGGERED" if triggered else ("DETECTED" if detected else "MONITORING")

    cv2.putText(panel, f"Status: {status_text}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(panel, f"Confidence: {confidence:.2f}", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(panel, "Recent events:", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    for i, ev in enumerate(last_events[-3:]):
        cv2.putText(panel, ev[:70], (12, 104 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return np.vstack([top_row, panel])


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    log_event("Operator dashboard started")

    cam1 = FrameSource(FRAME1_URL, "CAM1")
    cam2 = FrameSource(FRAME2_URL, "CAM2")
    cam1.start()
    cam2.start()

    engine = DetectionEngine()
    logic = DecisionLogic(log_event)
    recent_events = []
    triggered_display_until = 0.0

    try:
        while True:
            frame1 = cam1.read()
            frame2 = cam2.read()

            detected, boxes, confidence = engine.detect(frame2)
            should_trigger = logic.update(detected, confidence)

            if should_trigger:
                ok = fire_trigger()
                triggered_display_until = time.time() + 1.5
                recent_events.append(f"Trigger fired ({'ok' if ok else 'failed'}), conf={confidence:.2f}")

            display_triggered = time.time() < triggered_display_until

            dash = draw_dashboard(
                frame1, frame2, detected, confidence, boxes, display_triggered, recent_events,
                cam1_error=cam1.last_error, cam2_error=cam2.last_error,
            )
            cv2.imshow("Anti-Piracy Rig - Operator Dashboard", dash)

            # waitKey also paces the loop (~30fps ceiling) so a burst of
            # None frames during a network hiccup doesn't spin the CPU
            # or flood the ESP32-S3 with retries.
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cam1.stop()
        cam2.stop()
        cv2.destroyAllWindows()
        log_event("Operator dashboard stopped")


if __name__ == "__main__":
    main()