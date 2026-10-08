"""Windows webcam: YOLO fire/smoke with per-region temporal confirmation.

Place this file beside weights/best.pt. S saves the raw frame; Q quits.
Matching uses bounding-box overlap, not motion or screen recognition.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import time


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "weights" / "best.pt"
CAPTURE_DIR = PROJECT_DIR / "data_collect" / "flash" / "images"

CONFIDENCE = 0.4
IMAGE_SIZE = 320
ALERT_INTERVAL = 5

# Moi vung duoc kiem tra rieng; moi lop fire/smoke co nguong rieng.
WINDOW_SECONDS = 2.0
MIN_HIT_RATIO = {"fire": 0.80, "smoke": 0.80}
MIN_MEAN_CONFIDENCE = {"fire": 0.60, "smoke": 0.60}
MATCH_IOU = 0.30
MIN_SAMPLES = 5
# Neu camera/xu ly bi ngat lau, bat dau quan sat lai.
MAX_FRAME_GAP = 1.0


def box_iou(a, b):
    """Muc chong lap cua hai khung, tu 0 den 1."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class TemporalVerifier:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.previous_time = None

    def update(self, detections, now):
        """detections: list of (label, confidence, (x1, y1, x2, y2))."""
        if self.previous_time is not None:
            if now - self.previous_time > MAX_FRAME_GAP:
                self.tracks.clear()
        self.previous_time = now

        # Xoa vung da mat qua lau.
        for track_id in list(self.tracks):
            if now - self.tracks[track_id]["last_seen"] > WINDOW_SECONDS:
                del self.tracks[track_id]

        # Ghep cung lop, uu tien cap co IoU lon nhat, moi khung chi ghep mot lan.
        pairs = []
        for track_id, track in self.tracks.items():
            for index, (label, score, box) in enumerate(detections):
                if label == track["label"]:
                    overlap = box_iou(track["box"], box)
                    if overlap >= MATCH_IOU:
                        pairs.append((overlap, track_id, index))

        matched_tracks = {}
        matched_detections = set()
        for overlap, track_id, index in sorted(pairs, reverse=True):
            if track_id not in matched_tracks and index not in matched_detections:
                matched_tracks[track_id] = index
                matched_detections.add(index)

        for track_id, track in self.tracks.items():
            score = 0.0  # Khong phat hien o khung nay: van tinh vao trung binh.
            track["visible"] = track_id in matched_tracks
            if track["visible"]:
                label, score, box = detections[matched_tracks[track_id]]
                track["box"] = box
                track["last_seen"] = now
            track["history"].append((now, score))

        for index, (label, score, box) in enumerate(detections):
            if index in matched_detections:
                continue
            self.tracks[self.next_id] = {
                "label": label,
                "box": box,
                "born": now,
                "last_seen": now,
                "visible": True,
                "history": deque([(now, score)]),
            }
            self.next_id += 1

        states = []
        for track_id, track in self.tracks.items():
            history = track["history"]
            while history and history[0][0] < now - WINDOW_SECONDS:
                history.popleft()

            scores = [score for timestamp, score in history]
            sample_count = len(scores)
            mean = sum(scores) / sample_count
            ratio = sum(score > 0 for score in scores) / sample_count
            age = now - track["born"]
            label = track["label"]
            ready = age + 1e-9 >= WINDOW_SECONDS and sample_count >= MIN_SAMPLES
            alarm = (
                ready
                and track["visible"]
                and ratio + 1e-9 >= MIN_HIT_RATIO[label]
                and mean + 1e-9 >= MIN_MEAN_CONFIDENCE[label]
            )
            states.append({
                "id": track_id, "label": label, "box": track["box"],
                "mean": mean, "ratio": ratio, "age": age,
                "samples": sample_count, "ready": ready, "alarm": alarm,
                "visible": track["visible"],
            })
        return states


def main():
    import threading
    import winsound
    import cv2
    from ultralytics import YOLO

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Dang tai model...")
    model = YOLO(str(MODEL_PATH))
    names = model.names
    labels = names.values() if isinstance(names, dict) else names
    if not {"fire", "smoke"}.issubset({str(x).lower().strip() for x in labels}):
        raise ValueError(f"Model can co hai lop fire va smoke. Hien tai: {names}")
    print("Cac lop:", names)

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("Khong mo duoc camera. Kiem tra quyen Camera cua Windows.")

    verifier = TemporalVerifier()
    last_alert_time = float("-inf")
    print("Nhan S de luu anh goc; Q de thoat. Bam vao cua so camera truoc.")
    print("Anh luu tai:", CAPTURE_DIR)

    try:
        while True:
            start_time = time.perf_counter()
            success, frame = camera.read()
            if not success:
                print("Khong doc duoc hinh anh tu camera.")
                break
            captured_at = time.monotonic()

            result = model.predict(
                source=frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                iou=0.5, device="cpu", max_det=20, verbose=False,
            )[0]

            detections = []
            if result.boxes is not None:
                # Moi dong: x1, y1, x2, y2, confidence, class_id.
                for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist():
                    label = str(names[int(class_id)]).lower().strip()
                    if label in MIN_MEAN_CONFIDENCE:
                        detections.append((label, score, (x1, y1, x2, y2)))

            states = verifier.update(detections, captured_at)
            alarms = [state for state in states if state["alarm"]]
            display_frame = result.plot()
            elapsed = time.perf_counter() - start_time
            fps = 1 / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"CPU - FPS: {fps:.1f} | S: save | Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            status = "CANH BAO: LUA/KHOI!" if alarms else "DANG GIAM SAT / XAC MINH"
            color = (0, 0, 255) if alarms else (0, 255, 255)
            cv2.putText(display_frame, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Hien thi toi da 4 vung, uu tien vung dang canh bao.
            visible_states = sorted(states, key=lambda s: (s["alarm"], s["visible"], s["mean"]), reverse=True)
            for row, state in enumerate(visible_states[:4]):
                phase = "ALARM" if state["alarm"] else ("CHECK" if state["ready"] else "WAIT")
                text = (
                    f"ID{state['id']} {state['label']}: avg={state['mean']:.2f} "
                    f"hit={state['ratio']:.0%} "
                    f"t={min(state['age'], WINDOW_SECONDS):.1f}s {phase}"
                )
                cv2.putText(display_frame, text, (10, 85 + row * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                if state["visible"]:
                    x1, y1, x2, y2 = map(int, state["box"])
                    cv2.putText(display_frame, f"ID{state['id']}",
                                (max(0, x1), min(frame.shape[0] - 8, max(15, y2))),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            now = time.monotonic()
            if alarms and now - last_alert_time >= ALERT_INTERVAL:
                # Beep chan luong goi; dung luong rieng de camera khong dung 0.5 giay.
                threading.Thread(target=winsound.Beep, args=(1800, 500), daemon=True).start()
                last_alert_time = now
                for state in alarms:
                    print(f"CANH BAO ID{state['id']} {state['label']}: "
                          f"avg={state['mean']:.3f}, hit={state['ratio']:.1%}")

            cv2.imshow("Fire and Smoke Detection - Press Q to quit", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                filename = datetime.now().strftime("flash_%Y%m%d_%H%M%S_%f.jpg")
                image_path = CAPTURE_DIR / filename
                if cv2.imwrite(str(image_path), frame):
                    print(f"Da luu anh: {image_path}")
                else:
                    print("Khong luu duoc anh.")
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
