"""Windows webcam: confirm persistent, confident detections before Zalo alerts.

Place this file beside fire.pt and zalo_config.json. S saves raw frames; Q quits.
All model classes are eligible; each region must pass temporal confirmation.
Alert images are uploaded to ImgBB (expire after 1 hour) then sent to Zalo.
"""

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import time
import base64
import json
import threading
import urllib.request
import urllib.parse
import urllib.error
from camera_source import CAMERA_NAME, find_camera_index


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "fire.pt"
CAPTURE_DIR = PROJECT_DIR / "data_collect" / "flash" / "images"

# Model hien tai duoc huan luyen o imgsz=640.
# Giu 640x480 de han che chi phi xu ly tren CPU.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
IMAGE_SIZE = 640
# Giu nguong suy luan nhu cua so test; nguong gui canh bao nghiem hon.
CONFIDENCE = 0.25
ALERT_CONFIDENCE = 0.60
WINDOW_SECONDS = 2.0
MIN_HIT_RATIO = 0.80
MIN_MEAN_CONFIDENCE = 0.65  # Tinh ca 0 cho khung hinh khong phat hien.
MIN_SAMPLES = 5 #khung hinh de xu ly
MATCH_IOU = 0.30
MATCH_CENTER_DISTANCE = 0.35
MAX_AREA_RATIO = 4.0
MAX_FRAME_GAP = WINDOW_SECONDS
ALERT_INTERVAL = 5 #coi
ZALO_INTERVAL = 30.0  # Giay thuc; dung chung cho tat ca vung fire/smoke.
PHOTO_EXPIRATION_SECONDS = 3600  # Thoi gian ton tai cua link anh tren ImgBB.
CONFIG_PATH = PROJECT_DIR / "zalo_config.json"
ALERT_DIR = PROJECT_DIR / "outputs" / "alerts"


def box_iou(a, b):
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_score(a, b):
    overlap = box_iou(a, b)
    if overlap >= MATCH_IOU:
        return 1.0 + overlap
    aw, ah = max(0.0, a[2] - a[0]), max(0.0, a[3] - a[1])
    bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    area_a, area_b = aw * ah, bw * bh
    if min(area_a, area_b) <= 0 or max(area_a, area_b) / min(area_a, area_b) > MAX_AREA_RATIO:
        return None
    distance = (((a[0] + a[2] - b[0] - b[2]) / 2) ** 2
                + ((a[1] + a[3] - b[1] - b[3]) / 2) ** 2) ** 0.5
    scale = max((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5)
    tolerance = MATCH_CENTER_DISTANCE * scale
    return 1.0 - distance / tolerance if distance <= tolerance else None


class TemporalVerifier:
    """Confirm each region independently; weak or missing detections count as misses."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.previous_time = None

    def update(self, detections, now):
        # detections: (class_id, confidence, (x1, y1, x2, y2)).
        detections = [detection for detection in detections if detection[1] >= ALERT_CONFIDENCE]
        if self.previous_time is not None and now - self.previous_time > MAX_FRAME_GAP:
            self.tracks.clear()
        self.previous_time = now
        for track_id in list(self.tracks):
            if now - self.tracks[track_id]["last_seen"] > WINDOW_SECONDS:
                del self.tracks[track_id]

        pairs = []
        for track_id, track in self.tracks.items():
            for index, (class_id, _, box) in enumerate(detections):
                if class_id == track["class_id"]:
                    similarity = match_score(track["box"], box)
                    if similarity is not None:
                        pairs.append((similarity, track_id, index))
        matched_tracks = {}
        matched_detections = set()
        for _, track_id, index in sorted(pairs, reverse=True):
            if track_id not in matched_tracks and index not in matched_detections:
                matched_tracks[track_id] = index
                matched_detections.add(index)
        for track_id, track in self.tracks.items():
            score = 0.0
            track["visible"] = track_id in matched_tracks
            if track["visible"]:
                _, score, box = detections[matched_tracks[track_id]]
                track["box"] = box
                track["last_seen"] = now
            track["history"].append((now, score))
        for index, (class_id, score, box) in enumerate(detections):
            if index not in matched_detections:
                self.tracks[self.next_id] = {
                    "class_id": class_id, "box": box, "born": now,
                    "last_seen": now, "visible": True, "history": deque([(now, score)]),
                }
                self.next_id += 1

        states = []
        for track_id, track in self.tracks.items():
            history = track["history"]
            while history and history[0][0] < now - WINDOW_SECONDS:
                history.popleft()
            scores = [score for _, score in history]
            mean = sum(scores) / len(scores)
            ratio = sum(score > 0 for score in scores) / len(scores)
            age = now - track["born"]
            ready = age + 1e-9 >= WINDOW_SECONDS and len(scores) >= MIN_SAMPLES
            alarm = (ready and track["visible"]
                     and ratio + 1e-9 >= MIN_HIT_RATIO
                     and mean + 1e-9 >= MIN_MEAN_CONFIDENCE)
            states.append({
                "id": track_id, "box": track["box"], "class_id": track["class_id"],
                "mean": mean, "ratio": ratio, "age": age, "samples": len(scores),
                "ready": ready, "visible": track["visible"], "alarm": alarm,
            })
        return states


def load_zalo_config():
    """Chi doc file, khong hoi Token/Chat ID moi lan chay."""
    template = {"bot_token": "", "chat_id": "", "imgbb_api_key": ""}
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(template, indent=2), encoding="utf-8")
        raise ValueError(f"Da tao {CONFIG_PATH}. Dien 3 thong tin mot lan, luu va chay lai.")
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        raise ValueError("Khong doc duoc zalo_config.json. Kiem tra cu phap JSON.") from None
    if not isinstance(config, dict):
        raise ValueError("zalo_config.json phai la mot doi tuong JSON.")
    for field in template:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Hay dien {field} trong {CONFIG_PATH}, roi chay lai.")
        config[field] = value.strip()
    return config


def read_json_response(request, timeout):
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def upload_photo(config, jpeg_bytes):
    # Zalo sendPhoto can URL anh: tai anh len ImgBB truoc.
    # Anh tren ImgBB tu xoa sau 1 gio; ban luu tren may van con.
    payload = urllib.parse.urlencode({
        "key": config["imgbb_api_key"],
        "image": base64.b64encode(jpeg_bytes).decode("ascii"),
        "expiration": str(PHOTO_EXPIRATION_SECONDS),
    }).encode("ascii")
    request = urllib.request.Request(
        "https://api.imgbb.com/1/upload", data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
    )
    result = read_json_response(request, timeout=30)
    if result.get("success") is not True:
        raise RuntimeError("ImgBB tu choi tai anh. Kiem tra API key / han muc.")
    data = result.get("data", {})
    photo_url = data.get("image", {}).get("url") or data.get("url")
    if not isinstance(photo_url, str) or not photo_url.startswith("https://"):
        raise RuntimeError("ImgBB khong tra ve URL anh hop le.")
    return photo_url


def check_photo_url(photo_url):
    """Reject HTML error pages and invalid images before asking Zalo to fetch."""
    import cv2
    import numpy as np

    request = urllib.request.Request(photo_url, headers={"Accept": "image/jpeg"})
    with urllib.request.urlopen(request, timeout=15) as response:
        content_type = response.headers.get_content_type()
        jpeg_bytes = response.read(5 * 1024 * 1024 + 1)
    if content_type != "image/jpeg" or len(jpeg_bytes) > 5 * 1024 * 1024:
        raise RuntimeError("Link ImgBB khong tra ve anh JPEG hop le.")
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Khong giai ma duoc anh tu link ImgBB.")
    print(f"Link anh da kiem tra: JPEG {image.shape[1]}x{image.shape[0]}, "
          f"{len(jpeg_bytes)} bytes.", flush=True)


def send_zalo_photo(config, photo_url, caption):
    suffix = (f"\nMở ảnh trực tiếp: {photo_url}\n"
              f"Link ảnh tự xóa sau {PHOTO_EXPIRATION_SECONDS / 3600:g} giờ.")
    payload = json.dumps({
        "chat_id": config["chat_id"], "photo": photo_url,
        "caption": caption[:max(0, 2000 - len(suffix))] + suffix,
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"https://bot-api.zaloplatforms.com/bot{config['bot_token']}/sendPhoto",
        data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    result = read_json_response(request, timeout=20)
    if result.get("ok") is not True:
        raise RuntimeError("Zalo tu choi gui anh. Kiem tra Token, Chat ID va quyen bot.")
    return result.get("result", {}).get("message_id")


class ZaloAlertSender:
    """Mot luong gui, khong xep hang anh cu, khong chan camera."""
    def __init__(self, config, clock=time.monotonic):
        self.config = config
        self.clock = clock
        self.lock = threading.Lock()
        self.busy = False
        self.next_allowed = float("-inf")
        self.status = "READY"
        self.worker = None

    def overlay_status(self):
        with self.lock:
            if self.busy:
                return "Zalo: SENDING"
            remaining = max(0.0, self.next_allowed - self.clock())
            return f"Zalo: {self.status} | wait {remaining:.0f}s"

    def try_submit(self, image, captured_time):
        with self.lock:
            if self.busy or self.clock() < self.next_allowed:
                return False
            self.busy = True
            self.status = "SENDING"
        caption = (
            "🚨 NGUY HIỂM - CẢNH BÁO CAMERA AI\n"
            f"Thời gian chụp: {captured_time.strftime('%d/%m/%Y %H:%M:%S %z')}\n"
            "Camera phát hiện dấu hiệu nghi là lửa/khói. Vui lòng kiểm tra ngay!\n"
            f"Vùng phát hiện đã được xác minh trong {WINDOW_SECONDS:g} giây.\n"
            "Ảnh camera tại thời điểm phát hiện."
        )
        try:
            self.worker = threading.Thread(
                target=self._send, args=(image.copy(), captured_time, caption), daemon=True,
            )
            self.worker.start()
        except Exception:
            with self.lock:
                self.busy = False
                self.status = "ERROR"
                self.next_allowed = self.clock() + ZALO_INTERVAL
            raise
        return True

    def _send(self, image, captured_time, caption):
        import cv2
        status = "ERROR"
        stage = "luu anh"
        try:
            ALERT_DIR.mkdir(parents=True, exist_ok=True)
            path = ALERT_DIR / captured_time.strftime("alert_%Y%m%d_%H%M%S_%f.jpg")
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Khong ma hoa duoc anh JPG.")
            jpeg_bytes = encoded.tobytes()
            path.write_bytes(jpeg_bytes)  # Ho tro duong dan Windows co dau.
            print("Da luu anh canh bao:", path)
            stage = "tai anh len ImgBB"
            photo_url = upload_photo(self.config, jpeg_bytes)
            metadata_path = path.with_suffix(".json")
            metadata = {
                "captured_at": captured_time.isoformat(),
                "photo_url": photo_url,
                "expires_at": (datetime.now().astimezone()
                               + timedelta(seconds=PHOTO_EXPIRATION_SECONDS)).isoformat(),
                "photo_checked": False,
                "zalo_accepted": False,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            stage = "kiem tra link anh ImgBB"
            check_photo_url(photo_url)
            metadata["photo_checked"] = True
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            stage = "gui anh Zalo"
            message_id = send_zalo_photo(self.config, photo_url, caption)
            metadata.update(zalo_accepted=True, message_id=message_id)
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            status = "SENT"
            print("Zalo da chap nhan tin nhan anh. "
                  "Neu anh khong hien, mo link trong chu thich. "
                  f"Thong tin kiem tra: {metadata_path}", flush=True)
        except urllib.error.HTTPError as error:
            # Khong in URL: URL Zalo chua Token.
            print(f"Loi HTTP {error.code} khi {stage}. Kiem tra cau hinh / han muc.")
        except Exception as error:
            # Khong in error/response goc de tranh lo Token/API key.
            print(f"Khong thanh cong khi {stage} ({type(error).__name__}). "
                  "Kiem tra mang va cau hinh; camera van tiep tuc.")
        finally:
            with self.lock:
                # Sau khi gui xong (hoac that bai), doi it nhat 30 giay.
                self.next_allowed = self.clock() + ZALO_INTERVAL
                self.status = status
                self.busy = False


def main():
    import winsound
    import cv2
    from ultralytics import YOLO

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Dang tai model:", MODEL_PATH, flush=True)
    model = YOLO(str(MODEL_PATH))
    zalo_sender = ZaloAlertSender(load_zalo_config())

    camera = cv2.VideoCapture(find_camera_index(), cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Khong mo duoc {CAMERA_NAME}. Kiem tra quyen Camera cua Windows.")

    verifier = TemporalVerifier()
    last_alert_time = float("-inf")
    printed_camera_size = False
    window_title = f"Fire and Smoke Detection - {CAMERA_NAME} - {MODEL_PATH.name} - Press Q to quit"
    print("Nhan S de luu anh goc; Q de thoat. Bam vao cua so camera truoc.")
    print("Anh luu tai:", CAPTURE_DIR)
    print(f"Canh bao: confidence >= {ALERT_CONFIDENCE:.2f}, xac minh {WINDOW_SECONDS:g}s, "
          f"hit >= {MIN_HIT_RATIO:.0%}, diem TB >= {MIN_MEAN_CONFIDENCE:.2f}. "
          "Moi lan gui Zalo cach it nhat 30 giay.", flush=True)

    try:
        while True:
            start_time = time.perf_counter()
            success, frame = camera.read()
            if not success:
                print("Khong doc duoc hinh anh tu camera.")
                break
            if not printed_camera_size:
                height, width = frame.shape[:2]
                print(f"Camera thuc te: {width}x{height}; YOLO imgsz={IMAGE_SIZE}.", flush=True)
                printed_camera_size = True
            captured_at = time.monotonic()
            captured_time = datetime.now().astimezone()

            # Giu ket qua nhu cua so test, chi canh bao vung da qua xac minh.
            result = model.predict(
                source=frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                iou=0.5, device="cpu", max_det=20, verbose=False,
            )[0]
            detections = []
            if result.boxes is not None:
                detections = [(int(class_id), score, (x1, y1, x2, y2))
                              for x1, y1, x2, y2, score, class_id
                              in result.boxes.data.cpu().tolist()]
            states = verifier.update(detections, captured_at)
            alarms = [state for state in states if state["alarm"]]
            detected = bool(alarms)
            display_frame = result.plot(labels=False, conf=False)
            elapsed = time.perf_counter() - start_time
            fps = 1 / elapsed if elapsed > 0 else 0
            cv2.putText(display_frame, f"{MODEL_PATH.name} | CPU FPS: {fps:.1f} | S: save | Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            visible_states = [state for state in states if state["visible"]]
            status = "NGUY HIEM: PHAT HIEN LUA/KHOI!" if detected else (
                "DANG XAC MINH - CHUA GUI ZALO" if visible_states else "DANG GIAM SAT")
            color = (0, 0, 255) if detected else (0, 255, 255)
            cv2.putText(display_frame, status, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            if visible_states:
                strongest = max(visible_states, key=lambda state: (state["alarm"], state["mean"]))
                progress = (f"Confirm: {min(strongest['age'], WINDOW_SECONDS):.1f}/{WINDOW_SECONDS:g}s "
                            f"| avg={strongest['mean']:.2f} | hit={strongest['ratio']:.0%}")
                cv2.putText(display_frame, progress, (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            frame_height = display_frame.shape[0]
            cv2.putText(display_frame, captured_time.strftime("%d/%m/%Y %H:%M:%S"),
                        (10, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, zalo_sender.overlay_status(),
                        (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
            if detected:
                zalo_sender.try_submit(display_frame, captured_time)
                now = time.monotonic()
                if now - last_alert_time >= ALERT_INTERVAL:
                    threading.Thread(target=winsound.Beep, args=(1800, 500), daemon=True).start()
                    last_alert_time = now
                    print("NGUY HIEM: model phat hien dau hieu lua/khoi.", flush=True)

            cv2.imshow(window_title, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                filename = captured_time.strftime("flash_%Y%m%d_%H%M%S_%f.jpg")
                image_path = CAPTURE_DIR / filename
                ok, encoded = cv2.imencode(".jpg", frame)
                if ok:
                    image_path.write_bytes(encoded.tobytes())
                    print(f"Da luu anh: {image_path}", flush=True)
                else:
                    print("Khong luu duoc anh.")
            elif key in (ord("q"), ord("Q")) or cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if zalo_sender.worker is not None and zalo_sender.worker.is_alive():
            print("Dang hoan tat lan gui Zalo cuoi...", flush=True)
            zalo_sender.worker.join()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(error)
