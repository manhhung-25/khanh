"""Select a local video; verify fire/smoke and send timestamped images to Zalo.
Use existing zalo_config.json and weights/best.pt. Q quits; S saves a frame.
Temporal confirmation uses video time; Zalo cooldown uses wall-clock time.
"""

from collections import deque
from datetime import datetime
from pathlib import Path
import time
import base64
import json
import threading
import urllib.request
import urllib.parse
import urllib.error


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "weights" / "best.pt"
CAPTURE_DIR = PROJECT_DIR / "outputs" / "video_snapshots"

IMAGE_SIZE = 640
CONFIDENCE = 0.25
MIN_DETECTION_CONFIDENCE = {"fire": 0.25, "smoke": 0.40}
ALERT_INTERVAL = 5
ZALO_INTERVAL = 30.0  # Giay thuc; dung chung cho tat ca vung fire/smoke.
CONFIG_PATH = PROJECT_DIR / "zalo_config.json"
ALERT_DIR = PROJECT_DIR / "outputs" / "alerts"

# Moi vung duoc kiem tra rieng; moi lop fire/smoke co nguong rieng.
WINDOW_SECONDS = 2.0
MIN_HIT_RATIO = {"fire": 0.80, "smoke": 0.80}
# Diem trung binh bao gom 0 cho nhung khung hinh khong phat hien.
MIN_MEAN_CONFIDENCE = {"fire": 0.32, "smoke": 0.60}
MATCH_IOU = 0.30
MATCH_CENTER_DISTANCE = 0.35
MIN_CENTER_DISTANCE_PX = 12.0  # Khung lua nho de bi lech vai pixel giua cac frame.
MAX_AREA_RATIO = 4.0
MIN_SAMPLES = 5
# Neu camera/xu ly bi ngat lau, bat dau quan sat lai.
MAX_FRAME_GAP = WINDOW_SECONDS


def box_iou(a, b):
    """Muc chong lap cua hai khung, tu 0 den 1."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_score(a, b, min_center_distance_px=0.0):
    """Cho phep khung doi kich thuoc/vi tri vua phai ma khong tao ID moi."""
    overlap = box_iou(a, b)
    if overlap >= MATCH_IOU:
        return 1.0 + overlap

    aw, ah = max(0.0, a[2] - a[0]), max(0.0, a[3] - a[1])
    bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    area_a, area_b = aw * ah, bw * bh
    if min(area_a, area_b) <= 0 or max(area_a, area_b) / min(area_a, area_b) > MAX_AREA_RATIO:
        return None
    center_distance = (((a[0] + a[2] - b[0] - b[2]) / 2) ** 2
                       + ((a[1] + a[3] - b[1] - b[3]) / 2) ** 2) ** 0.5
    scale = max((aw * aw + ah * ah) ** 0.5, (bw * bw + bh * bh) ** 0.5)
    tolerance = max(MATCH_CENTER_DISTANCE * scale, min_center_distance_px)
    if center_distance <= tolerance:
        return 1.0 - center_distance / tolerance
    return None


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

        # Ghep cung lop; uu tien chong lap, sau do tam gan + kich thuoc tuong tu.
        # Vung mat khung ngan van giu ID va lich su; mau mat van la 0.
        pairs = []
        for track_id, track in self.tracks.items():
            for index, (label, score, box) in enumerate(detections):
                if label == track["label"]:
                    floor_px = MIN_CENTER_DISTANCE_PX if label == "fire" else 0.0
                    similarity = match_score(track["box"], box, floor_px)
                    if similarity is not None:
                        pairs.append((similarity, track_id, index))

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
        "expiration": "3600",
    }).encode("ascii")
    request = urllib.request.Request(
        "https://api.imgbb.com/1/upload", data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST",
    )
    result = read_json_response(request, timeout=30)
    if result.get("success") is not True:
        raise RuntimeError("ImgBB tu choi tai anh. Kiem tra API key / han muc.")
    photo_url = result.get("data", {}).get("url")
    if not isinstance(photo_url, str) or not photo_url.startswith("https://"):
        raise RuntimeError("ImgBB khong tra ve URL anh hop le.")
    return photo_url


def send_zalo_photo(config, photo_url, caption):
    payload = json.dumps({
        "chat_id": config["chat_id"], "photo": photo_url, "caption": caption[:2000],
    }, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"https://bot-api.zaloplatforms.com/bot{config['bot_token']}/sendPhoto",
        data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    result = read_json_response(request, timeout=20)
    if result.get("ok") is not True:
        raise RuntimeError("Zalo tu choi gui anh. Kiem tra Token, Chat ID va quyen bot.")


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

    def try_submit(self, image, captured_time, alarms, video_name, video_second):
        with self.lock:
            if self.busy or self.clock() < self.next_allowed:
                return False
            self.busy = True
            self.status = "SENDING"
        details = "\n".join(
            f"- {s['label']} ID{s['id']}: diem TB {s['mean']:.2f}; "
            f"ti le phat hien {s['ratio']:.0%} ({s['samples']} khung)"
            for s in alarms[:8]
        )
        caption = (
            "🔥 CẢNH BÁO AI — KIỂM TRA VIDEO\n"
            f"Thời gian xử lý: {captured_time.strftime('%d/%m/%Y %H:%M:%S %z')}\n"
            "Phát hiện nghi ngờ lửa/khói sau kiểm tra chuỗi hình:\n"
            f"{details}\n"
            f"Video: {video_name}\n"
            f"Vị trí trong video: {format_video_time(video_second)}\n"
            "Ảnh trích từ video đã chọn, không phải camera trực tiếp."
        )
        try:
            # Copy de luong gui khong doc anh dang bi main thay doi.
            worker = threading.Thread(
                target=self._send, args=(image.copy(), captured_time, caption), daemon=False,
            )
            self.worker = worker
            worker.start()
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
            stage = "gui anh Zalo"
            send_zalo_photo(self.config, photo_url, caption)
            status = "SENT"
            print("Da gui anh va thoi gian canh bao ve Zalo.")
        except urllib.error.HTTPError as error:
            # Khong in URL: URL Zalo chua Token.
            print(f"Loi HTTP {error.code} khi {stage}. Kiem tra cau hinh / han muc.")
        except Exception as error:
            # Khong in error/response goc de tranh lo Token/API key.
            print(f"Khong thanh cong khi {stage} ({type(error).__name__}). "
                  "Kiem tra mang va cau hinh; video van tiep tuc.")
        finally:
            with self.lock:
                # Sau khi gui xong (hoac that bai), doi it nhat 30 giay.
                self.next_allowed = self.clock() + ZALO_INTERVAL
                self.status = status
                self.busy = False


def format_video_time(seconds):
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, ms = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


def choose_video():
    from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Chon video nhan dien lua / khoi",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()


def main():
    import cv2
    from ultralytics import YOLO
    try:
        import winsound
    except ImportError:
        winsound = None

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    config = load_zalo_config()
    video_path = choose_video()
    if not video_path:
        print("Chua chon video. Da thoat.")
        return
    print("Dang tai model...")
    model = YOLO(str(MODEL_PATH))
    names = model.names
    labels = names.values() if isinstance(names, dict) else names
    if not {"fire", "smoke"}.issubset({str(x).lower().strip() for x in labels}):
        raise ValueError(f"Model can co hai lop fire va smoke. Hien tai: {names}")

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        video.release()
        raise RuntimeError("Khong mo duoc video da chon.")
    writer = None
    sender = ZaloAlertSender(config)
    verifier = TemporalVerifier()
    last_beep = float("-inf")
    frame_number = 0
    previous_video_second = -1.0
    fps = video.get(cv2.CAP_PROP_FPS)
    if not 0 < fps < 1000:
        fps = 25.0
        print("Khong doc duoc FPS hop le; tam dung 25 FPS.")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = PROJECT_DIR / "outputs" / datetime.now().strftime("video_result_%Y%m%d_%H%M%S_%f.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    window_title = "Video Fire / Smoke - S: save - Q: quit"
    print("Bo loc 2s theo VIDEO; gui Zalo cach it nhat 30s THUC.")
    print("Neu CPU cham, video se phat cham hon toc do goc; khong bo khung.")
    try:
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 960, 540)
        while True:
            started = time.perf_counter()
            ok, frame = video.read()
            if not ok:
                break
            frame_number += 1
            captured_time = datetime.now().astimezone()
            # Prefer decoder timestamps, fall back when missing/non-increasing.
            raw_second = video.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            fallback = (frame_number - 1) / fps
            video_second = raw_second if raw_second >= 0 and raw_second > previous_video_second else max(fallback, previous_video_second + 1 / fps)
            previous_video_second = video_second
            result = model.predict(source=frame, imgsz=IMAGE_SIZE, conf=CONFIDENCE,
                                   iou=0.5, device="cpu", max_det=20, verbose=False)[0]
            detections = []
            if result.boxes is not None:
                for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist():
                    label = str(names[int(class_id)]).lower().strip()
                    if label in MIN_DETECTION_CONFIDENCE and score >= MIN_DETECTION_CONFIDENCE[label]:
                        detections.append((label, score, (x1, y1, x2, y2)))
            states = verifier.update(detections, video_second)
            alarms = [state for state in states if state["alarm"]]
            display = result.plot()
            # Draw in a separate banner so text remains legible on small videos.
            height, width = display.shape[:2]
            banner_height = 205
            annotated = cv2.copyMakeBorder(display, banner_height, 0, 0, max(0, 760-width),
                                          cv2.BORDER_CONSTANT, value=(25, 25, 25))
            lines = [
                (f"Now: {captured_time.strftime('%d/%m/%Y %H:%M:%S %z')}", (255,255,255)),
                (f"Video: {format_video_time(video_second)} | frame {frame_number}/{total_frames}", (255,255,255)),
                ("ALARM: FIRE / SMOKE" if alarms else "MONITORING / VERIFYING", (0,0,255) if alarms else (0,255,255)),
                (sender.overlay_status(), (255,255,255)),
            ]
            for state in sorted(states, key=lambda x:(x["alarm"],x["visible"],x["mean"]), reverse=True)[:3]:
                phase = "ALARM" if state["alarm"] else ("CHECK" if state["ready"] else "WAIT")
                lines.append((f"ID{state['id']} {state['label']} avg={state['mean']:.2f} hit={state['ratio']:.0%} t={min(state['age'], WINDOW_SECONDS):.1f}s {phase}", (0,255,255)))
            for row, (text, color) in enumerate(lines):
                cv2.putText(annotated, text, (10, 24 + row*26), cv2.FONT_HERSHEY_SIMPLEX, .56, color, 1, cv2.LINE_AA)
            for state in states:
                if state["visible"]:
                    x1,y1,x2,y2 = map(int,state["box"])
                    cv2.putText(annotated, f"ID{state['id']}", (max(0,x1), min(height-5,max(15,y2))+banner_height),
                                cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            if alarms:
                sender.try_submit(annotated, captured_time, alarms, Path(video_path).name, video_second)
                now = time.monotonic()
                if winsound and now-last_beep >= ALERT_INTERVAL:
                    threading.Thread(target=winsound.Beep,args=(1800,500),daemon=True).start()
                    last_beep = now
            # Output includes banner and keeps original FPS; create writer lazily
            # with the exact annotated dimensions on first frame.
            if frame_number == 1:
                ah, aw = annotated.shape[:2]
                writer = cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(aw,ah))
                if not writer.isOpened():
                    raise RuntimeError("Khong tao duoc video ket qua.")
            writer.write(annotated)
            cv2.imshow(window_title, annotated)
            delay_ms = max(1, round(1000 * (1/fps - (time.perf_counter()-started))))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("s"),ord("S")):
                CAPTURE_DIR.mkdir(parents=True,exist_ok=True)
                destination = CAPTURE_DIR / captured_time.strftime("snapshot_%Y%m%d_%H%M%S_%f.jpg")
                encoded_ok, encoded = cv2.imencode(".jpg",annotated)
                if encoded_ok:
                    destination.write_bytes(encoded.tobytes())
                    print("Da luu anh:",destination)
            elif key in (ord("q"),ord("Q")):
                break
    finally:
        video.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if sender.worker is not None and sender.worker.is_alive():
            print("Dang hoan tat lan gui Zalo cuoi. Vui long cho...")
            sender.worker.join()
    if frame_number:
        print("Video ket qua (khong kem am thanh):",output_path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(error)
