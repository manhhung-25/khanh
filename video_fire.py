from pathlib import Path
from datetime import datetime
from tkinter import Tk, filedialog
import getpass
import json
import urllib.request
import urllib.error

import cv2
from ultralytics import YOLO




PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "weights" / "best.pt"
OUTPUT_DIR = PROJECT_DIR / "outputs"




IMAGE_SIZE = 320
CONFIDENCE = 0.4


ALERT_AFTER_FRAMES = 3

# Sau 60 giây trong video mới được gửi cảnh báo tiếp
ALERT_COOLDOWN_SECONDS = 60



# HÀM GỬI TIN NHẮN ZALO


def send_zalo_message(bot_token, chat_id, message):
    url = (
        "https://bot-api.zaloplatforms.com/"
        f"bot{bot_token}/sendMessage"
    )

    payload = json.dumps(
        {
            "chat_id": chat_id,
            "text": message,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))

        if result.get("ok"):
            print("Da gui canh bao len Zalo.")
            return True

        print("Zalo khong gui duoc tin nhan:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return False

    except urllib.error.HTTPError as error:
        print("Loi HTTP khi gui Zalo:", error.code)
        print(error.read().decode("utf-8", errors="ignore"))
        return False

    except Exception as error:
        print("Loi khi gui Zalo:", error)
        return False



# KIỂM TRA NHÃN NGUY HIỂM


def is_dangerous_label(label):
    label = label.lower()

    danger_keywords = [
        "fire",
        "smoke",
        "flame",
        "lua",
        "khoi",
        "lửa",
        "khói",
    ]

    return any(keyword in label for keyword in danger_keywords)



# NHẬP THÔNG TIN ZALO


print("Nhap thong tin Zalo Bot.")

BOT_TOKEN = getpass.getpass("Nhap Bot Token: ").strip()
CHAT_ID = input("Nhap CHAT_ID: ").strip()

if not BOT_TOKEN or not CHAT_ID:
    print("Bot Token hoac CHAT_ID dang bi trong.")
    raise SystemExit



# CHỌN VIDEO


window = Tk()
window.withdraw()

video_path = filedialog.askopenfilename(
    title="Chọn video cần nhận diện",
    filetypes=[
        ("Video files", "*.mp4 *.avi *.mov *.mkv"),
        ("All files", "*.*"),
    ],
)

window.destroy()

if not video_path:
    print("Ban chua chon video.")
    raise SystemExit



# TẢI MODEL


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model tai: {MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(exist_ok=True)

output_name = datetime.now().strftime(
    "result_%Y%m%d_%H%M%S.mp4"
)
output_path = OUTPUT_DIR / output_name

print("Dang tai model...")

model = YOLO(str(MODEL_PATH))

print("Cac lop cua model:", model.names)



# MỞ VIDEO


video = cv2.VideoCapture(video_path)

if not video.isOpened():
    raise RuntimeError("Khong mo duoc video da chon.")

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height),
)

if not writer.isOpened():
    video.release()
    raise RuntimeError("Khong tao duoc video ket qua.")



# XỬ LÝ VIDEO


frame_number = 0
consecutive_danger_frames = 0
last_alert_video_second = -ALERT_COOLDOWN_SECONDS

print(f"Video: {video_path}")
print(f"Tong so khung hinh: {total_frames}")
print("Nhan Q de dung som.")

while True:
    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=0.5,
        device="cpu",
        max_det=20,
        verbose=False,
    )

    result = results[0]
    display_frame = result.plot()

    detected_dangers = []
    highest_confidence = 0

    if result.boxes is not None:
        class_ids = result.boxes.cls.cpu().tolist()
        confidence_values = result.boxes.conf.cpu().tolist()

        for class_id, confidence_value in zip(
            class_ids,
            confidence_values,
        ):
            label = str(model.names[int(class_id)])

            if is_dangerous_label(label):
                detected_dangers.append(label)
                highest_confidence = max(
                    highest_confidence,
                    confidence_value,
                )

    if detected_dangers:
        consecutive_danger_frames += 1
    else:
        consecutive_danger_frames = 0

    video_second = video.get(cv2.CAP_PROP_POS_MSEC) / 1000

    can_send_alert = (
        consecutive_danger_frames >= ALERT_AFTER_FRAMES
        and video_second - last_alert_video_second
        >= ALERT_COOLDOWN_SECONDS
    )

    if can_send_alert:
        detected_names = ", ".join(
            sorted(set(detected_dangers))
        )

        video_time = str(
            datetime.utcfromtimestamp(video_second).strftime(
                "%H:%M:%S"
            )
        )

        alert_message = (
            "🔥 CẢNH BÁO CAMERA AI\n"
            f"Phát hiện: {detected_names}\n"
            f"Độ tin cậy cao nhất: "
            f"{highest_confidence * 100:.1f}%\n"
            f"Thời điểm trong video: {video_time}\n"
            f"Video: {Path(video_path).name}"
        )

        sent_successfully = send_zalo_message(
            BOT_TOKEN,
            CHAT_ID,
            alert_message,
        )

        if sent_successfully:
            last_alert_video_second = video_second

        consecutive_danger_frames = 0

    cv2.putText(
        display_frame,
        f"Frame: {frame_number}/{total_frames}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    if detected_dangers:
        cv2.putText(
            display_frame,
            "WARNING: FIRE OR SMOKE",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    writer.write(display_frame)

    cv2.imshow(
        "Fire and Smoke Detection - Press Q to stop",
        display_frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



# GIẢI PHÓNG TÀI NGUYÊN


video.release()
writer.release()
cv2.destroyAllWindows()

print("Da xu ly xong video.")
print(f"Ket qua duoc luu tai: {output_path}") 