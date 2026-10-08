"""Local realtime model preview. Q quits; S saves a raw frame for review."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from camera_fire_temporal_zalo import MODEL_PATH, IMAGE_SIZE, CONFIDENCE
from camera_source import CAMERA_NAME, find_camera_index


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    import cv2
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--camera", type=int, default=None,
                        help="Override the default Rapoo USB webcam with a DirectShow index")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Khong tim thay model: {args.model}")

    model = YOLO(str(args.model))
    print("Model:", args.model, "| Classes:", model.names, flush=True)
    camera_index = find_camera_index() if args.camera is None else args.camera
    camera_name = CAMERA_NAME if args.camera is None else f"Camera {camera_index}"
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    title = f"Realtime model test - {camera_name} - {args.model.name} - S: save - Q: quit"
    save_dir = PROJECT_DIR / "data_collect" / "model_test" / "images"
    previous_tick = None
    smoothed_fps = None
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Khong mo duoc {camera_name} (index={camera_index}).")
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, 960, 720)
        printed_size = False
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Khong doc duoc anh webcam.")
            captured_time = datetime.now().astimezone()
            result = model.predict(
                frame, imgsz=args.imgsz, conf=CONFIDENCE, iou=0.5,
                device="cpu", max_det=20, verbose=False,
            )[0]
            tick = time.perf_counter()
            if previous_tick is not None:
                current_fps = 1.0 / max(tick - previous_tick, 1e-9)
                smoothed_fps = current_fps if smoothed_fps is None else 0.9 * smoothed_fps + 0.1 * current_fps
            previous_tick = tick
            display = result.plot()
            fps_text = "warming up" if smoothed_fps is None else f"{smoothed_fps:.1f} FPS"
            cv2.putText(display, f"{args.model.name} | CPU {fps_text} | S: save | Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(display, "MODEL PREVIEW - class IDs from checkpoint",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow(title, display)
            if not printed_size:
                print(f"REALTIME_READY: {frame.shape[1]}x{frame.shape[0]}, imgsz={args.imgsz}", flush=True)
                printed_size = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                save_dir.mkdir(parents=True, exist_ok=True)
                stem = captured_time.strftime("test_%Y%m%d_%H%M%S_%f")
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if not encoded_ok:
                    raise RuntimeError("Khong ma hoa duoc anh webcam.")
                destination = save_dir / f"{stem}.jpg"
                destination.write_bytes(encoded.tobytes())
                (save_dir / f"{stem}.json").write_text(json.dumps({
                    "model": args.model.name,
                    "classes": model.names,
                    "captured_at": captured_time.isoformat(),
                    "detections_xyxy_conf_class": result.boxes.data.cpu().tolist(),
                }, indent=2), encoding="utf-8")
                print("Da luu anh goc va du doan:", destination, flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("REALTIME_CLOSED", flush=True)


if __name__ == "__main__":
    main()
