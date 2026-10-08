"""Raspberry Pi/Linux USB camera: temporal fire detection and optional Zalo alerts."""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import signal
import sys
import time

import camera_fire_temporal_zalo as alerts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=alerts.PROJECT_DIR / "fire_ncnn_model")
    parser.add_argument("--camera", default="/dev/video0",
                        help="USB camera device path, or a numeric index")
    parser.add_argument("--imgsz", type=int, choices=(320, 416, 640), default=640)
    parser.add_argument("--window", type=float, default=5.0,
                        help="Confirmation window in seconds (default: 5 for a slower CPU)")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV preview on a desktop")
    parser.add_argument("--no-zalo", action="store_true", help="Local test without uploads or messages")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames; 0 runs until Ctrl+C")
    args = parser.parse_args(argv)
    if args.window <= 0 or args.max_frames < 0:
        parser.error("--window must be positive; --max-frames must be nonnegative")
    return args


def stop_requested(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    args = parse_args(argv)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Script nay dung tren Raspberry Pi/Linux. Windows dung camera_fire_temporal_zalo.py.")
    if not args.model.exists():
        raise FileNotFoundError(f"Khong tim thay model: {args.model}. Xuat fire.pt sang NCNN truoc.")

    import cv2
    from ultralytics import YOLO

    # NCNN manages inference threads itself. Avoid extra OpenCV/PyTorch pools
    # for the small preprocessing/postprocessing tasks on a four-core Pi.
    cv2.setNumThreads(1)
    if args.model.is_dir():
        import torch
        torch.set_num_threads(1)

    # This process shares the verification/sending code, with a longer Pi window.
    alerts.WINDOW_SECONDS = args.window
    alerts.MAX_FRAME_GAP = args.window
    model = YOLO(str(args.model), task="detect")
    sender = None if args.no_zalo else alerts.ZaloAlertSender(alerts.load_zalo_config())
    source = int(args.camera) if args.camera.isdecimal() else args.camera
    camera = cv2.VideoCapture(source, cv2.CAP_V4L2)
    title = "Raspberry Pi - Fire Detection - Q: quit"
    frame_count = 0
    durations = deque(maxlen=20)
    fps = 0.0
    last_log = float("-inf")
    verifier = alerts.TemporalVerifier()
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Khong mo duoc {args.camera}. Kiem tra v4l2-ctl --list-devices va quyen video.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, alerts.CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, alerts.CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Model: {args.model}; imgsz={args.imgsz}; camera={args.camera}", flush=True)
        print(f"Xac minh {args.window:g}s, >= {alerts.MIN_SAMPLES} mau, "
              f"confidence >= {alerts.ALERT_CONFIDENCE}, hit >= {alerts.MIN_HIT_RATIO:.0%}, "
              f"mean >= {alerts.MIN_MEAN_CONFIDENCE}. Zalo: {'OFF' if sender is None else 'ON'}", flush=True)
        print("Nhan Ctrl+C de dung." if not args.show else "Nhan Q hoac Ctrl+C de dung.", flush=True)
        while args.max_frames == 0 or frame_count < args.max_frames:
            started = time.monotonic()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Mat hinh tu camera USB.")
            captured_at = time.monotonic()
            captured_time = datetime.now().astimezone()
            if frame_count == 0:
                print(f"Camera thuc te: {frame.shape[1]}x{frame.shape[0]}", flush=True)
            predict_started = time.monotonic()
            result = model.predict(
                source=frame, imgsz=args.imgsz, conf=alerts.CONFIDENCE,
                iou=0.5, device="cpu", max_det=20, verbose=False,
            )[0]
            predict_seconds = time.monotonic() - predict_started
            detections = [] if result.boxes is None else [
                (int(class_id), score, (x1, y1, x2, y2))
                for x1, y1, x2, y2, score, class_id in result.boxes.data.cpu().tolist()
            ]
            states = verifier.update(detections, captured_at)
            confirmed = any(state["alarm"] for state in states)
            display = None
            if args.show or (confirmed and sender is not None):
                display = result.plot(labels=False, conf=False)
                status = "NGUY HIEM: LUA/KHOI!" if confirmed else "DANG GIAM SAT / XAC MINH"
                color = (0, 0, 255) if confirmed else (0, 255, 255)
                cv2.putText(display, f"Pi CPU: {fps:.1f} FPS | {status}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(display, captured_time.strftime("%d/%m/%Y %H:%M:%S"),
                            (10, display.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2)
                if sender is not None:
                    cv2.putText(display, sender.overlay_status(), (10, display.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if confirmed and sender is not None:
                sender.try_submit(display, captured_time)
            if args.show:
                cv2.imshow(title, display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            frame_count += 1
            durations.append((time.monotonic() - started,
                              captured_at - started, predict_seconds))
            total_seconds = sum(sample[0] for sample in durations)
            fps = len(durations) / max(total_seconds, 1e-9)
            now = time.monotonic()
            if now - last_log >= 5:
                read_ms = 1000 * sum(sample[1] for sample in durations) / len(durations)
                predict_ms = 1000 * sum(sample[2] for sample in durations) / len(durations)
                other_ms = max(0.0, 1000 * total_seconds / len(durations) - read_ms - predict_ms)
                print(f"FPS={fps:.2f} | read_ms={read_ms:.1f} | predict_ms={predict_ms:.1f} | "
                      f"other_ms={other_ms:.1f} | regions={len(states)} | alarm={confirmed} | "
                      f"{sender.overlay_status() if sender else 'Zalo: OFF'}", flush=True)
                last_log = now
    except KeyboardInterrupt:
        print("Dang dung camera...", flush=True)
    finally:
        camera.release()
        if args.show:
            cv2.destroyAllWindows()
        if sender is not None and sender.worker is not None and sender.worker.is_alive():
            print("Dang hoan tat lan gui Zalo cuoi...", flush=True)
            sender.worker.join()
        print(f"Da dung. Frames={frame_count}; FPS gan nhat={fps:.2f}", flush=True)


if __name__ == "__main__":
    main()
