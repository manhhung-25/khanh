"""Find the selected webcam by its Windows DirectShow device name."""

import re
import shutil
import subprocess


CAMERA_NAME = "Rapoo camera"


def find_camera_index():
    """Return the selected device's current OpenCV CAP_DSHOW index."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Can ffmpeg trong PATH de tim webcam theo ten.")
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Khong liet ke duoc camera DirectShow.") from error
    # The dummy input normally fails; the device list is printed to stderr.
    names = re.findall(r'"([^"\r\n]+)" \(video\)', result.stderr)
    for index, name in enumerate(names):
        if name.casefold() == CAMERA_NAME.casefold():
            print(f"Camera: {name} | DirectShow index={index}", flush=True)
            return index
    raise RuntimeError(
        f"Khong tim thay {CAMERA_NAME}. Kiem tra ket noi va quyen Camera cua Windows. "
        f"Camera hien co: {', '.join(names) or '(khong co)'}"
    )
