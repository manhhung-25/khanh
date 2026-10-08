# Chạy trên Raspberry Pi 4 với camera USB

Script: `camera_pi4.py`. Dùng model `fire.pt` đã huấn luyện, chuyển sang NCNN
để chạy CPU ARM. Không cần huấn luyện lại. Hướng dẫn này dành cho camera USB
Rapoo/UVC, không phải camera CSI của Pi.

## 1. Chuẩn bị Pi

Cài Raspberry Pi OS **64-bit** (Bookworm là một lựa chọn), kết nối Internet
và cắm camera USB. Nên dùng Pi 4 RAM 4 GB trở lên, nguồn đủ công suất và
tản nhiệt. Có thể thao tác bằng terminal trên Pi hoặc SSH từ máy tính.

Kiểm tra kiến trúc:

```bash
uname -m
```

Kết quả phải là `aarch64`. Nếu là `armv7l`, hãy cài OS 64-bit trước.

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip v4l-utils libgl1 libglib2.0-0
sudo timedatectl set-timezone Asia/Ho_Chi_Minh
git clone https://github.com/manhhung-25/khanh.git
cd khanh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir ultralytics ncnn
```

Nếu đã clone trước đó, vào `~/khanh` và chạy `git pull` thay vì clone lại.
Các lệnh Python bên dưới chạy trong thư mục `khanh`, sau khi kích hoạt `.venv`.
Không dùng `sudo pip` hoặc `--break-system-packages`.

Cài PyTorch từ **CPU index** trước để tránh pip tự chọn bản CUDA trên ARM64.
Bộ 2.7.1/0.22.1 có wheel Python 3.13; không cần hạ Python hệ thống của Pi.
Đây là bộ phiên bản đề xuất để kiểm tra tương thích, chưa đo trên Pi của bạn.

Nếu gặp `No space left on device`, kiểm tra trước:

```bash
df -h / /tmp "$HOME"
python -m pip cache purge
```

Nếu `/tmp` hết chỗ nhưng thư mục người dùng còn đủ dung lượng, tạo
`mkdir -p "$HOME/pip-tmp"` và thêm `TMPDIR="$HOME/pip-tmp"` trước mỗi lệnh
`python -m pip install`. `--no-cache-dir` giảm dung lượng cache nhưng vẫn
cần chỗ để tải và giải nén thư viện.

## 2. Xác định camera USB

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-id/
```

Tìm thiết bị `Rapoo camera` và node video tương ứng. Ví dụ bên dưới giả sử
node nhận ảnh là `/dev/video0`; thay bằng node thực tế trên máy của bạn.
Một camera có thể có nhiều node; node metadata không dùng để nhận ảnh.

```bash
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

Khi có nhiều camera, có thể truyền đường dẫn ổn định
`/dev/v4l/by-id/<tên-camera>-video-index0` cho `--camera`.

Nếu bị `Permission denied`:

```bash
sudo usermod -aG video "$USER"
```

Sau đó đăng xuất/đăng nhập lại hoặc khởi động lại Pi.

## 3. Xuất model sang NCNN

Xuất ngay trên Pi, trong môi trường `.venv`:

```bash
yolo export model=fire.pt format=ncnn imgsz=640
```

Lệnh tạo thư mục `fire_ncnn_model/`. Chờ export hoàn tất; không cần training.
Lần đầu có thể phải tải thêm công cụ export. Giữ file `fire.pt` để xuất lại.

Nếu gặp `Illegal instruction` khi import/export, chưa có đủ dữ liệu để
khẳng định thư viện nào bị lỗi. Tuy nhiên, đây thường là mã máy không phù
hợp CPU. Log `torch-2.14.1+cu130` trên Pi 4 là lý do để thay bộ PyTorch trước.
PyTorch 2.7 có bản sửa tương thích ARMv8-A/Pi 4; có thể thử CPU 2.7.1 với
torchvision 0.22.1 trong chính `.venv` hiện tại, kể cả **Python 3.13**:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
mkdir -p "$HOME/pip-tmp"
TMPDIR="$HOME/pip-tmp" python -m pip install --no-cache-dir "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cpu
TMPDIR="$HOME/pip-tmp" python -m pip install --no-cache-dir ultralytics ncnn
```

Kiểm tra phiên bản, convolution CPU và toán tử NMS trước khi export:

```bash
python - <<'PY'
import torch
import torchvision
print('torch:', torch.__version__, 'torchvision:', torchvision.__version__)
print('CUDA build:', torch.version.cuda)
print('Conv:', torch.nn.Conv2d(3, 8, 3)(torch.rand(1, 3, 32, 32)).shape)
print('NMS:', torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), 0.5))
PY
```

Mong đợi torch `2.7.1+cpu`, `CUDA build: None`, và hai phép tính hoàn tất.
Nếu vẫn `Illegal instruction`, dừng ở bước này và gửi output cùng
`python --version`, `uname -m`, `python -m pip freeze` để kiểm tra tiếp.
Đừng cài torch 2.5.1 bằng Python 3.13 ARM64; wheel CPU ARM64 của phiên bản
đó chỉ có đến Python 3.12. Khi test thành công, chạy lại:

```bash
yolo export model=fire.pt format=ncnn imgsz=640
```

Chỉ chạy `camera_pi4.py` sau khi export thành công. `FileNotFoundError` cho
`fire_ncnn_model` sau lần export bị dừng là hệ quả chưa có model NCNN.

## 4. Test camera và đo FPS trước

```bash
python camera_pi4.py --camera /dev/video0 --no-zalo --max-frames 60
```

Lệnh chạy model NCNN mặc định, in FPS và dừng sau 60 khung hình.
Chế độ `--no-zalo` không đọc cấu hình Zalo, không upload ảnh, không gửi tin nhắn.
Lần suy luận đầu có thời gian khởi tạo; xem FPS sau khi chương trình đã chạy
một lúc. FPS bao gồm đọc camera và xử lý model, không chỉ inference.
Log còn có `read_ms` (đọc camera), `predict_ms` (toàn bộ lời gọi YOLO, gồm
tiền xử lý/suy luận/hậu xử lý), và `other_ms` (vẽ, xác minh và phần còn lại).
Các số là trung bình tối đa 20 khung gần nhất; lúc khởi động còn bao gồm tải
backend. Khi dùng NCNN, script giới hạn luồng phụ của OpenCV/PyTorch ở 1;
NCNN tự quản lý luồng suy luận.

Muốn test `.pt` trước khi export (chậm hơn):

```bash
python camera_pi4.py --model fire.pt --camera /dev/video0 --no-zalo --max-frames 30
```

Nếu dùng Desktop và màn hình nối trực tiếp với Pi, thêm `--show` để xem ảnh.
Khi SSH không có màn hình, chạy mặc định. Nhấn `Ctrl+C` để dừng; chế độ có
cửa sổ còn hỗ trợ phím Q.

## 5. Cấu hình Zalo và chạy thật

Chỉ sao chép file mẫu nếu chưa có `zalo_config.json` trên Pi:

```bash
cp zalo_config.example.json zalo_config.json
nano zalo_config.json
```

Điền `bot_token`, `chat_id`, `imgbb_api_key`. Dùng thông tin còn hiệu lực;
file cấu hình thật không được đưa lên GitHub. Trong nano: Ctrl+O, Enter để
lưu, Ctrl+X để thoát.

```bash
python camera_pi4.py --camera /dev/video0
```

Mặc định: mọi lớp của model đủ điều kiện, confidence >= 0,60, hit >= 80%,
mean >= 0,65 (kể cả mẫu bỏ sót), tối thiểu 5 mẫu trong cửa sổ **5 giây**.
Cửa sổ dài hơn bản Windows (2 giây) để phù hợp CPU chậm; cảnh báo cũng đến
muộn hơn. Ảnh được lưu ở `outputs/alerts`, upload ImgBB rồi gửi Zalo.
Khoảng chờ gửi và thời hạn link ảnh dùng chung với bản Windows.

Pi báo qua terminal và Zalo. Script chưa điều khiển còi GPIO; `winsound`
chỉ dùng ở bản Windows. Khi dừng, chương trình chờ hoàn tất lần gửi đang chạy.

## 6. Nếu còn chậm

Đo thực tế trên Pi của bạn với YOLO11n NCNN 640: khoảng **2,4 FPS** sau khi
khởi động. Dòng 0,23 FPS đầu tiên có cả chi phí tải backend. Zalo đang OFF
trong phép đo này, nên phần upload không gây chậm.

Thử 416 trước khi giảm xuống 320. Mỗi bản được xuất với tên riêng để có thể
so sánh cùng một cảnh với bản 640 đang chạy được:

```bash
cp fire.pt fire_416.pt
yolo export model=fire_416.pt format=ncnn imgsz=416
python camera_pi4.py --model fire_416_ncnn_model --camera /dev/video0 --imgsz 416 --no-zalo --max-frames 60
```

Nếu cần thử 320:

```bash
cp fire.pt fire_320.pt
yolo export model=fire_320.pt format=ncnn imgsz=320
python camera_pi4.py --model fire_320_ncnn_model --camera /dev/video0 --imgsz 320 --no-zalo --max-frames 60
```

416 và 320 có ít pixel hơn nên có thể xử lý nhanh hơn; FPS cụ thể cần đo lại
trên Pi. Giảm kích thước có thể làm bỏ sót lửa nhỏ/xa. Đối chiếu các mức bằng
cùng một cảnh trước khi chọn. Khi chạy cảnh báo thật, bỏ hai tham số
`--no-zalo --max-frames 60`, giữ đúng `--model` và `--imgsz` đã test.

Nếu FPS thấp đến mức không đủ 5 mẫu, tăng thời gian xác minh (cảnh báo sẽ chậm hơn):

```bash
python camera_pi4.py --camera /dev/video0 --window 8
```

Lệnh này giả sử model NCNN đang xuất ở 640; thêm `--imgsz` tương ứng nếu đã
xuất ở 416/320. `--window` không thay đổi ngưỡng confidence/hit/mean.

Kiểm tra nhiệt độ và dấu hiệu throttling bằng:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
```

## Chạy lại sau khi khởi động Pi

```bash
cd ~/khanh
source .venv/bin/activate
python camera_pi4.py --camera /dev/video0
```

## Nguồn và phạm vi kiểm tra

- [Ultralytics: Raspberry Pi và NCNN](https://docs.ultralytics.com/guides/raspberry-pi/)
- [Raspberry Pi: Python và virtual environment](https://www.raspberrypi.com/documentation/computers/os.html#python-on-raspberry-pi)
- [PyTorch: lỗi export trên Pi 4 ở phiên bản 2.6.0](https://github.com/pytorch/pytorch/issues/146792)
- [PyTorch: cặp phiên bản 2.7.1/0.22.1 và CPU index](https://pytorch.org/get-started/previous-versions/#v271)
- [PyTorch 2.7: sửa tương thích ARMv8-A/Pi 4](https://github.com/pytorch/pytorch/releases/tag/v2.7.0)

Script đã kiểm tra cú pháp và luồng chạy với camera/model mô phỏng trên máy
phát triển. Chưa đo camera, export NCNN, FPS hoặc hiển thị Zalo trên Pi 4 thật;
làm bước test `--no-zalo` trước để xác nhận bộ phần mềm và camera của bạn.
