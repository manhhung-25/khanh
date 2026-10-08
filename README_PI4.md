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
python -m pip install ultralytics ncnn
```

Nếu đã clone trước đó, vào `~/khanh` và chạy `git pull` thay vì clone lại.
Các lệnh Python bên dưới chạy trong thư mục `khanh`, sau khi kích hoạt `.venv`.
Không dùng `sudo pip` hoặc `--break-system-packages`.

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

Nếu gặp `Illegal instruction (core dumped)` khi import/export, có thể là
wheel thư viện không phù hợp CPU. PyTorch từng ghi nhận lỗi export trên
Pi 4 với 2.6.0, trong khi 2.5.1 chạy được ở ca lỗi đó. Với **Bookworm,
Python 3.11, aarch64**, có thể thử bộ phiên bản cũ sau trong `.venv`:

```bash
python -m pip install --force-reinstall "numpy<2" "opencv-python<4.12" "torch==2.5.1" "torchvision==0.20.1" "ultralytics==8.3.70" ncnn
```

Đây là phương án xử lý lỗi tương thích, chưa được đo với phần cứng của bạn.
Nếu vẫn lỗi, lưu kết quả `python --version`, `uname -m` và `python -m pip freeze`
để kiểm tra đúng bộ thư viện, không tăng epoch để xử lý lỗi này.

## 4. Test camera và đo FPS trước

```bash
python camera_pi4.py --camera /dev/video0 --no-zalo --max-frames 60
```

Lệnh chạy model NCNN mặc định, in FPS và dừng sau 60 khung hình.
Chế độ `--no-zalo` không đọc cấu hình Zalo, không upload ảnh, không gửi tin nhắn.
Lần suy luận đầu có thời gian khởi tạo; xem FPS sau khi chương trình đã chạy
một lúc. FPS bao gồm đọc camera và xử lý model, không chỉ inference.

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

Thử 416 trước khi giảm xuống 320. Cần xuất model lại ở kích thước sẽ chạy;
lệnh dưới cập nhật thư mục NCNN đã xuất:

```bash
yolo export model=fire.pt format=ncnn imgsz=416
python camera_pi4.py --camera /dev/video0 --imgsz 416 --no-zalo --max-frames 60
```

Hoặc thay cả hai số 416 bằng 320. Sau khi test, bỏ `--no-zalo --max-frames 60`
để chạy cảnh báo thật. Giảm kích thước có thể làm bỏ sót lửa nhỏ/xa; so sánh
với 640 bằng cùng một cảnh trước khi chọn.

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

Script đã kiểm tra cú pháp và luồng chạy với camera/model mô phỏng trên máy
phát triển. Chưa đo camera, export NCNN, FPS hoặc hiển thị Zalo trên Pi 4 thật;
làm bước test `--no-zalo` trước để xác nhận bộ phần mềm và camera của bạn.
