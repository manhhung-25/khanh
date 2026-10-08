![Fire Detection Demo](https://github.com/luminous0219/fire-and-smoke-detection-yolov8/blob/main/fire%20demo%201.gif)
![Fire Detection Demo](https://github.com/luminous0219/fire-and-smoke-detection-yolov8/blob/main/fire%20demo%202.gif)

**Fire and Smoke Detection Model**

This repository contains a trained model for fire and smoke detection using the Ultralytics YOLOv8 architecture. The model has been fine-tuned to detect fire and smoke in video files and images with high accuracy and can be integrated into various applications such as surveillance systems, disaster monitoring, and safety systems.

**Model Information**

Architecture: YOLOv8n (Nano) - optimized for high speed and efficiency
Original repository dataset reference: https://universe.roboflow.com/fire-rqbio/fire-and-smoke-yikzn

The local `weights/best.pt` checkpoint records a separate training data path;
that dataset is not included here.

Detection Capabilities:
Fire
Smoke

**Features**
Real-time fire and smoke detection
Lightweight and efficient, suitable for deployment on edge devices
Can be used with video streams or image inputs
Configurable confidence and IoU thresholds for fine-tuning detection sensitivity

Usage
1. Loading the Model
You can load the pre-trained model using Ultralytics YOLOv8 library or PyTorch:

from ultralytics import YOLO

# Load the custom-trained fire and smoke model
model = YOLO("best.pt")

2. Running Inference on an Image or Video
To run inference on an image or video, use the following code:

Inference on an image
results = model("path_to_image.jpg")
results.show()

Inference on a video file
results = model("path_to_video.mp4")
results.show()

3. Deployment
The model can be deployed in various environments such as:

Web applications: Integrated with Gradio or Streamlit for creating an interactive web interface.
Mobile apps: Utilize the model with mobile cameras for fire and smoke detection.
Edge devices: Deploy the model on devices with limited resources for real-time detection.

## Lửa nhỏ hoặc ở xa (webcam và video Zalo)

Trước khi chạy, sao chép `zalo_config.example.json` thành `zalo_config.json`
và điền `bot_token`, `chat_id`, `imgbb_api_key` trên máy của bạn. File cấu hình
thật được bỏ qua bởi Git; không đưa token hoặc API key lên GitHub.

Chạy `python camera_fire_temporal_zalo.py` cho webcam hoặc
`python video_fire_temporal_zalo.py` cho video. Script webcam dùng `fire.pt`
(YOLO11n, kích thước huấn luyện 640) và bật cảnh báo Zalo cho **tất cả các lớp**,
kể cả tên lớp dạng số 0–4. Script video vẫn dùng `weights/best.pt`; checkpoint
này ghi lần huấn luyện ở 320, 20 epoch, khác với file `args.yaml` (640/150).
Hai script camera mặc định dùng webcam USB ngoài **Rapoo camera**, chọn theo tên
thiết bị DirectShow bằng `ffmpeg` (cần có trong PATH, máy hiện tại đã có).
Tên camera mặc định nằm ở `CAMERA_NAME` trong `camera_source.py`. Nếu không tìm
thấy thiết bị, script báo lỗi. Cửa sổ ghi tên webcam để kiểm tra nguồn hình.
Webcam yêu cầu 640×480 và in kích thước thực tế. Cửa sổ test có thể chọn camera
khác bằng `--camera <số>` nếu cần.

Webcam: suy luận ở confidence 0,25 như cửa sổ test. Để gửi Zalo, cùng một vùng
phải có confidence từ 0,60, được quan sát ít nhất 2 giây với tối thiểu 5 mẫu,
xuất hiện trong ít nhất 80% khung hình và đạt điểm trung bình 0,65 (tính cả
khung hình bỏ sót). Màn hình hiển thị “DANG XAC MINH” trước khi đủ điều kiện.
Ảnh chỉ vẽ khung, không hiện tên lớp/ID. Tin nhắn ghi “NGUY HIỂM”, thời gian
chụp và ảnh camera. Mỗi lần gửi cách nhau ít nhất 30 giây.

Ảnh được tải lên ImgBB và link tự xóa sau `PHOTO_EXPIRATION_SECONDS` (mặc định
1 giờ). Script kiểm tra link trả về JPEG giải mã được trước khi gửi Zalo,
kèm link mở trực tiếp trong chú thích. Mỗi ảnh trong `outputs/alerts` có file
JSON cùng tên chứa URL, thời điểm hết hạn dự kiến và trạng thái Zalo chấp nhận.
Trạng thái `SENT` nghĩa là API đã chấp nhận tin nhắn; không xác nhận ảnh đã
hiển thị trên điện thoại. Nếu ảnh trắng, thử link trực tiếp khi còn hạn để
phân biệt lỗi tải ảnh ImgBB với lỗi hiển thị trong Zalo.

`camera_test_model.py` dùng chung model mặc định, kích thước suy luận và
confidence với script webcam; cửa sổ test hiển thị các ứng viên và không gửi
Zalo. Các ngưỡng gửi nằm ở đầu `camera_fire_temporal_zalo.py`. Bộ lọc này giảm
phát hiện yếu/thoáng qua; dự đoán nhầm có điểm cao kéo dài vẫn có thể qua lọc.

Độ phân giải và ngưỡng không thể tạo lại chi tiết đã mất. Để đo độ chính xác
ở khoảng cách sử dụng thực tế, dùng phím S trong **script webcam** để lưu ảnh
gốc; thu thập cả ảnh lửa nhỏ/xa và ảnh không có lửa (đèn, phản chiếu, da
người). Gán nhãn, đánh giá và huấn luyện lại model nếu vẫn bỏ sót. Theo dõi
cả số lần bỏ sót và báo nhầm trước khi dùng cho cảnh báo an toàn cháy nổ.
