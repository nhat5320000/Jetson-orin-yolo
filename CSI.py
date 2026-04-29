import cv2
import numpy as np
from ultralytics import YOLO
import random

# Đường dẫn đến mô hình YOLO
model_path = "yolo26s.engine"
object_detect = YOLO(model_path, task='detect')

# Tên các lớp và màu sắc
class_names = {0: "Object1", 1: "Object2", 2: "Object3", 3: "Object4", 4: "Object5"}
color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (0, 255, 255)}

# Biến điều khiển
detecting = True
show_window = True

def open_camera():
    global cap
    # Cấu hình GStreamer pipeline cho CSI camera
    gst_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)640, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Không thể kết nối CSI camera!")
        exit()

open_camera()  # Mở camera ngay khi khởi động

# Tạo cửa sổ điều khiển
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 300, 100)
cv2.imshow('Control Panel', np.zeros((100, 300, 3), dtype=np.uint8))

while True:
    if detecting:
        success, frame = cap.read()
        if not success:
            print("❌ Lỗi đọc frame từ camera!")
            break

        # Xử lý frame và detect
        frame_resized = cv2.resize(frame, (640, 640))  # Đảm bảo đầu vào mô hình
        results = object_detect(frame_resized, conf=0.11, imgsz=640)
        detections = results[0].boxes

        # Vẽ bounding boxes
        for box, cls, conf in zip(detections.xyxy.int().tolist(), detections.cls.tolist(), detections.conf.tolist()):
            x1, y1, x2, y2 = box
            color = color_map.get(int(cls), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị cửa sổ
        if show_window:
            cv2.imshow("Object Detection", frame_resized)
            #if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                #show_window = False
        else:
            try: cv2.destroyWindow("Object Detection")
            except: pass

    # Xử lý phím điều khiển
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        print("🛑 Thoát chương trình...")
        break
    elif key in [ord('e'), ord('E')]:
        if not detecting:
            open_camera()
            detecting = True
            print("📸 Camera và phát hiện đối tượng BẬT")
    elif key in [ord('r'), ord('R')]:
        if detecting:
            detecting = False
            cap.release()
            print("📴 Camera và phát hiện đối tượng TẮT")
    elif key in [ord('w'), ord('W')]:
        show_window = not show_window
        print(f"🔧 Trạng thái hiển thị: {'BẬT' if show_window else 'TẮT'}")

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
