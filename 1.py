import cv2
from ultralytics import YOLO
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import gpiod

# ==========================
#  GPIO (libgpiod) SETUP
# ==========================
chip = gpiod.Chip('gpiochip0') 
line = chip.get_line(11)   # <-- THAY SỐ "36" thành GPIO line bạn kiểm tra được
line.request(consumer="yolo_out", type=gpiod.LINE_REQ_DIR_OUT)

def gpio_set(state):
    line.set_value(1 if state else 0)

# ==========================
# YOLO MODEL
# ==========================
model_number1 = YOLO("yolov11_last.engine")
NUMBER1_CLASS_NAME = "Eye"

# ===== FLAGS =====
detecting = True
show_window = True

# ===== STREAM =====
output_frame = None
frame_lock = threading.Lock()

app = FastAPI()

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video")
def video_stream():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ===== CAMERA =====
def open_camera():
    global cap
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
        print("❌ Không thể mở camera!")
        exit()

open_camera()

threading.Thread(target=run_server, daemon=True).start()
print("🚀 Stream running at http://<JETSON_IP>:8000/video")

# ===== MAIN LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 640))
    result = model_number1(frame_resized, conf=0.005)
    boxes = result[0].boxes

    count_number1 = 0
    for box, cls, conf in zip(boxes.xyxy.int().tolist(),
                              boxes.cls.tolist(),
                              boxes.conf.tolist()):
        label = result[0].names[int(cls)]
        if label == NUMBER1_CLASS_NAME:
            count_number1 += 1
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(frame_resized,
                        f"{label}:{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 128, 255), 2)

    # Display count + OK/NG
    cv2.putText(frame_resized,
                f"Count: {count_number1}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    if count_number1 >= 1:
        cv2.putText(frame_resized,
                    "OK", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 255, 0), 3)
        gpio_set(False)  # OFF
        print("GPIO set to LOW:", line.get_value())
    else:
        cv2.putText(frame_resized,
                    "NG", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)
        gpio_set(True)   # ON
        print("GPIO set to HIGH:", line.get_value())

    with frame_lock:
        output_frame = frame_resized.copy()

    if show_window:
        try:
            cv2.imshow("Detection", frame_resized)
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
