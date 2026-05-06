import cv2
from ultralytics import YOLO
import threading
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.responses import HTMLResponse
import os
os.environ['JETSON_MODEL_NAME'] = 'JETSON_ORIN_NANO'

import Jetson.GPIO as GPIO
import time
# Khai báo chân vật lý
GPIO.setmode(GPIO.BOARD)

# Chọn chân output
led_pin = 13
GPIO.setup(led_pin, GPIO.OUT)

# ==========================
# YOLO MODEL, đăt theo tên train
# ==========================
model_number1 = YOLO("20.engine")
NUMBER1_CLASS_NAME = "OK"

# ===== FLAGS =====
detecting = True
show_window = True

# ===== STREAM =====
output_frame = None
frame_lock = threading.Lock()

app = FastAPI()
raw_frame = None

# ===== DEBOUNCE VARIABLES =====
stable_counter = 0
stable_count = -1  # -1: NG, 3: OK
STABLE_FRAMES = 2  # Cần 5 frame liên tiếp để xác nhận(1=>5)

@app.get("/capture")
def capture_image():
    global raw_frame
    import time, os

    if raw_frame is None:
        return {"error": "no raw frame"}

    save_dir = "captures"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"capture_{int(time.time())}.jpg"
    filepath = os.path.join(save_dir, filename)

    cv2.imwrite(filepath, raw_frame)
    print("📸 Saved:", filepath)

    return {"status": "saved", "file": filepath}

@app.get("/")
def home():
    html = """
    <html>
    <head>
        <title>YOLO Stream</title>
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px 30px;
                background: #000;
                border-bottom: 1px solid #333;
            }
            .header h1 {
                margin: 0;
                font-size: 24px;
            }
            .header button {
                padding: 10px 20px;
                font-size: 18px;
                background: #28a745;
                border: none;
                color: white;
                cursor: pointer;
                border-radius: 5px;
            }
            .header button:hover {
                background: #218838;
            }
            .video-container {
                text-align: center;
                margin-top: 15px;
            }
            #video-stream {
                width: 90vw;
                max-width: 1728px;
                aspect-ratio: 16/9;
                object-fit: contain;
                border: 2px solid #444;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Live Stream</h1>
            <button onclick="capture()">📸 Capture Image</button>
        </div>

        <div class="video-container">
            <img id="video-stream" src="/video" />
        </div>

        <script>
        function capture() {
            fetch('/capture')
                .then(r => r.json())
                .then(data => alert('Saved: ' + data.file))
                .catch(err => alert('Error: ' + err));
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)
    
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
    "nvarguscamerasrc aeantibanding=3 aelock=false exposuretimerange=\"50000 80000000\" gainrange=\"1 16\" ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv noise-reduction=4 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=(string)BGRx ! "
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
    
    raw_frame = frame.copy()
    frame_resized = cv2.resize(frame, (2048, 1152))
    result = model_number1(frame_resized, conf=0.5)
    
    if result[0].boxes is not None and len(result[0].boxes) > 0:
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
    else:
        count_number1 = 0

    cv2.putText(frame_resized,
                f"Count: {count_number1}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # ===== DEBOUNCE LOGIC =====
    current_state = "OK" if count_number1 == 3 else "NG"

    # smoothing: tăng khi OK, giảm khi NG
    if current_state == "OK":
        stable_counter = min(STABLE_FRAMES, stable_counter + 1)
    else:
        stable_counter = max(-STABLE_FRAMES, stable_counter - 1)

    # chuyển sang OK
    if stable_counter == STABLE_FRAMES and stable_count != 3:
        stable_count = 3
        GPIO.output(led_pin, GPIO.LOW)
        print("✅ STABLE OK → GPIO LOW")

    # chuyển sang NG
    if stable_counter == -STABLE_FRAMES and stable_count != -1:
        stable_count = -1
        GPIO.output(led_pin, GPIO.HIGH)
        print("❌ STABLE NG → GPIO HIGH")

    # ===== HIỂN THỊ OK/NG TRÊN WEB =====
    if stable_count == 3:
        cv2.putText(frame_resized, "OK", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame_resized, "NG", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    
    with frame_lock:
        output_frame = frame_resized.copy()

    if show_window:
        try:
            cv2.imshow("Detection", frame_resized)
        except:
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

gpio_set(False)
line.release()
cap.release()
cv2.destroyAllWindows()
