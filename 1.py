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
raw_frame = None

from fastapi.responses import HTMLResponse

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

    cv2.imwrite(filepath, raw_frame)   # LƯU ẢNH GỐC
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
    #"nvarguscamerasrc ! "
    "nvarguscamerasrc aeantibanding=3 aelock=false exposuretimerange=\"50000 80000000\" gainrange=\"1 16\" ! "
    "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv noise-reduction=4 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
)
    #gst_str = (
    #"nvarguscamerasrc ! "
    #"video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! "
    #"nvvidconv flip-method=0 ! "
    #"video/x-raw, format=(string)BGRx ! "
    #"videoconvert ! "
    #"video/x-raw, format=(string)BGR ! appsink"
#)
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
    # LƯU ẢNH GỐC TRƯỚC KHI RESIZE
    raw_frame = frame.copy()

    # Ảnh cho AI và web stream
    frame_resized = cv2.resize(frame, (2048, 1152))
    #frame_resized = cv2.resize(frame, (1280, 1280))
    result = model_number1(frame, conf=0.005)
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

    if count_number1 == 3:
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
