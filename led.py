import os
# QUAN TRỌNG: Khai báo model ngay từ đầu
os.environ['JETSON_MODEL_NAME'] = 'JETSON_ORIN_NANO'

import Jetson.GPIO as GPIO
import time

# Sử dụng cách đánh số chân vật lý 1-40
GPIO.setmode(GPIO.BOARD)

# Ví dụ: Chân số 7 trên header
led_pin = 13
GPIO.setup(led_pin, GPIO.OUT)

print("Bắt đầu nhấp nháy LED. Nhấn CTRL+C để dừng.")
try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH) # Bật LED (3.3V)
        print("LED ON")
        time.sleep(20)
        GPIO.output(led_pin, GPIO.LOW)  # Tắt LED (0V)
        print("LED OFF")
        time.sleep(20)
except KeyboardInterrupt:
    # Dọn dẹp GPIO trước khi thoát
    GPIO.cleanup()
    print("Đã dọn dẹp GPIO và thoát chương trình.")
