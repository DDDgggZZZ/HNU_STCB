import cv2
import os
import time

url = 'http://192.168.5.1:81/stream'  # 你的 ESP32-CAM 流地址
save_dir = 'captured_images'
os.makedirs(save_dir, exist_ok=True)

# 尝试指定 FFmpeg 后端（可选，Windows/Linux 如果默认不行可以打开）
# cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Could not open video stream. Try using CAP_FFMPEG or check the URL.")
    exit(1)

cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Stream', 960, 540)

frame_count = 0
last_saved = None

print("Instructions:")
print("- Focus the video window and press 's' or Space to save a frame.")
print("- Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Warning: Failed to retrieve frame. Retrying...")
        # 给摄像头一点时间恢复
        time.sleep(0.05)
        continue

    cv2.imshow('Camera Stream', frame)

    # waitKey 设为 10~30ms 比 1 更稳
    key = cv2.waitKey(20) & 0xFF
    if key in (ord('s'), 32):  # 's' 或 空格保存
        frame_count += 1
        filename = os.path.join(save_dir, f'capture_{frame_count:04d}.jpg')
        ok = cv2.imwrite(filename, frame)
        if ok:
            last_saved = filename
            print(f"Saved: {filename}")
        else:
            print(f"Error: Failed to save image to {filename}. Check permissions/disk space/path.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()