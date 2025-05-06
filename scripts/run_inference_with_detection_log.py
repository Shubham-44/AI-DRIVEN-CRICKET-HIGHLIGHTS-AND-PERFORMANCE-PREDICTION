import cv2
import os
from ultralytics import YOLO

# --- CONFIG ---
VIDEO_PATH = 'data/videos/WIVsIND.mp4'
MODEL_PATH = 'runs/weights/best.pt'
OUTPUT_FOLDER = 'data/detected_frames'
PROGRESS_FILE = 'data/logs/frame_log.txt'
LOG_FILE = 'data/logs/detection_log.txt'

CONF_THRESHOLD = 0.5
MAX_FRAMES = 200
SKIP_TIME_SEC = 120
VALID_CLASSES = {
    0: 'catch',
    1: 'boundary',
    2: 'celebration',
    3: 'wicket'
}
# --------------

start_frame = 0
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        try:
            start_frame = int(f.read().strip())
        except:
            pass

model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if start_frame == 0:
    cap.set(cv2.CAP_PROP_POS_MSEC, SKIP_TIME_SEC * 1000)
else:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
saved = 0
log_file = open(LOG_FILE, 'a')

while cap.isOpened() and saved < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or unreadable frame.")
        break

    print(f"▶️ Reading frame {frame_id}")
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if boxes:
        for cls, conf in zip(boxes.cls, boxes.conf):
            cls_id = int(cls)
            if cls_id in VALID_CLASSES and conf >= CONF_THRESHOLD:
                class_name = VALID_CLASSES[cls_id]
                frame_filename = f"{class_name}_frame_{frame_id:05d}.jpg"
                frame_path = os.path.join(OUTPUT_FOLDER, frame_filename)
                cv2.imwrite(frame_path, frame)
                log_file.write(f"{frame_filename} → {class_name} → {conf:.2f}\n")
                print(f"✅ Saved {class_name} frame {frame_id}")
                saved += 1
                break

    frame_id += 1

with open(PROGRESS_FILE, 'w') as f:
    f.write(str(frame_id))

log_file.close()
cap.release()
print(f"🎯 Done! Saved and logged {saved} frames from frame {start_frame} to {frame_id}")
