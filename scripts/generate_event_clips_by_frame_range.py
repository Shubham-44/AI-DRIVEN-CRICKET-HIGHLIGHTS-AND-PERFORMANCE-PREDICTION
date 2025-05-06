import cv2
import os
import csv
from ultralytics import YOLO
from datetime import timedelta, datetime

# --- CONFIG ---
VIDEO_PATH = 'data/videos/WIVsIND.mp4'
MODEL_PATH = 'runs/weights/best.pt'
CLIP_OUTPUT_DIR = 'data/clips_batch'
CSV_LOG_DIR = 'data/logs'
CONF_THRESHOLD = 0.5

# --- FPS CONFIG ---
USE_MANUAL_FPS = True
MANUAL_FPS = 10  # Used only if USE_MANUAL_FPS = True

EVENT_CLASSES = {
    0: 'catch',
    1: 'boundary',
    2: 'celebration',
    3: 'wicket'
}
EVENT_COMBOS = [
    ('wicket', 'celebration'),
    ('catch', 'celebration'),
    ('boundary', 'celebration')
]
PRE_SECONDS = 4
POST_SECONDS = 4
MIN_GAP_SECONDS = 6  # Prevent near-duplicate events


# ----------------

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def save_clip(video_path, start_sec, end_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = MANUAL_FPS if USE_MANUAL_FPS else cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    while cap.isOpened():
        current_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_sec > end_sec:
            break
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()


# --- Load model and video metadata ---
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = MANUAL_FPS if USE_MANUAL_FPS else cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"🎬 Total Frames: {total_frames}")
print(f"🎯 Using FPS: {fps}")

# --- Prediction phase ---
results = model.predict(VIDEO_PATH, stream=True, conf=CONF_THRESHOLD)
detected_events = []
frame_idx = 0

for result in results:
    labels = result.names
    boxes = result.boxes.cls.tolist()
    timestamp = frame_idx / fps
    detected = set([labels[int(cls)] for cls in boxes])
    detected_events.append((timestamp, detected))
    frame_idx += 1

# --- CSV Log Setup ---
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(CSV_LOG_DIR, exist_ok=True)
csv_path = os.path.join(CSV_LOG_DIR, f'highlight_log_{timestamp_str}.csv')

saved_start_times = []

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['clip_name', 'event_1', 'event_2', 'start_time', 'end_time', 'start_frame', 'end_frame', 'fps'])

    for i in range(len(detected_events) - 1):
        ts1, ev1 = detected_events[i]
        ts2, ev2 = detected_events[i + 1]
        combined = ev1.union(ev2)

        for e1, e2 in EVENT_COMBOS:
            if e1 in combined and e2 in combined:
                clip_start = max(0, int(ts1) - PRE_SECONDS)
                clip_end = int(ts2) + POST_SECONDS

                if any(abs(clip_start - prev) < MIN_GAP_SECONDS for prev in saved_start_times):
                    break

                saved_start_times.append(clip_start)

                start_frame = int(clip_start * fps)
                end_frame = int(clip_end * fps)
                clip_name = f"{os.path.basename(VIDEO_PATH).split('.')[0]}_{e1}_{e2}_{format_time(clip_start)}.mp4"
                clip_path = os.path.join(CLIP_OUTPUT_DIR, clip_name)

                print(f"📦 Saving clip: {clip_name}")
                save_clip(VIDEO_PATH, clip_start, clip_end, clip_path)
                writer.writerow(
                    [clip_name, e1, e2, format_time(clip_start), format_time(clip_end), start_frame, end_frame, fps])
                break
