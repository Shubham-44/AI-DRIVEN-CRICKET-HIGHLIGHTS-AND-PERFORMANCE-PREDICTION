import os
from datetime import datetime
import subprocess

# --- CONFIG ---
CLIP_FOLDER = 'data/clips_batch'
OUTPUT_FOLDER = 'data/final_clips'
LOG_FOLDER = 'data/logs'
MERGE_LIST_FILE = os.path.join(LOG_FOLDER, 'clip_list.txt')

# --- Ensure folders exist ---
os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Get and sort clips ---
clip_files = sorted([f for f in os.listdir(CLIP_FOLDER) if f.endswith('.mp4')])
if not clip_files:
    print("❌ No clips found to merge.")
    exit()

# --- Write to safe FFmpeg merge list ---
with open(MERGE_LIST_FILE, 'w') as f:
    for clip in clip_files:
        abs_path = os.path.abspath(os.path.join(CLIP_FOLDER, clip))
        f.write(f"file '{abs_path}'\n")

# --- Output filename with timestamp ---
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = os.path.join(OUTPUT_FOLDER, f"merged_highlights_{timestamp}.mp4")

# --- FFmpeg merge command ---
command = [
    'ffmpeg', '-f', 'concat', '-safe', '0',
    '-i', MERGE_LIST_FILE,
    '-c', 'copy',
    output_path, '-y'
]
subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print(f"✅ Done boss! Merged {len(clip_files)} clips → {output_path}")
