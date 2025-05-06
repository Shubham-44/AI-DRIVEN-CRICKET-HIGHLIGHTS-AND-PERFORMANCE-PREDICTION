import pandas as pd
import streamlit as st
import cv2
import tempfile
import os
import subprocess
from ultralytics import YOLO
from datetime import timedelta, datetime
import joblib
import pickle
import io
import shutil

# --- CONFIG ---
MODEL_PATH = 'runs/weights/best.pt'
CLIP_OUTPUT_DIR = 'data/clips_batch'
FINAL_OUTPUT_PATH = 'data/final_clips/highlights_final.mp4'
CSV_LOG_DIR = 'data/logs'
CONF_THRESHOLD = 0.5
EVENT_CLASSES = {0: 'catch', 1: 'boundary', 2: 'celebration', 3: 'wicket'}
EVENT_COMBOS = [('wicket', 'celebration'), ('catch', 'celebration'), ('boundary', 'celebration')]
PRE_SECONDS = 4
POST_SECONDS = 4
MIN_GAP_SECONDS = 6


# --- UTILS ---

def extract_frames(video_path, start_frame, end_frame, fps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


def save_clip(video_path, start_sec, end_sec, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

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


def run_event_detection(temp_video_path, start_frame, end_frame, fps, progress):
    # STEP A: Extract frames first
    frames_dir = "data/extracted_frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    extract_frames(temp_video_path, start_frame, end_frame, fps, frames_dir)

    model = YOLO(MODEL_PATH)
    results = model.predict(temp_video_path, stream=True, conf=CONF_THRESHOLD)

    detected_events = []
    frame_idx = start_frame
    total_frames = end_frame - start_frame
    progress.progress(0)

    for result in results:
        if frame_idx > end_frame:
            break

        labels = result.names
        boxes = result.boxes.cls.tolist()
        timestamp = frame_idx / fps
        detected = set([labels[int(cls)] for cls in boxes])
        detected_events.append((timestamp, detected))
        frame_idx += 1

        if frame_idx % 10 == 0:
            percent = int(100 * (frame_idx - start_frame) / total_frames)
            progress.progress(min((frame_idx - start_frame) / total_frames, 1.0),
                              text=f"Detecting events... {percent}%")

    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(CSV_LOG_DIR, exist_ok=True)
    csv_path = os.path.join(CSV_LOG_DIR, f'highlight_log_{timestamp_str}.csv')

    saved_start_times = []
    clip_paths = []

    with open(csv_path, 'w') as log_file:
        log_file.write("clip_name,event_1,event_2,start_time,end_time,start_frame,end_frame,fps\n")
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
                    start_f = int(clip_start * fps)
                    end_f = int(clip_end * fps)
                    clip_name = f"highlight_{e1}_{e2}_{format_time(clip_start)}.mp4"
                    clip_path = os.path.join(CLIP_OUTPUT_DIR, clip_name)

                    save_clip(temp_video_path, clip_start, clip_end, clip_path)
                    clip_paths.append(clip_path)

                    log_file.write(
                        f"{clip_name},{e1},{e2},{format_time(clip_start)},{format_time(clip_end)},{start_f},{end_f},{fps}\n")
                    break
    return clip_paths


def merge_clips(clip_paths, output_path):
    temp_list_path = "merge_list.txt"
    with open(temp_list_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{path}'\n")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", temp_list_path, "-c", "copy", output_path
    ])
    os.remove(temp_list_path)


# --- STREAMLIT UI ---
st.set_page_config(page_title="🏏 Cricket Highlights AI", layout="wide")
st.title("🏏 AI-Powered Cricket Highlights Generator & Performance Prediction")

mode = st.sidebar.radio("Select Mode", [
    "🎬 Upload & Process Match",
    "🖼️ Single Frame Prediction",
    "📊 Team Performance Prediction"
])

if mode == "🎬 Upload & Process Match":
    st.header("🎬 Generate Highlight")
    uploaded_video = st.file_uploader("📁 Upload Match Video (.mp4)", type=["mp4"])

    if uploaded_video:
        st.success(f"✅ Video uploaded: {uploaded_video.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            temp_video_path = tmp.name

        cap = cv2.VideoCapture(temp_video_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = total_frames / fps if fps else 0
            duration_hr = duration_sec / 3600

            st.subheader("📊 Video Info")
            st.write(f"**Total Frames:** {total_frames}")
            st.write(f"**FPS (Auto):** {fps:.2f}")
            st.write(f"**Duration:** {duration_sec:.1f} sec ({duration_hr:.2f} hours)")

            st.subheader("🎯 Frame Range Selection")
            start_frame = st.number_input("Start Frame", min_value=0, max_value=total_frames - 1, value=0)
            end_frame = st.number_input("End Frame", min_value=start_frame + 1, max_value=total_frames, value=1000)
            manual_fps = st.number_input("Manual FPS (optional)", min_value=1.0, max_value=60.0, value=float(fps),
                                         step=0.5)

            # NEW: Adjusted duration based on selected frame range & manual FPS
            range_duration_sec = (end_frame - start_frame) / manual_fps
            st.write(
                f"**Selected Range Duration:** {range_duration_sec:.1f} sec ({range_duration_sec / 3600:.2f} hours)")

            if st.button("🚀 Start Event Detection"):
                progress = st.progress(0, text="Detecting events...")
                clips = run_event_detection(temp_video_path, start_frame, end_frame, manual_fps, progress)

                if clips:
                    st.success(f"✅ {len(clips)} clips generated. Merging...")
                    merge_clips(clips, FINAL_OUTPUT_PATH)
                    st.success("✅ Merging complete. Ready to download.")

                    with open(FINAL_OUTPUT_PATH, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Highlights Video",
                            data=f.read(),
                            file_name="highlights_final.mp4",
                            mime="video/mp4"
                        )
                else:
                    st.warning("⚠️ No valid event combinations found in selected frame range.")
        cap.release()

elif mode == "🖼️ Single Frame Prediction":
    st.header("🖼️ Single Frame Prediction")

    uploaded_image = st.file_uploader("📸 Upload a Frame (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Frame", use_container_width=True)
        temp_img_path = os.path.join("temp_single.jpg")
        with open(temp_img_path, "wb") as f:
            f.write(uploaded_image.read())

        st.success("✅ Image uploaded. Running detection...")
        model = YOLO(MODEL_PATH)
        result = model.predict(temp_img_path)[0]

        boxes = result.boxes
        if boxes is not None and len(boxes.cls) > 0:
            st.subheader("🎯 Detected Events:")
            labels = result.names
            for cls_id in boxes.cls.tolist():
                st.write(f"🔹 {labels[int(cls_id)]}")
        else:
            st.warning("❌ No event detected in this frame.")

elif mode == "📊 Team Performance Prediction":
    st.subheader("🏏 Team Score Predictor (T20 IPL Match Context)")
    st.info("Here you’ll be able to predict match outcomes or performance based on team and player stats.")
    # Load model and encoders
    model = joblib.load("outputs/enhanced_rf_model.pkl")
    venue_enc = pickle.load(open("outputs/venue_encoder.pkl", "rb"))
    team_enc = pickle.load(open("outputs/team_encoder.pkl", "rb"))
    df = pd.read_csv("outputs/enhanced_player_stats.csv")
    avg_runs = df["runs_scored"].mean()

    # st.title("🏏 Team Score Predictor (T20 IPL Match Context)")
    st.markdown("This app predicts the total runs a team might score in a T20 IPL match based on match context and "
                "aggregate performance metrics.")

    # Inputs
    balls_faced = st.slider("Balls Faced", 1, 120, 30)
    strike_rate = st.slider("Strike Rate", 0.0, 300.0, 130.0)
    fours = st.slider("Fours Hit", 0, 20, 4)
    sixes = st.slider("Sixes Hit", 0, 15, 2)
    venue = st.selectbox("Venue", venue_enc.classes_)
    team = st.selectbox("Batting Team", team_enc.classes_)
    year = st.slider("Match Year", 2008, 2025, 2023)

    # Encode
    venue_encoded = venue_enc.transform([venue])[0]
    team_encoded = team_enc.transform([team])[0]

    # Prepare input
    feature_order = ['balls_faced', 'fours', 'sixes', 'strike_rate', 'venue_enc', 'team_enc', 'year']
    X = pd.DataFrame([[
        balls_faced, fours, sixes, strike_rate, venue_encoded, team_encoded, year
    ]], columns=feature_order)

    # Predict
    if st.button("🚀 Predict Runs"):
        prediction = model.predict(X)[0]
        st.success(f"🎯 Predicted Runs: **{prediction:.2f}**")

        # Category
        if prediction < 10:
            category = "Poor"
        elif prediction <= 30:
            category = "Average"
        else:
            category = "Good"

        st.info(f"📊 Performance Category: **{category}**")

        # IPL average comparison
        diff = prediction - avg_runs
        st.markdown(f"📌 Compared to IPL average ({avg_runs:.2f} runs): **{diff:+.2f} runs**")

        # CSV download
        result = X.copy()
        result['predicted_runs'] = prediction
        csv = result.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction as CSV",
            data=csv,
            file_name='predicted_runs.csv',
            mime='text/csv'
        )
