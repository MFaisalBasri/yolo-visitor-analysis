import streamlit as st
import cv2
import csv
from datetime import datetime, timedelta
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from script.yoloDetector import YoloDetector
from script.trackerModule import TrackerModule

# Set waktu mulai video
video_start_time = datetime.strptime("20:24:46", "%H:%M:%S")

def format_time(seconds, start_time):
    return (start_time + timedelta(seconds=seconds)).strftime("%H:%M:%S")

def run_tracking(video_path, frame_placeholder, progress_bar):
    output_dir = "tracking_results"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "tracking_data.csv")
    output_video_path = os.path.join(output_dir, "hasil_tracking.mp4")
    grafik_path = os.path.join(output_dir, "grafik_durasi.png")

    detector = YoloDetector(model_path="D:\\Code\\object-detection\\yolo\\computer-vision\\model\\yolo11s.pt", conf_threshold=0.5)
    tracker = TrackerModule()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Tidak bisa membuka video: {video_path}")
        return None, None, None

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = total_frame_count / fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Track ID", "Tanggal", "Waktu Muncul (Video)", "Waktu Terakhir (Video)",
        "Durasi (detik)", "Jumlah Frame", "Frame Muncul", "Frame Terakhir",
        "Total Frame Video", "FPS Video", "Durasi Video (detik)"
    ])

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        start_time_frame = time.time()
        current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame, current_video_time, frame_number)

        # Overlay teks
        cv2.putText(frame, f"Jumlah Objek Terdeteksi: {len(tracks)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for track_id, (x1, y1, x2, y2), duration in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID {track_id} | {duration:.1f} detik"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Tampilkan frame
        frame_placeholder.image(frame_rgb, caption=f"Frame: {frame_number}", channels="RGB")

        # Update progress bar
        progress = frame_number / total_frame_count
        progress_bar.progress(min(progress, 1.0))

    for track_id, info in tracker.track_info.items():
        if info["durasi"] >= 3.0:
            csv_writer.writerow([
                track_id,
                info["tanggal"],
                format_time(info["start_time"], video_start_time),
                format_time(info["end_time"], video_start_time),
                round(info["durasi"], 2),
                info["jumlah_frame"],
                info["frame_muncul"],
                info["frame_terakhir"],
                total_frame_count,
                round(fps, 2),
                round(video_duration, 2)
            ])

    csv_file.close()
    cap.release()
    out.release()
    return csv_path, output_video_path, grafik_path

# ========== STREAMLIT UI ==========
st.set_page_config(layout="wide")
st.title("Deteksi & Tracking (YOLOv11 + DeepSORT)")

uploaded_file = st.file_uploader("📁 Upload video (.mp4)", type=["mp4"])

if uploaded_file:
    temp_video_path = "assets/video.mp4"
    os.makedirs("assets", exist_ok=True)
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Video berhasil diunggah.")

    st.subheader("🎬 Pratinjau Video")
    st.video(temp_video_path)

    if st.button("🔍 Mulai Deteksi"):
        st.subheader("⏳ Progress Deteksi")
        progress_bar = st.progress(0)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📸 Proses Deteksi")
            frame_placeholder = st.empty()

        csv_path, video_path, grafik_path = run_tracking(temp_video_path, frame_placeholder, progress_bar)

        if csv_path and video_path:
            with col2:
                st.markdown("### 📊 Grafik Durasi per Track ID")
                df = pd.read_csv(csv_path)
                df["Track Label"] = df["Track ID"].apply(lambda x: f"{x}")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(df["Track Label"], df["Durasi (detik)"], color="skyblue")
                ax.set_xlabel("Track ID")
                ax.set_ylabel("Durasi (detik)")
                ax.set_title("Durasi Terdeteksi per Track ID")
                st.pyplot(fig)

                fig.savefig(grafik_path, bbox_inches="tight")

            st.success("✅ Deteksi selesai!")

            st.subheader("⬇️ Unduh Hasil")
            with open(csv_path, "rb") as f:
                st.download_button("📄 Unduh CSV", f, file_name="tracking_data.csv", mime="text/csv")

            with open(video_path, "rb") as f:
                st.download_button("🎞️ Unduh Video", f, file_name="hasil_tracking.mp4", mime="video/mp4")

            with open(grafik_path, "rb") as f:
                st.download_button("🖼️ Unduh Grafik", f, file_name="grafik_durasi.png", mime="image/png")
