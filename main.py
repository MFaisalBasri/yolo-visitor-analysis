import cv2
import csv
from datetime import datetime, timedelta
from script.yoloDetector import YoloDetector
from script.trackerModule import TrackerModule
import time
import os

# Waktu mulai aktual (misalnya video dimulai pukul 06:30:00)
video_start_time = datetime.strptime("20:24:46", "%H:%M:%S")

# Fungsi untuk konversi detik ke waktu aktual (HH:MM:SS)
def format_time(seconds, start_time):
    return (start_time + timedelta(seconds=seconds)).strftime("%H:%M:%S")

def main():
    video_path = 'assets/video1-uji.mp4'  # ubah disini, ambil video dari folder assets

    # Output folder
    output_dir = 'tracking_results'
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'tracking_data.csv')
    output_video_path = os.path.join(output_dir, 'hasil_tracking.mp4')

    detector = YoloDetector(model_path='model/yolo11s.pt', conf_threshold=0.5)
    tracker = TrackerModule()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {video_path}")
        return

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

        # Garis 1: Jumlah objek
        counted_objects = sum(1 for info in tracker.track_info.values() if info["durasi"] >= 2.0)
        cv2.putText(frame, f"Jumlah Objek: {counted_objects}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Tampilkan bounding box dan ID
        for track_id, (x1, y1, x2, y2), duration in tracks:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID {track_id} | {duration:.1f} detik"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out.write(frame)
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Simpan CSV
    MIN_DURASI = 3.0  # detik

    for track_id, info in tracker.track_info.items():
      if info["durasi"] >= MIN_DURASI:
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
    cv2.destroyAllWindows()
    print(f"Selesai. Data disimpan di {csv_path}")
    print(f"Video hasil tracking disimpan di {output_video_path}")

if __name__ == "__main__":
    main()
