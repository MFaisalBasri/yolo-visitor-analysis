import cv2
import time
from datetime import datetime, timedelta
from yoloDetector import YoloDetector
from trackerModule import TrackerModule

def main():
    video_path = '../assets/video1-uji.mp4'

    detector = YoloDetector(model_path='../model/yolo11s.pt', conf_threshold=0.3)
    tracker = TrackerModule()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Hitung FPS berjalan
        current_time = time.time()
        elapsed = current_time - prev_time
        fps_berjalan = 1 / elapsed if elapsed > 0 else 0
        prev_time = current_time

        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame, frame_number / video_fps, frame_number)

        # Tambahkan teks jumlah objek terdeteksi
        jumlah_objek = len(tracks)
        cv2.putText(frame, f"Jumlah objek terdeteksi: {jumlah_objek}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        for track_id, (x1, y1, x2, y2), _ in tracks:
            # Bounding box biru
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Label ID
            label = f"ID {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_x, label_y = x1, y1 - 10

            # Highlight biru
            cv2.rectangle(frame,
                          (label_x, label_y - text_h - 4),
                          (label_x + text_w + 6, label_y),
                          (255, 0, 0), -1)

            # Teks putih
            cv2.putText(frame, label, (label_x + 3, label_y - 3),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Tampilkan frame
        cv2.imshow("Hitung Jumlah Pengunjung", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking selesai.")

if __name__ == "__main__":
    main()
