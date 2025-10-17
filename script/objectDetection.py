print("Sistem dimulai...")

from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO('../model/yolo11s.pt')

# Inisialisasi video dan FPS
video_path = '../assets/video1-uji.mp4'  
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

# Jalankan prediksi
results = model.predict(
    source=video_path,
    stream=True,
    device=0,
    classes=[0],     # hanya manusia
    conf=0.3,
    show=False
)

# Timer untuk FPS berjalan
prev_time = time.time()

# Buat jendela bisa diskalakan
cv2.namedWindow("Deteksi Pengunjung", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteksi Pengunjung", 1280, 720)  # Sesuaikan ukuran jendela

# Tampilkan frame dengan bounding box dan label
for result in results:
    if result.orig_img is None:
        continue

    frame = result.orig_img.copy()
    boxes = result.boxes

    # Hitung FPS berjalan
    current_time = time.time()
    elapsed = current_time - prev_time
    fps_berjalan = 1 / elapsed if elapsed > 0 else 0
    prev_time = current_time

    # Gambar bounding box dan label
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Person {conf * 100:.1f}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_x, label_y = x1, y1 - 10
            bg_top_left = (label_x, label_y - text_h - 4)
            bg_bottom_right = (label_x + text_w + 6, label_y)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, bg_top_left, bg_bottom_right, (255, 0, 0), -1)
            cv2.putText(frame, label, (label_x + 3, label_y - 3),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Resize frame untuk tampilan
    display_frame = cv2.resize(frame, (1280, 720))  # Atau ukuran lain sesuai layar

    # Tampilkan frame
    cv2.imshow("Deteksi Pengunjung", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Selesai.")
