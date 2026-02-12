# YOLO Visitor Analysis

**YOLO Visitor Analysis** adalah aplikasi berbasis *Computer Vision* yang menggunakan **YOLOv11** dan **DeepSORT** untuk mendeteksi serta melacak pengunjung dalam video.  
Aplikasi ini menghitung **jumlah objek** dan **durasi keberadaan** setiap objek berdasarkan frame dan FPS video.

---

## 🚀 Fitur Utama

- Deteksi objek secara real-time menggunakan **YOLOv11 (Ultralytics)**  
- Pelacakan objek dengan **DeepSORT**  
- Perhitungan **durasi keberadaan** objek berdasarkan frame dan FPS  
- Output hasil dalam bentuk:
  - 📄 File CSV (data deteksi dan waktu)
  - 🎥 Video hasil deteksi

---

## 🧰 Teknologi yang Digunakan

- **Python 3.10+**
- **YOLOv11 (Ultralytics)**
- **DeepSORT Realtime**
- **OpenCV**

---


## ⚙️ Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/MFaisalBasri/yolo-visitor-analysis.git
cd yolo-visitor-analysis
```
## 🧱 Virtual Environment Setup

Gunakan **Virtual Environment (venv)** agar dependensi proyek tidak bercampur dengan sistem global.

---

### 🪟 Windows

# 1. Buat virtual environment
```bash
python -m venv venv
```
# 2. Aktifkan environment
```bash
venv\Scripts\activate
```
# 3. Instal dependencies
```bash
pip install -r requirements.txt
```
## 📦 Wajib Download Dependencies

Sebelum menjalankan program, pastikan kamu sudah menginstal library utama berikut:

```bash
pip install opencv-contrib-python
pip install ultralytics
pip install deep-sort-realtime
```

