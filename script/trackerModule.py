from deep_sort_realtime.deepsort_tracker import DeepSort
from datetime import datetime

class TrackerModule:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=30,
            n_init=2,
            nms_max_overlap=0.3,
            max_cosine_distance=0.4,
            embedder="mobilenet",
            half=True,
            bgr=True
        )
        self.track_info = {}

    def update(self, detections, frame, current_video_time, frame_number):
        tracks = self.tracker.update_tracks(detections, frame=frame)
        updated_tracks = []

        now_str_date = datetime.now().strftime("%Y-%m-%d")

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            if track_id not in self.track_info:
                self.track_info[track_id] = {
                    "tanggal": now_str_date,
                    "start_time": current_video_time,
                    "end_time": current_video_time,
                    "durasi": 0.0,
                    "jumlah_frame": 1,
                    "frame_muncul": frame_number,
                    "frame_terakhir": frame_number
                }
            else:
                info = self.track_info[track_id]
                info["end_time"] = current_video_time
                info["durasi"] = info["end_time"] - info["start_time"]
                info["jumlah_frame"] += 1
                info["frame_terakhir"] = frame_number

            updated_tracks.append((track_id, (x1, y1, x2, y2), self.track_info[track_id]["durasi"]))

        return updated_tracks
