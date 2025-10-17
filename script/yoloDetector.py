from ultralytics import YOLO

class YoloDetector:
    def __init__(self, model_path='D:\Code\object-detection\yolo\computer-vision\model\yolo11s.pt', conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if self.model.names[cls] == 'person':
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        return detections
