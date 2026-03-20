import cv2
import yaml
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, config):
        self.config = config
        model_path = config['models'].get('obj_model_path', 'yolo11n.pt')
        self.model = YOLO(model_path) 
        self.conf = config['thresholds'].get('conf_threshold', 0.25)
        self.classes = config['models'].get('target_classes', [2, 3, 5, 7])

    def process_frame(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, classes=self.classes, verbose=False)
        detections = []
        if len(results) > 0:
            for box in results[0].boxes:
                detections.append({
                    "box": box.xyxy[0].tolist(),
                    "class": int(box.cls[0]),
                    "conf": float(box.conf[0])
                })
        return detections