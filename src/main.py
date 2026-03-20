import os
import sys
import cv2
import yaml
import time
import csv
from perception.obj_detector import ObjectDetector

# --- STEP 1: DYNAMIC PATH FIXING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR) 

# --- STEP 2: SECURE IMPORTS ---
try:
    from perception.lane_detection import LaneDetector
    from perception.obj_detector import ObjectDetector
    print("[SUCCESS] All ADAS modules loaded correctly.")
except ImportError as e:
    print(f"[ERROR] Import Failure: {e}")
    sys.exit()

class ADASSystem:
    def __init__(self, config_path):
        # 1. Load Global Config
        if not os.path.exists(config_path):
            print(f"[ERROR] Config not found at: {config_path}")
            sys.exit()
            
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # 2. Initialize Engines
        self.lane_engine = LaneDetector(config_path)
        self.obj_engine = ObjectDetector(self.cfg)
        
        # 3. Setup Video Source
        source_rel = self.cfg.get('camera', {}).get('video_path', 'assets/samples/test_video.mp4')
        source = os.path.join(SCRIPT_DIR, "..", source_rel)
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"[ERROR] Could not open video: {source}")
            sys.exit()

        self.width = self.cfg.get('camera', {}).get('width', 1280)
        self.height = self.cfg.get('camera', {}).get('height', 720)

        # 4. Logger Setup
        self.log_file = os.path.join(SCRIPT_DIR, "..", "logs", "incident_log.csv")
        self.screenshot_dir = os.path.join(SCRIPT_DIR, "..", "logs", "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)
        
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Type", "Offset", "Curve", "Nearest_Dist"])
        
        self.last_log_time = 0 # Prevents log flooding

    def log_incident(self, alert_type, frame, offset, curve, dist=None):
        """Saves telemetry data and a screenshot when a safety limit is breached."""
        current_time = time.time()
        if current_time - self.last_log_time > 3: # 3-second cooldown
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save Screenshot
            img_path = os.path.join(self.screenshot_dir, f"{alert_type}_{timestamp}.jpg")
            cv2.imwrite(img_path, frame)
            
            # Write to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                dist_str = f"{dist:.1f}" if dist is not None else "N/A"
                writer.writerow([timestamp, alert_type, f"{offset:.1f}", curve, dist_str])
            
            print(f"[LOG] {alert_type} Recorded: {timestamp}")
            self.last_log_time = current_time

    def run(self):
        print("--- ADAS RUNNING: Press 'q' to stop ---")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                # Loop video if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # A. PERCEPTION (Using Smoothed Filter Data)
            frame, offset, ldw_alert, curve_msg = self.lane_engine.process_frame(frame)
            detections = self.obj_engine.process_frame(frame)

            # B. SAFETY LOGIC (FCW) & DISTANCE
            fcw_alert = False
            nearest_dist = 999.0
            danger_y = int(self.height * self.cfg['thresholds'].get('danger_zone_y', 0.8))
            # Change this:
            # focal = self.cfg['calibration'].get('focal_length', 950)

            # To this (Safe way to handle missing keys):
            calib_data = self.cfg.get('calibration', {})
            focal = calib_data.get('focal_length', 950)
            real_h = calib_data.get('avg_car_height', 1.5)

            for det in detections:
                x1, y1, x2, y2 = map(int, det['box'])
                
                # Pinhole distance formula
                pixel_h = y2 - y1
                dist = (real_h * focal) / pixel_h if pixel_h > 0 else 0
                nearest_dist = min(nearest_dist, dist)
                
                # Collision Check
                if y2 > danger_y or dist < self.cfg['thresholds'].get('collision_dist_threshold', 5.0):
                    fcw_alert = True

                # Draw Object Boxes
                color = (0, 0, 255) if dist < 10 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{det['label']} {dist:.1f}m", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # C. EVENT LOGGING
            if ldw_alert:
                self.log_incident("LDW", frame, offset, curve_msg)
            if fcw_alert:
                self.log_incident("FCW", frame, offset, curve_msg, nearest_dist)

            # D. UI LAYERING & DASHBOARD
            # Red Danger Zone Overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, danger_y), (self.width, self.height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            # Alerts and Telemetry
            if ldw_alert: cv2.putText(frame, "LANE DEPARTURE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            if fcw_alert: cv2.putText(frame, "COLLISION WARNING!", (400, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
            
            cv2.putText(frame, f"ROAD: {curve_msg}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Offset: {int(offset)}px", (self.width-250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow("Modern ADAS 2025", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "configs", "model_cfg.yaml")
    adas = ADASSystem(CONFIG_PATH)
    from perception.lane_detection import LaneDetector
    from perception.obj_detector import ObjectDetector
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "configs", "model_cfg.yaml")
    adas = ADASSystem(CONFIG_PATH)
    adas.run()