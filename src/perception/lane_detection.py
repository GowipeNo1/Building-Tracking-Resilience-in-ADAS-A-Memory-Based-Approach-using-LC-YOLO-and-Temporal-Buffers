import cv2
import numpy as np
import yaml

class LaneDetector:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.width = self.config['camera']['width']
        self.height = self.config['camera']['height']
        self.limit = self.config['thresholds']['departure_pixel_limit']
        print(f"[INFO] Lane Detector Initialized with Robust Logic.")

    def get_roi(self, frame):
        mask = np.zeros_like(frame)
        # Trapezoidal ROI is better than triangular for curves
        polygon = np.array([[
            (int(self.width * 0.1), self.height), 
            (int(self.width * 0.9), self.height), 
            (int(self.width * 0.6), int(self.height * 0.6)),
            (int(self.width * 0.4), int(self.height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(frame, mask)

    def process_frame(self, frame):
        # 1. Pre-processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi_edges = self.get_roi(edges)
        
        # 2. Hough Line Transform
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, threshold=50, 
                                minLineLength=100, maxLineGap=150)
        
        lane_frame = frame.copy()
        left_lane_x = []
        right_lane_x = []
        all_slopes = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 999
                
                # Filter noise: Slopes should be diagonal, not horizontal or vertical
                if 0.5 < abs(slope) < 2.0:
                    all_slopes.append(slope)
                    # Split lines into Left vs Right side of the screen
                    if slope < 0: # Negative slope = Left Lane (in image coordinates)
                        left_lane_x.extend([x1, x2])
                        cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    else: # Positive slope = Right Lane
                        right_lane_x.extend([x1, x2])
                        cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 3. Improved Curvature Logic
        curve_direction = "Straight"
        if len(all_slopes) > 0:
            avg_slope = np.mean(all_slopes)
            # Refined thresholds: Negative is usually Left, Positive is usually Right
            if avg_slope < -0.15:
                curve_direction = "Curving Left"
            elif avg_slope > 0.15:
                curve_direction = "Curving Right"

        # 4. Improved Offset Logic (Center of the detected lanes)
        if left_lane_x and right_lane_x:
            lane_center = (np.mean(left_lane_x) + np.mean(right_lane_x)) / 2
        elif left_lane_x: # If only left is seen, estimate right
            lane_center = np.mean(left_lane_x) + 200 
        elif right_lane_x: # If only right is seen, estimate left
            lane_center = np.mean(right_lane_x) - 200
        else:
            lane_center = self.width / 2

        offset = lane_center - (self.width / 2)
        departure_detected = abs(offset) > self.limit
        
        return lane_frame, offset, departure_detected, curve_direction