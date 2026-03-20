import sys
import os

# Manually add the src path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from perception.obj_detector import ObjectDetector
    print("SUCCESS: The ObjectDetector class was found!")
except Exception as e:
    print(f"FAILED: {e}")
    # This will tell us what Python actually sees in that file
    import perception.obj_detector
    print("Available attributes in obj_detector:", dir(perception.obj_detector))