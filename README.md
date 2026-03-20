# ADAS-Path-Prediction: Resilient Autonomous Navigation
### Advanced AI-Driven Path Forecasting for Sensor-Insecure Environments

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green.svg)](https://github.com/ultralytics/ultralytics)

---

## 📌 Project Overview
This project addresses a critical challenge in **Advanced Driver Assistance Systems (ADAS)**: maintaining reliable path prediction during temporary sensor failure or high-noise environments. Utilizing **YOLOv8** for real-time object detection and a custom **Temporal History Buffer**, the system predicts vehicle trajectories even when visual tracking is momentarily lost.

The core innovation lies in the **Resilient Prediction Engine**, which prevents "system amnesia" during occlusions or camera dropouts—a vital feature for Level 3 and Level 4 autonomous driving standards.

---

## 🚀 Key Features
* **Object Detection:** Real-time multi-class detection (Vehicles, Pedestrians, Cyclists) using YOLOv8.
* **Temporal History Buffering:** Tracks the last $N$ frames of coordinate data to build a motion profile.
* **Resilient Path Prediction:** Implements **Polynomial Regression** to forecast the next 3-5 seconds of movement, mitigating "sensor dropout" gaps.
* **High-Performance Inference:** Optimized for CUDA-enabled environments (NVIDIA RTX series).
* **Real-time Visualization:** Dynamic overlay of predicted paths on live video feeds.

---

## 📊 Mathematical Foundation
Traditional ADAS often fails when an object is partially occluded. This system uses a **weighted historical average** and polynomial fitting:

$$P(t) = \beta_0 + \beta_1 t + \beta_2 t^2 + \epsilon$$

By calculating the second-order derivative (acceleration), we can estimate the path through blind spots, ensuring the vehicle "remembers" where an obstacle was heading even if it is currently hidden.

---

## 🛠 Technical Stack
* **Language:** Python
* **Computer Vision:** OpenCV, Ultralytics (YOLOv8)
* **Mathematics/ML:** NumPy, Scikit-learn (Polynomial Features)
* **Hardware:** Developed and tested on NVIDIA RTX 5060 Ti

---

## 📂 Repository Structure
```text
├── src/                # Core inference and prediction logic
├── notebooks/          # Experimental research and YOLO training logs
├── models/             # Exported ONNX/PyTorch weights
├── data/               # Sample testing footage (not for large datasets)
└── docs/               # Technical specifications and "Quantum Shield" paper
