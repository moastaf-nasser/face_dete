# Real-Time Biometric Face Recognition System

A high-performance facial recognition application powered by **Deep Learning** and **Computer Vision**. This system can identify individuals in real-time through a webcam feed by comparing live encodings against a pre-defined database of known faces.

## 🚀 Overview
The project implements a full biometric pipeline:
1. **Database Loading:** Automatically scans an image directory to build a known-face database.
2. **Feature Extraction:** Generates 128-dimensional facial encodings using the `face_recognition` library.
3. **Real-Time Identification:** Captures webcam frames, optimizes them via resizing, and performs Euclidean distance calculations to find the closest match.

## 🛠️ Technical Features
* **Live Encoding Comparison:** Utilizing **HOG (Histogram of Oriented Gradients)** and deep learning models for high-accuracy face locations.
* **Optimization:** Implemented frame scaling (0.25x) to ensure high FPS during real-time processing without sacrificing detection accuracy.
* **Smart Matching:** Uses `face_distance` metrics to determine the best match among the known dataset.

## 💻 Tech Stack
* **Language:** Python
* **Libraries:** OpenCV, Face_Recognition, Dlib, NumPy.
* **Concepts:** Biometric Authentication, Deep Learning Encodings, Real-Time Image Processing.

## 📖 Setup
1. Create a folder named `images` and add photos of people you want to recognize (name the files after the people).
2. Install dependencies: `pip install opencv-python face-recognition numpy`.
3. Run: `python face_recog_script.py`.

---
*Developed by Moustafa Nasser - Part of a research focus on AI and Biometric Security.*
