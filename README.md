# Multi-Mode Biometric Recognition & Communication System

An advanced engineering project that combines **Deep Learning** with **Distributed Systems**. This repository features a real-time facial recognition system capable of operating both locally and over a network using a custom-built communication protocol.

## 📡 System Architectures

### 1. Distributed Network Mode (Client-Server)
A centralized AI hub designed for remote biometric processing:
* **Server (`server.py`):** A robust backend that listens for incoming UDP packets, deserializes JSON/Base64 image data, and performs recognition.
* **Client (`client.py`):** Sends optimized image packets to the server for remote identification.
* **Communication Layer (`communication.py`):** A custom-coded socket management class handling reliable data transmission.

### 2. Standalone Vision Mode
A high-speed identification tool optimized for direct hardware access:
* **Real-Time Processing:** Uses 128-d face encodings and HOG algorithms for precision.
* **Efficiency:** Scaled frame analysis (0.25x) to maximize FPS on edge devices.

## 🛠️ Technical Highlights
* **Data Serialization:** Custom pipeline for converting binary images to Base64 and wrapping them in structured JSON packets.
* **Computer Vision:** Advanced implementation of **OpenCV** and **Dlib-based** face recognition.
* **Backend Engineering:** Managed socket programming, buffer handling, and dynamic identity learning.

## 📂 Repository Structure
* `recognition.py`: Core AI logic and feature extraction.
* `communication.py`: Network abstraction layer.
* `server.py` / `client.py`: Distributed system implementation.
* `Face Recognition.py`: Main standalone application.
* `haarcascade_frontalface_default.xml`: Pre-trained model for initial detection.

---
*Developed by Moustafa Nasser - Focused on the intersection of Software Engineering, AI, and Network Infrastructure.*
