# Digital Systems Project

**Student:** Saroofudheen Jamal
**ID:** 23026172

## Project Overview

This project implements a real-time Sign Language Recognition (SLR) system capable of translating signs into Dhivehi.
It utilizes computer vision and deep learning techniques to detect hand gestures and map them to their corresponding meanings.

## Features

- **Real-time Recognition**: Detects signs from a live webcam feed.
- **Dhivehi Translation**: Automatically translates recognized English signs into Dhivehi text.
- **User Interface**: Built with Streamlit for an easy-to-use experience.

## Installation & Running

### 1. Prerequisites (Dependencies)

The project requires the following Python libraries:

*   `streamlit` (Web Interface)
*   `tensorflow==2.15.0` (Machine Learning Framework)
*   `mediapipe==0.10.9` (Hand & Pose Detection)
*   `opencv-python-headless` (Computer Vision)
*   `pandas` (Data Manipulation)
*   `numpy` (Numerical Computations)
*   `scikit-learn` (Machine Learning Utilities)
*   `streamlit-webrtc` (Real-time Video Streaming)
*   `streamlit-pills` (UI Component)
*   `python-bidi` (RTL Text Support for Dhivehi)
*   `protobuf==3.20.3` (Dependency Compatibility)
*   `matplotlib` (Plotting)

### 2. Install Dependencies

You can install all required packages at once:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
cd streamlit
streamlit run app.py
```
