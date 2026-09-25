# ISCMS Local Python Face Recognition Microservice

This microservice provides genuine OpenCV-based Computer Vision face detection and biometric facial recognition for the Integrated Campus Management System (ISCMS).

## Technologies Used
- **Python 3.10+ / 3.14+**
- **OpenCV 4.8+**
- **YuNet ONNX Face Detector** (Lightweight, real-time facial bounding box and landmark detection)
- **SFace ONNX Face Recognizer** (Deep neural network computing 128-dimensional L2-normalized biometric embeddings with Cosine similarity)
- **Flask REST API** (JSON base64 endpoints)

## Installation
```bash
cd python_face_service
python -m pip install -r requirements.txt
```

## Running the Service
```bash
python app.py
```
The service will start on `http://127.0.0.1:5000`.

## Endpoints
- `GET /health` - Diagnostic health check and loaded model status.
- `POST /register-face` - Accepts base64 image, detects face, extracts 128-dim biometric embedding.
- `POST /verify-face` - Compares live camera probe frames against stored embedding using Cosine similarity.
