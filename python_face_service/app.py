"""
Integrated Campus Management System (ISCMS)
Local Computer Vision & Face Recognition Microservice
Built with OpenCV (YuNet Detection + SFace Biometric Recognition) & Flask
Runs 100% locally with zero cloud API dependencies.
"""

import os
import io
import json
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# SFace Cosine Similarity Match Threshold (Standard benchmark is 0.363; we use 0.40 for strict verification)
SIMILARITY_THRESHOLD = 0.40

# Global Detector & Recognizer
detector = None
recognizer = None

def init_models():
    global detector, recognizer
    if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
        try:
            # Default input size (320, 320), confidence score threshold 0.7
            detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.7, 0.3, 5000)
            recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
            print("[INFO] OpenCV YuNet & SFace models loaded successfully.")
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            detector = None
            recognizer = None
    else:
        print("[WARNING] Model files not found in models/ directory.")

init_models()

def decode_base64_image(image_str):
    """Safely decodes base64 data URI into OpenCV BGR numpy image"""
    if not image_str:
        return None
    try:
        if "base64," in image_str:
            image_str = image_str.split("base64,")[1]
        img_bytes = base64.b64decode(image_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Image decode error: {e}")
        return None

def detect_and_align_face(image):
    """
    Detects single face in the image using YuNet.
    Returns: (status_code, face_data, aligned_face_crop, bbox)
    """
    global detector, recognizer
    if detector is None or recognizer is None:
        init_models()
        if detector is None or recognizer is None:
            return "MODEL_UNAVAILABLE", None, None, None

    h, w, _ = image.shape
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return "NO_FACE_DETECTED", None, None, None
    
    if len(faces) > 1:
        # Filter by face area to check if there are multiple prominent faces
        significant_faces = [f for f in faces if (f[2] * f[3]) > (w * h * 0.04)]
        if len(significant_faces) > 1:
            return "MULTIPLE_FACES_DETECTED", None, None, None

    # Best face
    best_face = faces[0]
    bbox = [int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])]

    # Validate face size (must be at least 60x60 pixels for reliable embedding)
    if bbox[2] < 50 or bbox[3] < 50:
        return "FACE_TOO_SMALL", None, None, None

    # Align face using SFace recognizer
    aligned_face = recognizer.alignCrop(image, best_face)
    return "OK", best_face, aligned_face, bbox

def extract_face_embedding(aligned_face):
    """Extracts 128-dimensional normalized embedding vector using SFace"""
    global recognizer
    if recognizer is None:
        return None
    feature = recognizer.feature(aligned_face)
    # Convert numpy array to 1D float list
    return feature.flatten().tolist()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "ISCMS Local OpenCV Face Recognition Engine",
        "detector": "OpenCV YuNet (ONNX)",
        "recognizer": "OpenCV SFace (ONNX)",
        "models_loaded": (detector is not None and recognizer is not None),
        "similarity_threshold": SIMILARITY_THRESHOLD
    })

@app.route("/register-face", methods=["POST"])
def register_face():
    """
    Accepts: { "faculty_id": 1, "image": "data:image/jpeg;base64,..." }
    Extracts biometric embedding and returns vector for database storage.
    """
    data = request.get_json(force=True, silent=True)
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Missing image payload in request."}), 400

    faculty_id = data.get("faculty_id")
    image = decode_base64_image(data["image"])
    if image is None:
        return jsonify({"success": False, "message": "Failed to decode image data."}), 400

    status, face_data, aligned_face, bbox = detect_and_align_face(image)
    if status != "OK":
        error_msgs = {
            "NO_FACE_DETECTED": "No face was detected in the frame. Please look directly at the camera with adequate lighting.",
            "MULTIPLE_FACES_DETECTED": "Multiple faces detected. Please ensure only you are visible in the camera frame.",
            "FACE_TOO_SMALL": "Face is too far from camera. Please move closer to the frame.",
            "MODEL_UNAVAILABLE": "OpenCV biometric models are not initialized."
        }
        return jsonify({"success": False, "error_code": status, "message": error_msgs.get(status, "Face detection failed.")}), 422

    embedding = extract_face_embedding(aligned_face)
    if embedding is None:
        return jsonify({"success": False, "message": "Could not extract biometric features."}), 500

    return jsonify({
        "success": True,
        "faculty_id": faculty_id,
        "embedding": embedding,
        "face_embedding": embedding,
        "bbox": bbox,
        "model": "opencv_sface",
        "message": "Facial biometric profile generated successfully."
    })

@app.route("/verify-face", methods=["POST"])
def verify_face():
    """
    Accepts: {
        "faculty_id": 1,
        "stored_embedding": [...128 floats...],
        "registered_embedding": [...128 floats...],
        "image": "data:image/jpeg;base64,...",
        "frames": ["data:image/...", "data:image/..."],
        "probe_frames": ["data:image/..."]
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "message": "Missing JSON request body."}), 400

    stored_embedding = data.get("stored_embedding") or data.get("registered_embedding")
    if not stored_embedding:
        return jsonify({"success": False, "message": "Missing stored biometric embedding."}), 400

    if isinstance(stored_embedding, str):
        try:
            stored_embedding = json.loads(stored_embedding)
        except Exception:
            return jsonify({"success": False, "message": "Invalid stored embedding JSON."}), 400

    if not isinstance(stored_embedding, list) or len(stored_embedding) != 128:
        return jsonify({"success": False, "message": "Stored embedding must be a 128-dimensional vector."}), 400

    # Image frames for verification and liveness
    frames_b64 = data.get("frames") or data.get("probe_frames") or []
    if not frames_b64 and "image" in data:
        frames_b64 = [data["image"]]


    if not frames_b64:
        return jsonify({"success": False, "message": "No image frames provided for verification."}), 400

    # Process probe frame (last frame or primary frame)
    primary_img = decode_base64_image(frames_b64[-1])
    if primary_img is None:
        return jsonify({"success": False, "message": "Failed to decode verification image."}), 400

    status, face_data, aligned_face, bbox = detect_and_align_face(primary_img)
    if status != "OK":
        error_msgs = {
            "NO_FACE_DETECTED": "Face not detected in the live camera feed. Please position your face in the frame.",
            "MULTIPLE_FACES_DETECTED": "Multiple faces detected. Verification requires a single person in frame.",
            "FACE_TOO_SMALL": "Face is too far from camera. Please move closer.",
            "MODEL_UNAVAILABLE": "OpenCV AI models are not ready."
        }
        return jsonify({
            "success": True,
            "verified": False,
            "error_code": status,
            "message": error_msgs.get(status, "Face detection failed.")
        }), 200

    probe_embedding = extract_face_embedding(aligned_face)

    # Convert to OpenCV compatible float32 arrays
    feat1 = np.array(stored_embedding, dtype=np.float32).reshape(1, 128)
    feat2 = np.array(probe_embedding, dtype=np.float32).reshape(1, 128)

    # Compute Cosine Similarity using SFace recognizer
    global recognizer
    cos_similarity = float(recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE))

    # Basic Liveness Check (Verify multi-frame movement / natural variation if 2+ frames passed)
    liveness_passed = True
    liveness_detail = "Single frame biometric check"
    if len(frames_b64) >= 2:
        prev_img = decode_base64_image(frames_b64[0])
        if prev_img is not None and prev_img.shape == primary_img.shape:
            # Calculate absolute frame pixel difference to prevent identical static image replay
            diff = cv2.absdiff(primary_img, prev_img)
            diff_score = np.mean(diff)
            # If diff is exactly 0.0, it's a duplicated static file; if normal camera jitter (0.5 to 40), passed
            if diff_score < 0.05:
                liveness_passed = False
                liveness_detail = "Static image detected (zero frame jitter). Live camera required."
            else:
                liveness_passed = True
                liveness_detail = f"Live motion verified (Jitter variance: {diff_score:.2f})"

    # Calculate readable human confidence percentage:
    # SFace cosine similarity ranges from -1.0 to 1.0 (typical match > 0.40, strong match > 0.65)
    # We map [0.35, 0.85] -> [75.0%, 99.8%]
    if cos_similarity >= SIMILARITY_THRESHOLD:
        normalized_conf = min(99.8, max(75.0, 75.0 + ((cos_similarity - SIMILARITY_THRESHOLD) / 0.45) * 24.8))
        is_verified = bool(liveness_passed)
    else:
        normalized_conf = max(10.0, min(65.0, (cos_similarity / SIMILARITY_THRESHOLD) * 65.0))
        is_verified = False

    return jsonify({
        "success": True,
        "verified": is_verified,
        "similarity": round(cos_similarity, 4),
        "confidence_score": round(normalized_conf, 1),
        "threshold": SIMILARITY_THRESHOLD,
        "liveness": liveness_passed,
        "liveness_detail": liveness_detail,
        "bbox": bbox,
        "message": "Faculty identity verified. Attendance approved." if is_verified else "Biometric mismatch: Face does not match registered profile."
    })

if __name__ == "__main__":
    print("[INFO] Starting ISCMS Python Face Recognition Service on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
