import cv2
import numpy as np
import mediapipe as mp
import joblib
import os
from collections import deque
import statistics
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from geometry import extract_geometric_features

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Loading Hybrid XGBoost Model and Translator...")
model = joblib.load('emotion_xgb_model.pkl')
encoder = joblib.load('label_encoder.pkl')

print("Initializing MediaPipe Mesh...")
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceLandmarker.create_from_options(options)

BUFFER_SIZE = 10
prediction_buffer = deque(maxlen=BUFFER_SIZE)

print("Starting Webcam... (Press 'q' to quit)")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    height, width, _ = frame.shape
    aspect_ratio = height / width
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    if len(detection_result.face_landmarks) > 0:
        landmarks = detection_result.face_landmarks[0]
        
        flat_coords = []
        for lm in landmarks:
            corrected_x = lm.x
            corrected_y = lm.y * aspect_ratio
            corrected_z = lm.z * aspect_ratio
            flat_coords.extend([corrected_x, corrected_y, corrected_z])
        
        try:
            geo_features = extract_geometric_features(flat_coords)
            prediction_encoded = model.predict([geo_features])
            raw_emotion = encoder.inverse_transform(prediction_encoded)[0]
            
            prediction_buffer.append(raw_emotion)
            
            smoothed_emotion = statistics.mode(prediction_buffer)
            
            text = f"{smoothed_emotion.upper()}"
            
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            
        except Exception as e:
            pass
            
    cv2.imshow('XGBoost Live Emotion Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()