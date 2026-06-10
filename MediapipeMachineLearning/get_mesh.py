import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from geometry import extract_geometric_features

def process_dataset(input_dir, output_csv_name):
    print(f"Initializing MediaPipe for: {input_dir}")
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, 
        num_faces=1, 
        running_mode=vision.RunningMode.IMAGE
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    dataset_rows = []

    for emotion_label in sorted(os.listdir(input_dir)):
        folder_path = os.path.join(input_dir, emotion_label)
        if not os.path.isdir(folder_path): continue
        
        print(f"  Extracting geometry from folder: {emotion_label}")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            frame = cv2.imread(img_path)
            if frame is None: continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            if len(detection_result.face_landmarks) > 0:
                landmarks = detection_result.face_landmarks[0]
                flat_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten().tolist()
                
                try:
                    geo_features = extract_geometric_features(flat_coords)
                    dataset_rows.append(geo_features + [emotion_label])
                except Exception:
                    continue

    if not dataset_rows:
        print("No faces found! Check your folder paths.")
        return

    num_features = len(dataset_rows[0]) - 1 
    
    columns = [f"feature_{i}" for i in range(num_features)] + ['label']
    
    df = pd.DataFrame(dataset_rows, columns=columns)
    df.to_csv(output_csv_name, index=False)
    print(f"Done! Saved {num_features} hybrid features per image to: {output_csv_name}\n")

if __name__ == "__main__":
    process_dataset("./TestImages", "geometric_test_data.csv")