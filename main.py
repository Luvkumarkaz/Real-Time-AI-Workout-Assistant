import cv2
import mediapipe as mp
import numpy as np
import torch
from model import WorkoutClassifier # Import your architecture

# --- CONFIGURATION ---
MODEL_PATH = "../models/exercise_classifier.pth"
INPUT_SIZE = 99  # 33 landmarks * 3 coords
NUM_CLASSES = 10   # CHANGE THIS to match your specific dataset

# UPDATE THIS MAP BASED ON YOUR TRAINING OUTPUT
class_names = {
    0: "Jumping jacks (down)",
    1: "Jumping jacks (up)",
    2: "Pull ups (down)",
    3: "Pull ups (up)",
    4: "Push ups (down)",
    5: "Push ups (up)",
    6: "Sit up (down)",
    7: "Sit up (up)",
    8: "Squats (down)",
    9: "Squats (up)"
}

# --- 1. SETUP MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading model on {device}...")

# Initialize the architecture (Must match training!)
model = WorkoutClassifier(INPUT_SIZE, NUM_CLASSES) 

# Load the trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # <--- CRITICAL: Turns off Dropout & BatchNorm for testing
print("Model loaded successfully!")

# --- 2. SETUP MEDIAPIPE ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
counter=0
current_stage = None
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    current_prediction = "Waiting..."
    confidence = 0.0
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        lm_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        hip_center = (lm_array[23] + lm_array[24]) / 2.0
        centered_array = lm_array - hip_center
        max_dist = np.max(np.linalg.norm(centered_array, axis=1))
        if max_dist < 1e-6: max_dist = 1
        normalized_array = centered_array / max_dist
        input_vector = normalized_array.flatten()
        tensor_input = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor_input)
            probabilities = torch.softmax(output, dim=1)
            max_prob, class_idx = torch.max(probabilities, 1)
            class_id = class_idx.item()
            confidence = max_prob.item()
            if class_id in class_names:
                current_prediction = class_names[class_id]
            else:
                current_prediction = f"Class {class_id}"
            
            if confidence > 0.7:
                if "Squats" in current_prediction:
                    if "down" in current_prediction:
                        current_stage = "down"  
                    elif "(up)" in current_prediction and current_stage == "down":
                        current_stage = "up"
                        counter += 1

                elif "Push ups" in current_prediction:
                    if "down" in current_prediction:
                        current_stage = "down"
                    elif "(up)" in current_prediction and current_stage == "down":
                        current_stage = "up"
                        counter += 1
                
                elif "Pull ups" in current_prediction:
                    if "down" in current_prediction:
                        current_stage = "down"
                    elif "(up)" in current_prediction and current_stage == "down":
                        current_stage = "up"
                        counter += 1
                
                elif "Sit up" in current_prediction:
                    if "down" in current_prediction:
                        current_stage = "down"
                    elif "(up)" in current_prediction and current_stage == "down":
                        current_stage = "up"
                        counter += 1
                
                elif "Jumping jacks" in current_prediction:
                    if "down" in current_prediction:
                        current_stage = "down"
                    elif "(up)" in current_prediction and current_stage == "down":
                        current_stage = "up"
                        counter += 1

    # --- DISPLAY UI ---
    # 1. TOP BOX: Exercise Name & Status
    cv2.rectangle(frame, (0, 0), (400, 80), (245, 117, 16), -1)
    
    # Show Class (Row 1)
    cv2.putText(frame, current_prediction, (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show Confidence and Stage together (Row 2)
    cv2.putText(frame, f"Conf: {confidence*100:.1f}%  |  Stage: {current_stage}", (15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # 2. REP COUNTER BOX: Right underneath
    # A smaller, slightly darker box just for the numbers
    cv2.rectangle(frame, (0, 80), (120, 180), (200, 100, 10), -1) 
    
    # "REPS" Label
    cv2.putText(frame, "REPS", (25, 105), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Actual Counter (Giant Font)
    cv2.putText(frame, str(counter), (25, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

    # --- DISPLAY ---
    cv2.imshow('AI Workout Assistant', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()