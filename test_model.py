import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyttsx3
from collections import deque

base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, "knn_asl_model.joblib"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

engine = pyttsx3.init()
last_predictions = deque(maxlen=15)  # For stability

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        h, w, _ = frame.shape
        x = [lm.x for lm in hand_landmarks.landmark]
        y = [lm.y for lm in hand_landmarks.landmark]
        x_min, x_max = int(min(x)*w), int(max(x)*w)
        y_min, y_max = int(min(y)*h), int(max(y)*h)
        cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max+20), (0, 255, 0), 2)

        features = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_char = chr(65 + prediction)
        last_predictions.append(predicted_char)

        # Speak only if the last few predictions are the same
        if len(set(last_predictions)) == 1 and len(last_predictions) == last_predictions.maxlen:
            engine.say(predicted_char)
            engine.runAndWait()
            last_predictions.clear()

        cv2.putText(frame, f"Prediction: {predicted_char}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
