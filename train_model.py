import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import joblib

# === Config ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "asl_mediapipe_data")
samples_per_letter = 100
os.makedirs(data_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# === Data Collection ===
for ascii_code in range(65, 91):  # A-Z
    label = chr(ascii_code)
    label_folder = os.path.join(data_dir, label)
    os.makedirs(label_folder, exist_ok=True)
    print(f"\n👉 Show sign '{label}' — Press 'C' to capture, 'Q' to skip")

    cap = cv2.VideoCapture(0)
    count = len([f for f in os.listdir(label_folder) if f.endswith(".npy")])

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

        cv2.putText(frame, f"{label}: {count}/{samples_per_letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("ASL Capture", frame)

        key = cv2.waitKey(1)
        if key == ord('c') and result.multi_hand_landmarks and count < samples_per_letter:
            landmarks = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            np.save(os.path.join(label_folder, f"{count}.npy"), landmarks)

            # Save cropped hand image
            x_min_crop = max(0, x_min - 20)
            y_min_crop = max(0, y_min - 20)
            x_max_crop = min(w, x_max + 20)
            y_max_crop = min(h, y_max + 20)
            cropped_hand = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            cv2.imwrite(os.path.join(label_folder, f"{count}.jpg"), cropped_hand)

            print(f"📸 Saved: {label}/{count}")
            count += 1

        elif key == ord('q') or count == samples_per_letter:
            print(f"✅ Finished '{label}'")
            break

    cap.release()
    cv2.destroyAllWindows()

# === Model Training ===
print("\n📦 Loading data...")
X, y = [], []
for index, label in enumerate([chr(i) for i in range(65, 91)]):
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            X.append(np.load(os.path.join(folder, file)))
            y.append(index)

X, y = np.array(X), np.array(y)

print("🤖 Training KNN...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy*100:.2f}%")

model_path = os.path.join(base_dir, "knn_asl_model.joblib")
joblib.dump(model, model_path)
print(f"💾 Model saved: {model_path}")

# === Plot 1: Accuracy Bar Chart ===
plt.figure(figsize=(6, 4))
plt.bar(['KNN Model'], [accuracy * 100], color='lightgreen')
plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig(os.path.join(base_dir, "knn_accuracy.png"))
plt.show()

# === Plot 2: Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[chr(i) for i in range(65, 91)])
disp.plot(xticks_rotation='vertical', cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "knn_confusion_matrix.png"))
plt.show()

# === Plot 3: Class-wise Accuracy ===
class_accuracies = []
for i in range(26):
    class_indices = np.where(y_test == i)[0]
    if len(class_indices) > 0:
        class_acc = np.mean(y_pred[class_indices] == y_test[class_indices])
        class_accuracies.append(class_acc * 100)
    else:
        class_accuracies.append(0)

plt.figure(figsize=(12, 6))
plt.bar([chr(i) for i in range(65, 91)], class_accuracies, color='skyblue')
plt.title("Class-wise Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.savefig(os.path.join(base_dir, "knn_classwise_accuracy.png"))
plt.show()
