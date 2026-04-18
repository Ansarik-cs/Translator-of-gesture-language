import cv2
import mediapipe as mp
import numpy as np
import os

# --- НАСТРОЙКИ ---
DATA_PATH = "dataset"
GESTURE_NAME = "my"
SAMPLES_COUNT = 30
SEQUENCE_LENGTH = 30

# Создаем папки
os.makedirs(os.path.join(DATA_PATH, GESTURE_NAME), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,              # <-- было 1, стало 2
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Цвета для каждой руки
HAND_COLORS = {
    "Left":  (255, 100, 0),   # синий
    "Right": (0, 100, 255),   # красный
}

cap = cv2.VideoCapture(0)
print(f"Готов к записи жеста: {GESTURE_NAME}")
print("Нажми 'S', чтобы начать запись. ESC — выход.")

def extract_landmarks(results):
    """
    Возвращает плоский массив 126 точек: [левая рука (63) + правая рука (63)]
    Если рука не найдена — заполняется нулями.
    """
    left_hand  = np.zeros(21 * 3)  # 63 нуля по умолчанию
    right_hand = np.zeros(21 * 3)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" или "Right"
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if label == "Left":
                left_hand = coords
            else:
                right_hand = coords

    return np.concatenate([left_hand, right_hand])  # итого 126 значений


start_recording = False

for sample in range(SAMPLES_COUNT):
    sequence = []

    # --- Ожидание нажатия S ---
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Рисуем обе руки с разными цветами
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                color = HAND_COLORS.get(label, (255, 255, 255))
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=color, thickness=2)
                )
                # Подпись над рукой
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0]) - 20
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Индикатор обнаруженных рук
        hand_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        indicator_color = (0, 255, 0) if hand_count == 2 else (0, 165, 255)
        cv2.putText(frame, f"Hands: {hand_count}/2", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, indicator_color, 2)
        cv2.putText(frame, f"Sample {sample+1}/{SAMPLES_COUNT} | Press 'S' to record",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Collector - Two Hands', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            start_recording = True
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # --- Запись ---
    if start_recording:
        print(f"Запись образца {sample+1}... (показывай обе руки!)")

        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Рисуем руки во время записи
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    color = HAND_COLORS.get(label, (255, 255, 255))
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3),
                        mp_draw.DrawingSpec(color=color, thickness=2)
                    )

            # Извлекаем 126 точек (обе руки)
            frame_data = extract_landmarks(results)
            sequence.append(frame_data)

            # Прогресс записи
            progress = int((frame_num / SEQUENCE_LENGTH) * 300)
            cv2.rectangle(frame, (10, 80), (10 + progress, 100), (0, 0, 255), -1)
            cv2.putText(frame, f"REC {frame_num+1}/{SEQUENCE_LENGTH}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Collector - Two Hands', frame)
            cv2.waitKey(1)

        # Сохраняем — теперь shape будет (30, 126) вместо (30, 63)
        file_path = os.path.join(DATA_PATH, GESTURE_NAME, f"{sample}.npy")
        np.save(file_path, np.array(sequence))
        print(f"Сохранено: {file_path} | shape: {np.array(sequence).shape}")
        start_recording = False

cap.release()
cv2.destroyAllWindows()
