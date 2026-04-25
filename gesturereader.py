import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import Counter

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

actions = np.array(['yes', 'no', 'bye', 'my', 'hi'])
model_path = 'model.tflite'

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
sequence = []

# --- ПАРАМЕТРЫ СТАБИЛИЗАЦИИ ---
COOLDOWN_FRAMES = 40       # пауза между предсказаниями (кадры)
VOTE_BUFFER_SIZE = 5       # сколько предсказаний накапливаем для голосования
CONFIDENCE_THRESHOLD = 0.7 # минимальная уверенность

cooldown_counter = 0
vote_buffer = []           # буфер последних предсказаний
current_gesture = ""       # последний подтверждённый жест

print("Нажми 'q' для выхода. Камера запускается...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Отсчёт кулдауна
    if cooldown_counter > 0:
        cooldown_counter -= 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            res = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            sequence.append(res)
            sequence = sequence[-30:]

            if len(sequence) == 30 and cooldown_counter == 0:
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])[0]

                confidence = prediction[np.argmax(prediction)]
                predicted_gesture = actions[np.argmax(prediction)]

                if confidence > CONFIDENCE_THRESHOLD:
                    vote_buffer.append(predicted_gesture)
                    vote_buffer = vote_buffer[-VOTE_BUFFER_SIZE:]

                    # Подтверждаем жест только если большинство голосов совпало
                    most_common, count = Counter(vote_buffer).most_common(1)[0]
                    if count >= 3 and most_common != current_gesture:
                        current_gesture = most_common
                        cooldown_counter = COOLDOWN_FRAMES
                        vote_buffer = []  # сбрасываем буфер после подтверждения
                        print(f"Жест: {current_gesture}")

    else:
        # Рука пропала — сбрасываем всё
        sequence = []
        vote_buffer = []
        current_gesture = ""

    # Отображение жеста и кулдауна
    if current_gesture:
        cv2.putText(frame, f'GESTURE: {current_gesture}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Прогресс-бар кулдауна
    if cooldown_counter > 0:
        bar_width = int((cooldown_counter / COOLDOWN_FRAMES) * 300)
        cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), (0, 165, 255), -1)
        cv2.putText(frame, "Ожидание...", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    cv2.imshow('AI Gesture Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
