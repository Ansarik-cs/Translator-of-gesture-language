import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- НАСТРОЙКИ ---
DATA_PATH = "dataset"  # Папка для данных
GESTURE_NAME = "yes"   # Название жеста, который записываем
SAMPLES_COUNT = 30     # Сколько раз запишем этот жест (по 50 лучше, но начни с 30)
SEQUENCE_LENGTH = 30   # Сколько кадров в одном жесте (1 секунда)

# Создаем папки
if not os.path.exists(DATA_PATH):
    os.makedirs(os.path.join(DATA_PATH, GESTURE_NAME))

# Инициализируем MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print(f"Готов к записи жеста: {GESTURE_NAME}")
print("Нажми 'S', чтобы начать запись серии.")

start_recording = False

for sample in range(SAMPLES_COUNT):
    sequence = []
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) # Зеркалим
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Рисуем точки для удобства
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.putText(frame, f"Sample {sample+1}/{SAMPLES_COUNT}. Press 'S' to start this gesture", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Collector', frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            start_recording = True
            break
        if key == 27: # Esc для выхода
            cap.release()
            cv2.destroyAllWindows()
            exit()

    if start_recording:
        print(f"Запись образца {sample+1}...")
        for frame_num in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Извлекаем координаты
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                # Собираем все 21 точку (x, y, z) в один плоский список
                res = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
                sequence.append(res)
            else:
                # Если рука пропала из кадра, заполняем нулями (чтобы массив не сломался)
                sequence.append(np.zeros(21*3))
            
            cv2.putText(frame, f"RECORDING FRAME {frame_num}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Collector', frame)
            cv2.waitKey(1)

        # Сохраняем результат
        file_path = os.path.join(DATA_PATH, GESTURE_NAME, f"{sample}.npy")
        np.save(file_path, sequence)
        print(f"Сохранено в {file_path}")
        start_recording = False

cap.release()
cv2.destroyAllWindows()