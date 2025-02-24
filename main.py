import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import logging
import matplotlib.pyplot as plt
import pandas as pd  # Додаємо pandas для обробки звітності

# Налаштування логування
logging.basicConfig(filename="gesture.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Налаштування matplotlib для live plotting
plt.ion()
fig, ax = plt.subplots()
stability_times = []
stability_values = []
(line_plot,) = ax.plot(stability_times, stability_values, 'b-', label="Стабільність")
ax.set_xlabel("Час (с)")
ax.set_ylabel("Стабільність (%)")
ax.set_title("Реальний графік стабільності розпізнавання")
ax.legend()

# Глобальна змінна для порогового значення відкритості (наприклад, 0.6)
openness_threshold = 0.6


def update_threshold(x):
    global openness_threshold
    # x змінюється від 0 до 1000; масштаб до [0, 1]
    openness_threshold = x / 1000.0


# Функція для обчислення коефіцієнта відкритості руки
def compute_openness(landmarks):
    """
    Обчислює відношення середньої відстані від зап’ястя (landmark 0)
    до кінчиків пальців (landmarks 4, 8, 12, 16, 20) до розміру руки.
    Розмір руки визначається як максимальна ширина або висота bounding box.
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y])
    fingertips_indices = [4, 8, 12, 16, 20]
    distances = []
    xs = []
    ys = []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
    for idx in fingertips_indices:
        tip = np.array([landmarks[idx].x, landmarks[idx].y])
        distances.append(np.linalg.norm(tip - wrist))
    avg_distance = np.mean(distances)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    hand_size = max(max_x - min_x, max_y - min_y)
    # Захищаємося від ділення на нуль
    if hand_size == 0:
        return 0
    return avg_distance / hand_size


# Функція класифікації на основі відкритості руки
def classify_gesture_by_openness(landmarks, threshold):
    ratio = compute_openness(landmarks)
    # Обчислюємо "схожість" як відсоток, де 100% – коли ratio == threshold.
    # Чим більше відхилення, тим менше схожість.
    similarity = max(0, min(100, (1 - abs(ratio - threshold) / threshold) * 100))
    # Якщо коефіцієнт відкритості менший за поріг – вважаємо, що це "Б"
    if ratio < threshold:
        return "Б", True, ratio, similarity
    else:
        return "Не Б", False, ratio, similarity


# Ініціалізація MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Ініціалізація вебкамери
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Помилка відкриття камери!")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Створення вікна для відео та trackbar для налаштування порогу відкритості
cv2.namedWindow("Відео")
cv2.createTrackbar("Openness Thresh", "Відео", int(openness_threshold * 1000), 1000, update_threshold)

# Налаштування для запису відео (за потреби)
record_video = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_file = "gesture_output.avi"

# Лог для збереження даних (опційно)
data_log = []  # Запис: час, FPS, gesture, коефіцієнт відкритості, схожість

# Для аналізу стабільності за останню секунду
recent_frames = []  # Список кортежів (timestamp, gesture_label)

start_time = time.time()
prev_frame_time = time.time()
last_console_update_time = time.time()

print("Натисніть 'q' для виходу, 'r' для старт/стоп запису відео, 's' для збереження даних у CSV.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Якщо кадр не прочитано, переходимо до наступного
    frame = cv2.flip(frame, 1)
    current_time = time.time()
    fps = 1 / (current_time - prev_frame_time)
    prev_frame_time = current_time

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # За замовчуванням – "Немає руки"
    gesture_label = "Немає руки"
    recognized = False
    ratio = 0
    similarity = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_label, recognized, ratio, similarity = classify_gesture_by_openness(hand_landmarks.landmark,
                                                                                        openness_threshold)
            cv2.putText(frame, f"Жест: {gesture_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break  # Обробка лише першої руки

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Ratio: {ratio:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Similarity: {similarity:.2f}%", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Запис відео, якщо увімкнено
    if record_video:
        if video_writer is None:
            video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
        video_writer.write(frame)

    cv2.imshow("Відео", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('r'):
        record_video = not record_video
        if not record_video and video_writer is not None:
            video_writer.release()
            video_writer = None
        print("Запис відео:", "Увімкнено" if record_video else "Вимкнено")
    elif key & 0xFF == ord('s'):
        with open('gesture_data_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'fps', 'gesture_label', 'ratio', 'similarity']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data_log:
                writer.writerow(row)
        print("Дані збережено у gesture_data_log.csv")
        logging.info("Дані збережено у gesture_data_log.csv")

    # Запис даних для аналізу
    data_log.append({
        'timestamp': current_time - start_time,
        'fps': fps,
        'gesture_label': gesture_label,
        'ratio': ratio,
        'similarity': similarity
    })

    # Додаємо поточну класифікацію до recent_frames
    recent_frames.append((current_time, gesture_label))
    recent_frames = [(t, g) for (t, g) in recent_frames if current_time - t <= 1]

    # Оновлення консолі кожну секунду: обчислення стабільності
    if current_time - last_console_update_time >= 1:
        if recent_frames:
            total = len(recent_frames)
            count_same = sum(1 for t, g in recent_frames if g == gesture_label)
            stability = (count_same / total) * 100
        else:
            stability = 0
        console_msg = (f"Поточний жест: {gesture_label} | Стабільність: {stability:.2f}% | "
                       f"Openness ratio: {ratio:.2f} | Схожість до 'Б': {similarity:.2f}% | Поріг: {openness_threshold:.3f}")
        print(console_msg)
        logging.info(console_msg)

        # Оновлення графіку стабільності
        stability_times.append(current_time - start_time)
        stability_values.append(stability)
        line_plot.set_data(stability_times, stability_values)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        last_console_update_time = current_time

cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()

plt.ioff()
plt.show()

print("Програма завершена.")
logging.info("Програма завершена.")

# Додаткова звітність результатів
# Формуємо підсумковий звіт за допомогою pandas
df = pd.DataFrame(data_log)
if not df.empty:
    avg_fps = df['fps'].mean()
    avg_ratio = df['ratio'].mean()
    avg_similarity = df['similarity'].mean()
    gesture_counts = df['gesture_label'].value_counts().to_dict()
    total_frames = len(df)

    report_text = (
        f"Звіт про розпізнавання жестів\n"
        f"=====================================\n"
        f"Загальна кількість кадрів: {total_frames}\n"
        f"Середній FPS: {avg_fps:.2f}\n"
        f"Середній коефіцієнт відкритості (ratio): {avg_ratio:.2f}\n"
        f"Середня схожість: {avg_similarity:.2f}%\n"
        f"Кількість розпізнаних жестів: {gesture_counts}\n"
    )

    # Запис звіту у текстовий файл
    with open('gesture_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    logging.info("Підсумковий звіт збережено у gesture_report.txt")
else:
    print("Немає даних для звітності.")
