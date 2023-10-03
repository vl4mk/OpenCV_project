import face_recognition
import cv2
import dlib
import numpy as np
import os
import time

# Инициализация детектора лиц
face_detector = dlib.get_frontal_face_detector()

# Инициализация модели для предсказания ключевых точек лица (shape predictor)
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Считывание базы данных с известными лицами и их эмбеддингами
known_data = np.load("known_faces.npz")
known_faces = known_data["faces"]
known_embeddings = known_data["embeddings"]

# Определение порога сравнения
threshold = 0.6

# Инициализация видеокамеры
video_capture = cv2.VideoCapture(0)

# Директория для сохранения фотографий "Unknown" лиц
output_directory = "unknown_faces"

# Проверяем, существует ли директория, и создаем её, если она не существует
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

while True:
    ret, frame = video_capture.read()

    # Обнаружение лиц в кадре
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    # Перевод кадра из BGR в RGB (для библиотеки face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for face in faces:
        # Получение координат лица
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Вычисление эмбеддинга для лица
        face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

        # Сравнение эмбеддинга с известными лицами
        if len(face_encoding) > 0:
            face_to_check = face_encoding[0]
            distances = face_recognition.face_distance(known_embeddings, face_to_check)
            min_distance = min(distances)

            if min_distance <= threshold:
                # Если нашли совпадение, определите имя лица
                index = np.argmin(distances)
                name = known_faces[index]
            else:
                # Если совпадений не найдено, установите имя "Unknown"
                name = "Unknown"

                # Сгенерируем уникальное имя для фотографии
                timestamp = int(time.time())
                photo_filename = f"unknown_{timestamp}.jpg"
                photo_path = os.path.join(output_directory, photo_filename)

                # Сохраняем кадр как фотографию "Unknown"
                cv2.imwrite(photo_path, frame)

            # Нарисовать рамку вокруг лица и отобразить имя
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окна
video_capture.release()
cv2.destroyAllWindows()
