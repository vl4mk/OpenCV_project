import cv2
import dlib
import face_recognition
import numpy as np
import queue
import time
import socket
import struct
import pickle
from datetime import datetime, timedelta


arrival_times = {
    "Vladimir": "16:00",
    "Anton": "08:15",
    "Jeka": "8:30",
    "Paul": "10:00",
    "Rustem": "08:30",
    "Eric": "08:00",
    # и так далее...
}

sent_signals = {}  # Отслеживаем, были ли уже отправлены сигналы для каждого сотрудника

command_queue = queue.Queue()

# Инициализация детектора лиц
face_detector = dlib.get_frontal_face_detector()

# Инициализация модели для предсказания ключевых точек лица (shape predictor)
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Считывание базы данных с известными лицами и их эмбеддингами
known_data = np.load("known_faces.npz")
known_faces = known_data["faces"]
known_embeddings = known_data["embeddings"]

# Определение порога сравнения
threshold = 0.65

#last_command_time = 0
#command_delay = 1  # Например, 5 секунд между командами

#last_face_detected = 0
#face_detect_interval = 1  # 5 секунд'

rpi_ip = "10.20.36.211"
rpi_port = 6001  # Замените на порт, который вы настроили на Raspberry Pi

# Функция для отправки команд на Raspberry Pi
def send_command():
    rpi_ip = "10.20.36.211"
    rpi_port = 6001
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((rpi_ip, rpi_port))


    while True:
        command, send_time = command_queue.get()
        if command == "exit":
            break

        # Ожидаем, пока не придет время отправки команды
        while datetime.now() < send_time:
            time.sleep(0.1)

        try:
            client_socket.send(command.encode())
        except (BrokenPipeError, socket.error):
            print("Соединение потеряно. Пытаемся переподключиться...")
            client_socket.close()
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((rpi_ip, rpi_port))
            client_socket.send(command.encode())  # Повторная попытка отправить команду после переподключения

    client_socket.close()

# IP-адрес и порт IP-камеры Lorex
ip_address = "10.20.37.161"
port = 554  # Порт по умолчанию для RTSP-потока
username = "admin"
password = ""

# URL для захвата потока с камеры с учетными данными
url = f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

cap = cv2.VideoCapture(url)

# Проверьте, удалось ли подключиться к потоку
if not cap.isOpened():
    print("Не удалось подключиться к IP-камере.")
    exit()


data = b""
payload_size = struct.calcsize("L")

while True:
    ret, frame = cap.read()  # Захват кадра из видеопотока

    if not ret:
        break

    # Обработка и распознавание лиц в кадре
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for face in faces:
        name = "Unknown"  # Сначала помечаем лицо как неизвестное

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])

        current_time = time.time()
        if len(face_encoding) > 0:
            face_to_check = face_encoding[0]
            distances = face_recognition.face_distance(known_embeddings, face_to_check)
            min_distance = min(distances)

            if min_distance <= threshold:
                index = np.argmin(distances)
                name = known_faces[index]

                current_time = datetime.now().time()
                expected_arrival = datetime.strptime(arrival_times[name], "%H:%M").time()

                if current_time > expected_arrival and name not in sent_signals:
                    # персона опоздала и сигнал ещё не был отправлен
                    command_queue.put(("fire", datetime.now()))
                    sent_signals[name] = True  # Указываем, что для этой персоны сигнал уже был отправлен

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Video", frame)  # Отображение текущего кадра

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Закрытие видеопотока
cv2.destroyAllWindows()  # Закрытие окон OpenCV

