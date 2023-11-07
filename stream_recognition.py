import face_recognition
import cv2
import dlib
import numpy as np
import queue
import time
import socket
import struct
import pickle
import threading  # Импортируйте библиотеку threading
from datetime import datetime, timedelta


arrival_times = {
    "Vladimir": "08:00",
    "Anton": "08:00",
    "Jeka": "08:00",
    "Paul": "08:00",
    "Rustem": "08:00",
    "Darina": "08:40",
    "Eric": "07:40",
    "Alex": "08:00",
    "Daniel": "08:00",
    "Vasya": "08:00",
    "Artur": "08:00",
    "Sergei": "08:00",

    # и так далее...
}

sent_signals = {}  # Отслеживаем, были ли уже отправлены сигналы для каждого сотрудника

# В начале кода
zoom_factor = 1.0  # Текущий коэффициент масштабирования
zoom_step = 1.0    # Шаг изменения зума (в данном случае, в 2 раза увеличиваем)

# Инициализация блокировки
sent_signals_lock = threading.Lock()

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
threshold = 0.40

last_command_time = 0
command_delay = 5  # Например, 5 секунд между командами

last_face_detected = 0
face_detect_interval = 5  # 5 seconds

# Функция для отправки команд на Raspberry Pi
def send_command():
    rpi_ip = "10.20.36.138" #211 wireless
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



# Создание сокета для приема видеопотока
server_ip = "0.0.0.0"  # Слушаем все интерфейсы
server_port = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)  # Ожидание соединения

print(f"Ожидание соединения от Raspberry Pi ({server_ip}:{server_port})...")

client_socket, client_address = server_socket.accept()
print(f"Подключение от {client_address}")

# Запустите поток для отправки команд на Raspberry Pi
command_thread = threading.Thread(target=send_command)
command_thread.start()

data = b""
payload_size = struct.calcsize("L")

while True:
    while len(data) < payload_size:
        data += client_socket.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Десериализация кадра и применение цифрового зума
    frame = pickle.loads(frame_data)
    frame = cv2.resize(frame, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

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

                with sent_signals_lock:
                    if current_time > expected_arrival and name not in sent_signals:
                        # персона опоздала и сигнал ещё не был отправлен
                        print(f"Отправлен сигнал для {name}")
                        command_queue.put(("fire", datetime.now()))
                        print(command_queue)
                        sent_signals[name] = True  # Указываем, что для этой персоны сигнал уже был отправлен
                        print(sent_signals)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Закрытие сокетов и ожидание завершения потока с командами
client_socket.close()
server_socket.close()
command_thread.join()
command_queue.put("exit")