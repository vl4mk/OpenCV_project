import face_recognition
import cv2
import dlib
import numpy as np
import queue
import time
import socket
import struct
import pickle
import threading
from datetime import datetime, timedelta


arrival_times = {
    "Vladimir": "08:00",
    "Anton": "08:00",
    "Jeka": "08:00",
    "Paul": "18:00",
    "Rustem": "08:30",
    "Darina": "09:00",
    "Eric": "08:00",
    "Alex": "08:00",
    "Daniel": "08:00",
    "Vasya": "18:00",
    "Artur": "18:00",
    "Sergei": "18:00",
    "Kirill": "18:00",

}


sent_signals = {}  # We track whether signals have already been sent for each employee

# Initialize lock
sent_signals_lock = threading.Lock()

command_queue = queue.Queue()

# Initialize the face detector
face_detector = dlib.get_frontal_face_detector()

# Initialize the model to predict key points of the face (shape predictor)
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Reading a database of famous faces and their embeddings
known_data = np.load("known_faces.npz")
known_faces = known_data["faces"]
known_embeddings = known_data["embeddings"]

# Determine the comparison threshold
threshold = 0.48


def reset_sent_signals_daily():
    while True:
        # Определите, сколько времени осталось до полуночи
        now = datetime.now()
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        sleep_time = (midnight - now).total_seconds()

        # Ждать до полуночи
        time.sleep(sleep_time)

        # Безопасно очистить словарь sent_signals
        with sent_signals_lock:
            print("Lock acquired reset_sent_signals_daily")
            sent_signals.clear()
            print("Sent signals reset for the day.")
            print(sent_signals)
            print("Lock released reset_sent_signals_daily ")


thread = threading.Thread(target=reset_sent_signals_daily)
#thread.daemon = True  # Это позволяет программе завершиться, даже если поток активен
thread.start()

# Function to send commands to Raspberry Pi
def send_command():
    rpi_ip = "10.20.36.138" #211 wireless
    rpi_port = 6001
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((rpi_ip, rpi_port))

    while True:
        command, send_time, name = command_queue.get()
        if command == "exit":
            break

        # Wait until it's time to send the command
        while datetime.now() < send_time:
            time.sleep(0.1)

        # Формирование сообщения, объединяющего команду и имя
        message = f"{command} {name}"
        client_socket.send(message.encode())

        # Переподключение в случае потери соединения
        print("Connection lost. Trying to reconnect...")
        client_socket.close()
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((rpi_ip, rpi_port))
        client_socket.send(message.encode())


    client_socket.close()
    print("after closed client socket")


# Create a socket to receive a video stream
server_ip = "0.0.0.0"  # Listen to all interfaces
server_port = 5000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_ip, server_port))
server_socket.listen(1)  # Waiting for connection

print(f"Waiting for connection from Raspberry Pi ({server_ip}:{server_port})...")

client_socket, client_address = server_socket.accept()
print(f"Connecting from {client_address}")

# Start a thread to send commands to the Raspberry Pi
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

    # Deserialize frame
    frame = pickle.loads(frame_data)

    # Processing and recognition of faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for face in faces:
        name = "Unknown"  # First we mark the face as unknown

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
                    if current_time < expected_arrival and name not in sent_signals:
                        print("Lock acquired register sent_signals")
                        print(f"Send register signal for {name}  {datetime.now()} ")
                        command_queue.put(("registered", datetime.now(), name))
                        sent_signals[name] = True
                        print(sent_signals)
                        print("Lock release register sent_signals")
                    elif current_time > expected_arrival and name not in sent_signals:
                        print("Lock acquired late sent_signals")
                        print(f"Send late signal for {name}  {datetime.now()} ")
                        command_queue.put(("fire", datetime.now(), name))
                        sent_signals[name] = True  # Indicate that a signal has already been sent for this person
                        print(sent_signals)
                        print("Lock release late sent_signals")


        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Closing sockets and waiting for the command thread to complete
client_socket.close()
server_socket.close()
command_thread.join()
command_queue.put("exit")