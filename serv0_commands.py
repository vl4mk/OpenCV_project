import socket

# IP и порт Raspberry Pi
rpi_ip = "10.20.36.138"
rpi_port = 6001  # Замените на порт, который вы настроили на Raspberry Pi

# Создайте сокет и подключитесь к Raspberry Pi
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((rpi_ip, rpi_port))

# Отправьте команду
command = "fire"
client_socket.send(command.encode())

# Закройте сокет
client_socket.close()
