import face_recognition
import numpy as np
import os

# Путь к директории с известными лицами
known_faces_dir = "/home/r2d2/PycharmProjects/know_faces"

# Получите список поддиректорий (по одной поддиректории на каждую личность)
known_people = [d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d))]

known_faces = []
known_embeddings = []

for person_dir in known_people:
    person_path = os.path.join(known_faces_dir, person_dir)
    person_images = [os.path.join(person_path, f) for f in os.listdir(person_path) if
                     os.path.isfile(os.path.join(person_path, f))]

    person_face_encodings = []

    for image_file in person_images:
        image = face_recognition.load_image_file(image_file)
        face_encoding = face_recognition.face_encodings(image)

        if len(face_encoding) > 0:
            person_face_encodings.append(face_encoding[0])

    if person_face_encodings:
        # Усредните эмбеддинги для лица (если есть несколько изображений)
        average_embedding = np.mean(person_face_encodings, axis=0)
        known_faces.append(person_dir)
        known_embeddings.append(average_embedding)

# Сохраните массивы известных лиц и эмбеддингов в файл "known_faces.npz"
np.savez("known_faces.npz", faces=known_faces, embeddings=known_embeddings)
