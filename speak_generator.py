from gtts import gTTS
import os

# Текст для преобразования в речь
text_to_speak = "Привет, как дела?"

# Создание объекта gTTS
tts = gTTS(text_to_speak, lang='ru')

# Сохранение аудиофайла
tts.save("output.mp3")

# Воспроизведение аудиофайла
os.system("mpg123 output.mp3")
