from gtts import gTTS
import pygame
import os
from googletrans import Translator

original_text = " "
target_language = input("Enter language code (e.g., 'en' for English): ")

translator = Translator()
translated_text = translator.translate(original_text, dest=target_language).text
tts = gTTS(text=translated_text, lang=target_language, slow=False)

file_path = "welcome.mp3"
tts.save(file_path)
pygame.mixer.init()
pygame.mixer.music.load(file_path)
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)  # Adjust tick() value for smoother playback
pygame.mixer.quit()

os.remove(file_path)
