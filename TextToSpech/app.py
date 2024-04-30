from flask import Flask, request, render_template
from gtts import gTTS
import pygame
import os
from googletrans import Translator, LANGUAGES

app = Flask(__name__)


def get_language_code(language_name):
    language_name = language_name.lower().strip()
    for code, name in LANGUAGES.items():
        if name.lower() == language_name:
            return code
    return None


@app.route("/")
def index():
    return render_template("index.html", translated_text=None)


@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        user_text = request.form["user_text"]
        target_language_name = request.form["target_language"]

        # Get language code from language name
        target_language_code = get_language_code(target_language_name)
        if not target_language_code:
            return "Error: Language not supported."

        translator = Translator()
        translated_text = translator.translate(
            user_text, dest=target_language_code
        ).text
        tts = gTTS(text=translated_text, lang=target_language_code, slow=False)

        file_path = "translated_audio.mp3"
        tts.save(file_path)
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        pygame.mixer.quit()
        os.remove(file_path)

        return render_template("index.html", translated_text=translated_text)
    else:
        return "Error: Method not allowed."


if __name__ == "__main__":
    app.run(debug=True)
