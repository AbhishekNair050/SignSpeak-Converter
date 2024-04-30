import speech_recognition as sr

recognizer = sr.Recognizer()


def audio_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)

    except sr.UnknownValueError:
        print("Could not understand audio")

    except sr.RequestError as e:
        print("Error occurred; {0}".format(e))


audio_to_text()
