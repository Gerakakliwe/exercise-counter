import speech_recognition as sr


def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        while True:
            try:
                audio = recognizer.listen(source)
                print(recognizer.recognize_google(audio))
            except sr.UnknownValueError:
                print("Couldn't recognize")
                continue
