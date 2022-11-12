import speech_recognition as sr


class SpeechRecognizer():

    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self):
        recognized_text = '...'
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source)
                recognized_text = self.recognizer.recognize_google(audio).lower()
                message = f"You said {recognized_text}"
                msg_type = 'default'
            except sr.UnknownValueError:
                message = "Couldn't recognize, press the button again"
                msg_type = 'warning'

        return recognized_text, message, msg_type
