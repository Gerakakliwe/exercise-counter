import shutil
import time
import tkinter as tk
import os
import PIL.Image, PIL.ImageTk
import cv2
import camera
import model
import speech_recognition as sr


class App:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Rep Counter")

        self.recognizer = sr.Recognizer()

        self.counters = [1, 1]
        self.counter_class_one = 0
        self.counter_class_two = 0
        self.rep_counter = 0

        self.extended = False
        self.contracted = False
        self.last_prediction = 0

        self.model = model.Model()
        self.model_trained = False

        self.counting_enabled = False

        self.camera = camera.Camera()

        self.init_gui()
        self.events = []
        self.place_for_text.config(state="normal")
        self.place_for_text.insert(tk.END, ' ')
        self.place_for_text.config(state="disabled")

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.config(bg='#dcdcdc')
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.grid(row=0, column=0, columnspan='2', stick='we')

        self.btn_toggle_speech_rc = tk.Button(self.window, text="TOGGLE SPEECH RECOGNITION",
                                              command=self.recognize_speech,
                                              height=2, font=("Arial", 14))
        self.btn_toggle_speech_rc.grid(row=1, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.btn_toggle_count = tk.Button(self.window, text="TOGGLE COUNTING", command=self.toggle_counting, width=10,
                                          height=2, font=("Arial", 14), state="disabled")
        self.btn_toggle_count.grid(row=2, column=0, padx=5, pady=5, stick='we')

        self.toggle_count_label = tk.Label(self.window, text="OFF", width=4, font=("Arial", 36))
        self.toggle_count_label.grid(row=2, column=1, padx=5, pady=5, stick='we')

        self.btn_class_one = tk.Button(self.window, text="EXTENDED", command=lambda: self.save_for_class(1), width=10,
                                       height=2, font=("Arial", 14))
        self.btn_class_one.grid(row=3, column=0, padx=5, pady=5, stick='we')

        self.btn_class_two = tk.Button(self.window, text="CONTRACTED", command=lambda: self.save_for_class(2), width=4,
                                       height=2, font=("Arial", 14))
        self.btn_class_two.grid(row=3, column=1, padx=5, pady=5, stick='we')

        self.btn_train = tk.Button(self.window, text="TRAIN MODEL",
                                   command=lambda: self.toggle_train_model(), height=2, font=("Arial", 14))
        self.btn_train.grid(row=4, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.btn_reset = tk.Button(self.window, text="RESET", command=self.reset, height=2, font=("Arial", 14))
        self.btn_reset.grid(row=5, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.counter_label = tk.Label(self.window, text=f"REPS: {self.rep_counter}", font=("Arial", 40))
        self.counter_label.grid(row=6, column=0, padx=5, pady=20, columnspan='3', stick='we')

        self.place_for_text = tk.Text(self.window, width=55, height=53, state='disabled')
        self.place_for_text.grid(row=0, column=2, padx=5, pady=5, rowspan='6', stick='we')

    def recognize_speech(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source)
                recognized_text = self.recognizer.recognize_google(audio).lower()
                self.log_message(f"You have said: {recognized_text}")
                if recognized_text == "first class":
                    for i in range(20):
                        time.sleep(0.05)
                        self.save_for_class(1)
                elif recognized_text == "second class":
                    for i in range(20):
                        time.sleep(0.05)
                        self.save_for_class(2)
                elif recognized_text == "train model":
                    self.toggle_train_model()
                elif recognized_text == "count":
                    if self.btn_toggle_count['state'] == 'active':
                        self.toggle_counting()
                    else:
                        self.log_message("Can't start counting until model is trained")
                elif recognized_text == "reset":
                    self.reset()
                else:
                    self.log_message("Try once more, you can use phrases like:\n"
                                     "first class, second class, train model, count, reset")
                    self.recognize_speech()
            except sr.UnknownValueError:
                self.log_message("Couldn't recognize, press the button again")

    def toggle_train_model(self):
        self.model.train_model(self.counters)
        self.btn_toggle_count['state'] = 'active'
        self.log_message("Model successfully trained")

    def update(self):
        if self.counting_enabled:
            self.predict()

        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1
            self.log_message("+1")

        # Update labels and buttons
        if self.counting_enabled:
            self.toggle_count_label.config(text=f"ON")
        else:
            self.toggle_count_label.config(text=f"OFF")

        self.btn_class_one['text'] = f"EXTENDED ({self.counter_class_one})"
        self.btn_class_two['text'] = f"CONTRACTED ({self.counter_class_two})"

        self.counter_label.config(text=f"REPS: {self.rep_counter}")

        ret, frame = self.camera.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        frame = self.camera.get_frame()
        prediction = self.model.predict(frame)

        if prediction != self.last_prediction:
            if prediction == 1:
                self.extended = True
                self.last_prediction = 1
            if prediction == 2:
                self.contracted = True
                self.last_prediction = 2

    def toggle_counting(self):
        if self.counting_enabled:
            self.log_message("Counting has been disabled")
        else:
            self.log_message("Counting has been enabled")
        self.counting_enabled = not self.counting_enabled

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        if class_num == 1:
            self.counter_class_one += 1
        else:
            self.counter_class_two += 1

        cv2.imwrite(f"{class_num}/frame{self.counters[class_num - 1]}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = PIL.Image.open(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")

        self.log_message(f"Image for the class {class_num} has been saved")

        self.counters[class_num - 1] += 1

    def reset(self):
        self.log_message("Model and counters has been reset")
        if os.path.exists("1"):
            shutil.rmtree("1")
        if os.path.exists("2"):
            shutil.rmtree("2")
        self.btn_toggle_count['state'] = 'disabled'
        self.counters = [1, 1]
        self.counter_class_one = 0
        self.counter_class_two = 0
        self.rep_counter = 0
        self.extended = False
        self.contracted = False
        self.last_prediction = 0
        self.model_trained = False
        self.counting_enabled = False

    def log_message(self, message):
        print(message)
        if len(self.events) == 50:
            self.events.pop(0)
        self.events.append(message+'\n')
        self.place_for_text.config(state="normal")
        self.place_for_text.delete('1.0', tk.END)
        self.place_for_text.insert(tk.END, self.events)
        self.place_for_text.config(state="disabled")
