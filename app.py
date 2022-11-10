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
        # Initialize tkinter instance and window
        self.window = tk.Tk()
        self.window.title("Rep Counter")

        # Initialize camera instance
        self.camera = camera.Camera()

        # Initialize model and fitness status
        self.model = model.Model()
        self.model_trained = False

        # Initialize speech recognizer instance
        self.recognizer = sr.Recognizer()

        # Counters and toggle counting
        self.counters = [0, 0]
        self.rep_counter = 0
        self.counting_enabled = False

        # States of exercise and last prediction
        self.extended = False
        self.contracted = False
        self.last_prediction = 0

        # Initialize GUI
        self.init_gui()

        # Delay attribute for updating
        self.delay = 15

        # Update function
        self.update()

        # Attributes for tkinter window
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

        self.place_for_text = tk.Text(self.window, width=40, height=38, font=("Helvetica", 14), state='disabled')
        self.place_for_text.grid(row=0, column=2, padx=5, pady=5, rowspan='6', stick='we')

    def recognize_speech(self):
        with sr.Microphone() as source:
            try:
                audio = self.recognizer.listen(source)
                recognized_text = self.recognizer.recognize_google(audio).lower()
                self.log_message(f"You said: {recognized_text}")
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
                        self.log_message(message="Can't start counting until model is trained", msg_type='warning')
                elif recognized_text == "reset":
                    self.reset()
                else:
                    self.log_message(message="Try once more, you can use phrases like:\n"
                                             "first class, second class, train model, count, reset", msg_type='warning')
                    self.recognize_speech()
            except sr.UnknownValueError:
                self.log_message(message="Couldn't recognize, press the button again", msg_type='warning')

    def toggle_train_model(self):
        try:
            self.model.train_model(self.counters)
            self.btn_toggle_count['state'] = 'active'
            self.log_message(message="Model successfully trained", msg_type='success')
        except Exception:
            self.log_message(message="Couldn't train model, take photo for both classes", msg_type='warning')

    def update(self):
        # Toggle counting
        if self.counting_enabled:
            self.predict()

        # Rep is done, increment counter, write log message
        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1
            self.log_message("+1 rep")

        # Update labels and buttons
        if self.counting_enabled:
            self.toggle_count_label.config(text=f"ON")
        else:
            self.toggle_count_label.config(text=f"OFF")

        # Update number of photos taken for each class
        self.btn_class_one['text'] = f"EXTENDED ({self.counters[0]})"
        self.btn_class_two['text'] = f"CONTRACTED ({self.counters[1]})"

        # Update number of reps
        self.counter_label.config(text=f"REPS: {self.rep_counter}")

        # Update image
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
            self.log_message(message="Counting has been disabled")
        else:
            self.log_message(message="Counting has been enabled")
        self.counting_enabled = not self.counting_enabled

    def save_for_class(self, class_num):
        # Get image from camera
        ret, frame = self.camera.get_frame()

        # Create folders for photos
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        # Work with image, save, recolor to gray
        cv2.imwrite(f"{class_num}/frame{self.counters[class_num - 1]}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        # Open image
        img = PIL.Image.open(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")
        # Resize image, image antialiasing
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        # Save updated image
        img.save(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")
        # Increment counters
        self.counters[class_num - 1] += 1
        # Log message
        self.log_message(message=f"Image for the class {class_num} has been saved")

    def reset(self):
        self.log_message("Data has been reset", msg_type='warning')
        if os.path.exists("1"):
            shutil.rmtree("1")
        if os.path.exists("2"):
            shutil.rmtree("2")
        self.btn_toggle_count['state'] = 'disabled'
        self.counters = [0, 0]
        self.rep_counter = 0
        self.extended = False
        self.contracted = False
        self.last_prediction = 0
        self.model_trained = False
        self.counting_enabled = False

    def log_message(self, message, msg_type='default_message'):
        # Print message into a terminal
        print(message)

        # Create and configure tags for warning and success messages
        self.place_for_text.tag_config('warning', foreground='red')
        self.place_for_text.tag_config('success', foreground='green')

        # Activate text widget so we can insert logs
        self.place_for_text.config(state="normal")

        # Warning - red color, success - green color, default - black color
        if msg_type == 'warning':
            self.place_for_text.insert(tk.END, message + '\n', 'warning')
        elif msg_type == 'success':
            self.place_for_text.insert(tk.END, message + '\n', 'success')
        else:
            self.place_for_text.insert(tk.END, message + '\n')

        # Update text widget, deactivate text widget so we can't write there, autoscroll to an end
        self.place_for_text.config(state="disabled")
        self.place_for_text.see('end')
