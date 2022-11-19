import shutil
import time
import tkinter as tk
import tkinter.filedialog
from tkinter import StringVar
import os
import PIL.Image, PIL.ImageTk
import cv2
import camera
import logger
import model
import speech_recognizer
import asyncio
import pickle

CLASSIFIERS = ['Auto (best)', 'LinearSVC', 'KNeighbors', 'RandomForest']
PHOTO_BATCH_OPTIONS = ['Take 1', 'Take 10', 'Take 25', 'Take 50']
DELAY_OPTIONS = ['Immediately', '1 sec delay', '3 sec delay', '5 sec delay']


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def open_file():
    return tk.filedialog.askopenfilename()


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
        self.recognizer = speech_recognizer.SpeechRecognizer()
        self.recognition_enabled = False
        self.recognized_text = None

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

        # Initialize logger instance with GUI's text widget as parameter
        self.logger = logger.Logger(app_place_for_text=self.place_for_text)

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
        self.canvas.grid(row=0, column=0, columnspan='4', stick='we')

        self.btn_toggle_speech_rc = tk.Button(self.window, text="MICROPHONE",
                                              command=lambda: self.toggle_speech_recognition(), height=2, width=14,
                                              font=("Arial", 14))
        self.btn_toggle_speech_rc.grid(row=1, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.label_toggle_speech_rc = tk.Label(self.window, text="OFF", font=("Arial", 36))
        self.label_toggle_speech_rc.grid(row=1, column=2, padx=5, pady=5, stick='we')

        self.btn_toggle_count = tk.Button(self.window, text="COUNTING", command=lambda: self.toggle_counting(),
                                          height=2, width=14, font=("Arial", 14), state="disabled")
        self.btn_toggle_count.grid(row=2, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.label_toggle_count = tk.Label(self.window, text="OFF", font=("Arial", 36))
        self.label_toggle_count.grid(row=2, column=2, padx=5, pady=5, stick='we')

        self.chosen_photo_amount_per_click = StringVar()
        self.chosen_photo_amount_per_click.set(PHOTO_BATCH_OPTIONS[0])
        self.cb_photo_amount = tk.OptionMenu(self.window, self.chosen_photo_amount_per_click, *PHOTO_BATCH_OPTIONS)
        self.cb_photo_amount.config(width=12, height=2, font=("Arial", 14))
        self.cb_photo_amount.grid(row=3, column=0, padx=5, pady=5, stick='we')

        self.btn_class_one = tk.Button(self.window, text="CONTRACTED",
                                       command=lambda: self.take_photo_for_class(1,
                                                                                 int(self.chosen_photo_amount_per_click.get()[
                                                                                     5:]), self.chosen_delay.get()),
                                       height=2,
                                       width=14, font=("Arial", 14))
        self.btn_class_one.grid(row=3, column=1, padx=5, pady=5, stick='we')

        self.btn_class_two = tk.Button(self.window, text="EXTENDED",
                                       command=lambda: self.take_photo_for_class(2,
                                                                                 int(self.chosen_photo_amount_per_click.get()[
                                                                                     5:]), self.chosen_delay.get()),
                                       height=2,
                                       width=14, font=("Arial", 14))
        self.btn_class_two.grid(row=3, column=2, padx=5, pady=5, stick='we')

        self.chosen_classifier = StringVar()
        self.chosen_classifier.set(CLASSIFIERS[0])
        self.cb_classifiers = tk.OptionMenu(self.window, self.chosen_classifier, *CLASSIFIERS)
        self.cb_classifiers.config(width=12, height=2, font=("Arial", 14))
        self.cb_classifiers.grid(row=4, column=0, padx=5, pady=5, stick='we')

        self.btn_train = tk.Button(self.window, text="TRAIN MODEL", command=self.toggle_train_model, height=2,
                                   width=14, font=("Arial", 14))
        self.btn_train.grid(row=4, column=1, padx=5, pady=5, stick='we')

        self.label_toggle_train = tk.Label(self.window, text="UNTRAINED", height=2, width=10, font=("Arial", 18))
        self.label_toggle_train.grid(row=4, column=2, padx=5, pady=5, stick='we')

        self.btn_load_model = tk.Button(self.window, text="LOAD MODEL", command=self.load_model, height=2,
                                        width=14, font=("Arial", 14))
        self.btn_load_model.grid(row=4, column=3, padx=5, pady=5, stick='we')

        self.btn_load_model = tk.Button(self.window, text="SAVE MODEL", command=self.save_model, height=2,
                                        width=14, font=("Arial", 14))
        self.btn_load_model.grid(row=3, column=3, padx=5, pady=5, stick='we')

        self.chosen_delay = StringVar()
        self.chosen_delay.set(DELAY_OPTIONS[0])
        self.cb_delay_options = tk.OptionMenu(self.window, self.chosen_delay, *DELAY_OPTIONS)
        self.cb_delay_options.config(width=12, height=2, font=("Arial", 14))
        self.cb_delay_options.grid(row=5, column=3, padx=5, pady=5, stick='we')

        self.btn_reset_photos = tk.Button(self.window, text="RESET PHOTOS", command=self.reset_photos, height=2,
                                          width=14,
                                          font=("Arial", 14))
        self.btn_reset_photos.grid(row=5, column=0, padx=5, pady=5, stick='we')

        self.btn_reset = tk.Button(self.window, text="RESET ALL", command=self.reset, height=2, width=14,
                                   font=("Arial", 14))
        self.btn_reset.grid(row=5, column=1, columnspan='2', padx=5, pady=5, stick='we')

        self.label_rep_counter = tk.Label(self.window, text=f"REPS: {self.rep_counter}", font=("Arial", 40))
        self.label_rep_counter.grid(row=6, column=0, padx=5, pady=20, columnspan='3', stick='we')

        self.place_for_text = tk.Text(self.window, width=45, height=38, font=("Helvetica", 14), state='disabled')
        self.place_for_text.grid(row=0, column=4, padx=5, pady=5, rowspan='6', stick='we')

        self.btn_clean = tk.Button(self.window, text="CLEAN", command=self.clean, height=2, font=("Arial", 14))
        self.btn_clean.grid(row=6, column=4, padx=5, pady=5, stick='we')

    def execute_recognized_text(self, recognized_text):
        self.logger.log_message("Executing recognized text...")
        if recognized_text == "first class":
            self.take_photo_for_class(1, int(self.chosen_photo_amount_per_click.get()[5:]))
        elif recognized_text == "second class":
            self.take_photo_for_class(2, int(self.chosen_photo_amount_per_click.get()[5:]))
        elif recognized_text in ["train model", "train"]:
            self.toggle_train_model()
        elif recognized_text == "count":
            if self.model_trained:
                self.toggle_counting()
            else:
                self.logger.log_message(message="Can't start counting until model is trained",
                                        msg_type='warning')
        elif recognized_text == "reset":
            self.reset()
        elif recognized_text == "clean":
            self.clean()
        else:
            self.logger.log_message(
                message="Try once more, you can use phrases like:\nfirst class, second class, train model, count, reset, clean",
                msg_type='warning')

    @background
    def toggle_speech_recognition(self):
        self.recognition_enabled = True
        self.logger.log_message(f"Speech recognition has been enabled")
        if self.recognition_enabled:
            try:
                self.recognized_text = self.recognizer.recognize_speech()
                self.logger.log_message(f"You said: {self.recognized_text}")
            except Exception:
                self.logger.log_message(message="Couldn't recognize text, push the button once more",
                                        msg_type='warning')
        self.recognition_enabled = False
        self.logger.log_message(f"Speech recognition has been disabled")

    @background
    def toggle_train_model(self):
        try:
            self.logger.log_message(message=f"Training model using {self.chosen_classifier.get()} classifier",
                                    msg_type='success')
            self.model.train_model(self.chosen_classifier.get(), self.counters)
            self.model_trained = True
            self.btn_toggle_count['state'] = 'active'
            self.logger.log_message(message=f"Model successfully trained using {str(self.model.model)[:-2]}", msg_type='success')
        except Exception:
            self.logger.log_message(message="Couldn't train model, take photo for both classes", msg_type='warning')

    def load_model(self):
        model_filename = open_file()
        with open(model_filename, 'rb') as file:
            self.model = pickle.load(file)

        self.model_trained = True
        self.btn_toggle_count['state'] = 'active'

    def save_model(self):
        model_filename = tk.filedialog.asksaveasfilename()
        with open(model_filename, 'wb') as file:
            pickle.dump(self.model, file)

    def update(self):
        # Toggle counting
        if self.counting_enabled:
            self.predict()

        if self.recognized_text:
            self.execute_recognized_text(self.recognized_text)
            self.recognized_text = None

        # Rep is done, increment counter, write log message
        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1
            self.logger.log_message(message="+1 rep", msg_type='success')

        # Update labels
        if self.counting_enabled:
            self.label_toggle_count.config(text="ON")
        else:
            self.label_toggle_count.config(text="OFF")

        if self.recognition_enabled:
            self.label_toggle_speech_rc.config(text="ON")
        else:
            self.label_toggle_speech_rc.config(text="OFF")

        if self.model_trained:
            self.label_toggle_train.config(text="TRAINED")
        else:
            self.label_toggle_train.config(text="UNTRAINED")

        # Update number of photos taken for each class
        self.btn_class_one['text'] = f"CONTRACTED ({self.counters[0]})"
        self.btn_class_two['text'] = f"EXTENDED ({self.counters[1]})"

        # Update number of reps
        self.label_rep_counter.config(text=f"REPS: {self.rep_counter}")

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
            self.logger.log_message(message="Counting has been disabled")
        else:
            self.logger.log_message(message="Counting has been enabled")
        self.counting_enabled = not self.counting_enabled

    @background
    def take_photo_for_class(self, class_num, amount, delay):
        if delay == '1 sec delay':
            self.countdown(1)
        elif delay == '3 sec delay':
            self.countdown(3)
        elif delay == '5 sec delay':
            self.countdown(5)
        else:
            pass

        for i in range(amount):
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
            if amount == 1:
                self.logger.log_message(message=f"Image for the class {class_num} has been saved")
            else:
                self.logger.log_message(message=f"Image for the class {class_num} has been saved ({i + 1}/{amount})")

    def reset(self):
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
        self.recognition_enabled = False
        self.logger.log_message("All data has been reset", msg_type='warning')

    def reset_photos(self):
        if os.path.exists("1"):
            shutil.rmtree("1")
        if os.path.exists("2"):
            shutil.rmtree("2")
        self.counters = [0, 0]
        self.logger.log_message("Photos has been reset", msg_type='warning')

    def clean(self):
        self.place_for_text.config(state='normal')
        self.place_for_text.delete('1.0', tk.END)
        self.logger.log_message(message="Text widget has been cleaned", msg_type='default')
        self.place_for_text.config(state='disabled')

    def countdown(self, time_sec):
        self.logger.log_message(f"Photo will be taken in {time_sec} seconds")
        while time_sec:
            self.logger.log_message(f"{time_sec}...")
            time.sleep(1)
            time_sec -= 1