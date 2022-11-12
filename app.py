import shutil
import tkinter as tk
import os
import PIL.Image, PIL.ImageTk
import cv2
import camera
import logger
import model
import speech_recognizer
import threading

IMAGES_TO_TAKE = 50


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
        self.canvas.grid(row=0, column=0, columnspan='2', stick='we')

        self.btn_toggle_speech_rc = tk.Button(self.window, text="SPEECH RECOGNITION",
                                              command=self.threading_toggle_speech_recognition, height=2,
                                              font=("Arial", 14))
        self.btn_toggle_speech_rc.grid(row=1, column=0, padx=5, pady=5, stick='we')

        self.label_toggle_speech_rc = tk.Label(self.window, text="OFF", font=("Arial", 36))
        self.label_toggle_speech_rc.grid(row=1, column=1, padx=5, pady=5, stick='we')

        self.btn_toggle_count = tk.Button(self.window, text="COUNTING", command=lambda: self.toggle_counting(),
                                          height=2, font=("Arial", 14), state="disabled")
        self.btn_toggle_count.grid(row=2, column=0, padx=5, pady=5, stick='we')

        self.label_toggle_count = tk.Label(self.window, text="OFF", font=("Arial", 36))
        self.label_toggle_count.grid(row=2, column=1, padx=5, pady=5, stick='we')

        self.btn_class_one = tk.Button(self.window, text="EXTENDED", command=lambda: self.save_for_class(1), height=2,
                                       font=("Arial", 14))
        self.btn_class_one.grid(row=3, column=0, padx=5, pady=5, stick='we')

        self.btn_class_two = tk.Button(self.window, text="CONTRACTED", command=lambda: self.save_for_class(2), height=2,
                                       font=("Arial", 14))
        self.btn_class_two.grid(row=3, column=1, padx=5, pady=5, stick='we')

        self.btn_train = tk.Button(self.window, text="TRAIN MODEL", command=self.threading_toggle_train_model, height=2,
                                   font=("Arial", 14))
        self.btn_train.grid(row=4, column=0, padx=5, pady=5, stick='we')

        self.label_toggle_train = tk.Label(self.window, text="NOT TRAINED", height=2, font=("Arial", 18))
        self.label_toggle_train.grid(row=4, column=1, padx=5, pady=5, stick='we')

        self.btn_reset = tk.Button(self.window, text="RESET", command=self.reset, height=2, font=("Arial", 14))
        self.btn_reset.grid(row=5, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.label_rep_counter = tk.Label(self.window, text=f"REPS: {self.rep_counter}", font=("Arial", 40))
        self.label_rep_counter.grid(row=6, column=0, padx=5, pady=20, columnspan='3', stick='we')

        self.place_for_text = tk.Text(self.window, width=45, height=38, font=("Helvetica", 14), state='disabled')
        self.place_for_text.grid(row=0, column=2, padx=5, pady=5, rowspan='5', stick='we')

        self.btn_clean = tk.Button(self.window, text="CLEAN", command=self.clean, height=2, font=("Arial", 14))
        self.btn_clean.grid(row=5, column=2, padx=5, pady=5, stick='we')

    def threading_toggle_speech_recognition(self):
        speech_recognition_thread = threading.Thread(target=self.toggle_speech_recognition)
        speech_recognition_thread.start()

    def toggle_speech_recognition(self):
        self.recognition_enabled = True
        self.logger.log_message(f"Speech recognition has been enabled")
        if self.recognition_enabled:
            try:
                recognized_text = self.recognizer.recognize_speech()
                self.logger.log_message(f"You said: {recognized_text}")

                if recognized_text == "first class":
                    for i in range(IMAGES_TO_TAKE):
                        self.save_for_class(1, i + 1, IMAGES_TO_TAKE)
                elif recognized_text == "second class":
                    for i in range(IMAGES_TO_TAKE):
                        self.save_for_class(2, i + 1, IMAGES_TO_TAKE)
                elif recognized_text == "train model":
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
                    self.toggle_speech_recognition()
            except Exception:
                self.logger.log_message(message="Couldn't recognize text, push the button once more",
                                        msg_type='warning')
            finally:
                self.recognition_enabled = False
                self.logger.log_message(f"Speech recognition has been disabled")

    def threading_toggle_train_model(self):
        train_model_thread = threading.Thread(target=self.toggle_train_model)
        train_model_thread.start()

    def toggle_train_model(self):
        try:
            self.logger.log_message(message="Training...", msg_type='success')
            self.model.train_model(self.counters)
            self.model_trained = True
            self.btn_toggle_count['state'] = 'active'
            self.logger.log_message(message="Model successfully trained", msg_type='success')
        except Exception:
            self.logger.log_message(message="Couldn't train model, take photo for both classes", msg_type='warning')

    def update(self):
        # Toggle counting
        if self.counting_enabled:
            self.predict()

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
            self.label_toggle_train.config(text="NOT TRAINED")

        # Update number of photos taken for each class
        self.btn_class_one['text'] = f"EXTENDED ({self.counters[0]})"
        self.btn_class_two['text'] = f"CONTRACTED ({self.counters[1]})"

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

    def save_for_class(self, class_num, loop_counter=1, amount=1):
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
            self.logger.log_message(message=f"Image for the class {class_num} has been saved ({loop_counter}/{amount})")

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
        self.logger.log_message("Data has been reset", msg_type='warning')

    def clean(self):
        self.place_for_text.config(state='normal')
        self.place_for_text.delete('1.0', tk.END)
        self.logger.log_message(message="Text widget has been cleaned", msg_type='default')
        self.place_for_text.config(state='disabled')
