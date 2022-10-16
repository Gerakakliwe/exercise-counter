import tkinter as tk
import os
import PIL.Image, PIL.ImageTk
import cv2
import camera
import model


class App:

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Rep Counter")

        self.counters = [1, 1]
        self.rep_counter = 0

        self.extended = False
        self.contracted = False
        self.last_prediction = 0

        self.model = model.Model()

        self.counting_enabled = False

        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.config(bg='#dcdcdc')
        self.window.mainloop()

    def init_gui(self):
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height)
        self.canvas.grid(row=0, column=0, columnspan='2', stick='we')

        self.btn_toggle_count = tk.Button(self.window, text="TOGGLE COUNTING", command=self.counting_toggle, width=10, height=2, font=("Arial", 14))
        self.btn_toggle_count.grid(row=1, column=0, padx=5, pady=5, stick='we')

        self.toggle_count_label = tk.Label(self.window, text="OFF", width=4, font=("Arial", 36))
        self.toggle_count_label.grid(row=1, column=1, padx=5, pady=5, stick='we')

        self.btn_class_one = tk.Button(self.window, text="EXTENDED", command=lambda: self.save_for_class(1), width=10, height=2, font=("Arial", 14))
        self.btn_class_one.grid(row=2, column=0, padx=5, pady=5, stick='we')

        self.btn_class_two = tk.Button(self.window, text="CONTRACTED", command=lambda: self.save_for_class(2), width=4, height=2, font=("Arial", 14))
        self.btn_class_two.grid(row=2, column=1, padx=5, pady=5, stick='we')

        self.btn_train = tk.Button(self.window, text="TRAIN MODEL",
                                   command=lambda: self.model.train_model(self.counters), height=2, font=("Arial", 14))
        self.btn_train.grid(row=3, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.btn_reset = tk.Button(self.window, text="RESET", command=self.reset, height=2, font=("Arial", 14))
        self.btn_reset.grid(row=4, column=0, columnspan='2', padx=5, pady=5, stick='we')

        self.counter_label = tk.Label(self.window, text=f"REPS: {self.rep_counter}", font=("Arial", 40))
        self.counter_label.grid(row=5, column=0, padx=5, pady=20, columnspan='2',  stick='we')

    def update(self):
        if self.counting_enabled:
            self.predict()

        if self.extended and self.contracted:
            self.extended, self.contracted = False, False
            self.rep_counter += 1

        self.counter_label.config(text=f"REPS: {self.rep_counter}")
        if self.counting_enabled:
            self.toggle_count_label.config(text=f"ON")
        else:
            self.toggle_count_label.config(text=f"OFF")

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

    def counting_toggle(self):
        if self.counting_enabled:
            print("Counting has been disabled")
        else:
            print("Counting has been enabled")
        self.counting_enabled = not self.counting_enabled

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        cv2.imwrite(f"{class_num}/frame{self.counters[class_num - 1]}.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
        img = PIL.Image.open(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save(f"{class_num}/frame{self.counters[class_num - 1]}.jpg")

        self.counters[class_num - 1] += 1

    def reset(self):
        print("Counter has been reset")
        self.rep_counter = 0
