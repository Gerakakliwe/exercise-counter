import tkinter as tk


class Logger:

    # Initialize logger using GUI's text widget
    def __init__(self, app_place_for_text):
        self.place_for_text = app_place_for_text

    def log_message(self, message, msg_type='default'):
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
