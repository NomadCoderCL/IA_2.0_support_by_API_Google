import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Try to import PyQt5, fall back gracefully if not available
try:
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
    PYQT_AVAILABLE = True
except ImportError:
    print("Warning: PyQt5 not available. GUI functionality will be disabled.")
    print("To enable GUI, install PyQt5: pip install PyQt5")
    PYQT_AVAILABLE = False

from models.model import MyModel
from utils import calculate_accuracy
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, IA_NAME

# Try to import Google API, fall back gracefully if not available
try:
    from google_api import predict_google_api
    GOOGLE_API_AVAILABLE = True
except ImportError:
    print("Warning: Google API not available. API functionality will be disabled.")
    print("To enable Google API, install: pip install google-generativeai")
    GOOGLE_API_AVAILABLE = False
    
    def predict_google_api(input_text):
        return f"Echo: {input_text} (Google API not available)"

import os

if PYQT_AVAILABLE:
    class MyApp(QWidget):
        def __init__(self):
            super().__init__()
            self.initUI()
            self.interactions = []

        def initUI(self):
            layout = QVBoxLayout()

            self.chat_display = QTextEdit(self)
            self.chat_display.setReadOnly(True)
            layout.addWidget(self.chat_display)

            self.input_entry = QLineEdit(self)
            layout.addWidget(self.input_entry)

            self.send_button = QPushButton('Send', self)
            self.send_button.clicked.connect(self.send_text)
            layout.addWidget(self.send_button)

            self.setLayout(layout)
            self.setWindowTitle(f'{IA_NAME} Chat Trainer')
            self.show()

        def send_text(self):
            input_text = self.input_entry.text().strip()
            if not input_text:
                return
            
            self.input_entry.clear()
            self.process_input(input_text)

        def process_input(self, input_text):
            try:
                self.log_interaction(input_text, "user")
                response = predict_google_api(input_text)
                self.log_interaction(response, "AI")
                self.chat_display.append(f"User: {input_text}\n{IA_NAME}: {response}\n")
                self.save_interaction(input_text, response)
            except Exception as e:
                error_msg = f"Error processing input: {e}"
                print(error_msg)
                self.chat_display.append(f"Error: {error_msg}\n")

        def log_interaction(self, text, role):
            self.interactions.append({"role": role, "text": text})

        def save_interaction(self, user_input, response):
            try:
                log_entry = {
                    "user_input": user_input,
                    "response": response
                }
                with open("interactions_log.json", "a") as log_file:
                    log_file.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                print(f"Error saving interaction: {e}")
else:
    # Dummy class for when PyQt5 is not available
    class MyApp:
        def __init__(self):
            raise ImportError("PyQt5 not available")

if __name__ == '__main__':
    if not PYQT_AVAILABLE:
        print("Cannot run GUI application without PyQt5")
        print("Install PyQt5 with: pip install PyQt5")
        sys.exit(1)
    
    try:
        app = QApplication(sys.argv)
        ex = MyApp()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

