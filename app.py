import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit, QPushButton
from models.model import MyModel
from utils import calculate_accuracy
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from google_api import predict_google_api
import os

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
        self.setWindowTitle('NOVA Chat Trainer')
        self.show()

    def send_text(self):
        input_text = self.input_entry.text()
        self.input_entry.clear()
        self.process_input(input_text)

    def process_input(self, input_text):
        self.log_interaction(input_text, "user")
        response = predict_google_api(input_text)
        self.log_interaction(response, "AI")
        self.chat_display.append(f"User: {input_text}\nAI: {response}\n")
        self.save_interaction(input_text, response)

    def log_interaction(self, text, role):
        self.interactions.append({"role": role, "text": text})

    def save_interaction(self, user_input, response):
        log_entry = {
            "user_input": user_input,
            "response": response
        }
        with open("interactions_log.json", "a") as log_file:
            log_file.write(json.dumps(log_entry) + "\n")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())

