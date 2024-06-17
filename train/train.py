import torch
import torch.nn as nn
import torch.optim as optim
from models.model import MyModel
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE

def train_model(data):
    model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for inputs, labels in data:
            inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model
