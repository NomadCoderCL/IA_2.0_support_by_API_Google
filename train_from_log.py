import torch
import torch.nn as nn
import torch.optim as optim
from models.model import MyModel
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
import json

def train_model(interactions, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for interaction in interactions:
            user_input = interaction["user_input"]
            response = interaction["response"]
            input_tensor = torch.tensor([ord(c) for c in user_input], dtype=torch.float32).unsqueeze(0)
            target_tensor = torch.tensor([ord(c) for c in response], dtype=torch.long).unsqueeze(0)
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output.view(-1, OUTPUT_SIZE), target_tensor.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(interactions)}")
    torch.save(model.state_dict(), 'model.pth')
if __name__ == "__main__":
    with open("interactions_log.json", "r") as log_file:
        interactions = [json.loads(line) for line in log_file]
    model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_model(interactions, model, criterion, optimizer)
