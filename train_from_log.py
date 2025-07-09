import torch
import torch.nn as nn
import torch.optim as optim
from models.model import MyModel
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
import json
import numpy as np

def preprocess_text(text, max_length=100):
    """
    Preprocess text by converting to character-level encoding and padding.
    
    Args:
        text (str): Input text
        max_length (int): Maximum sequence length
    
    Returns:
        torch.Tensor: Preprocessed tensor
    """
    # Convert to character codes and normalize to 0-1 range
    chars = [ord(c) / 255.0 for c in text[:max_length]]
    # Pad to max_length
    chars += [0.0] * (max_length - len(chars))
    return torch.tensor(chars, dtype=torch.float32)

def train_model(interactions, model, criterion, optimizer, num_epochs=10):
    """
    Train the model with proper data preprocessing.
    
    Args:
        interactions (list): List of interaction dictionaries
        model (nn.Module): The model to train
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of training epochs
    """
    if not interactions:
        print("Warning: No interactions found for training")
        return
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for interaction in interactions:
            try:
                user_input = interaction["user_input"]
                response = interaction["response"]
                
                # Preprocess input to match model expectations
                input_tensor = preprocess_text(user_input).unsqueeze(0)
                
                # For now, we'll use a simple classification approach
                # Map response to a class (this is a simplified approach)
                # In a real scenario, this would need proper sequence-to-sequence training
                response_class = hash(response) % OUTPUT_SIZE
                target_tensor = torch.tensor([response_class], dtype=torch.long)
                
                optimizer.zero_grad()
                output = model(input_tensor)
                loss = criterion(output, target_tensor)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error processing interaction: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}, No valid batches processed")
    
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved successfully")
if __name__ == "__main__":
    # Load interactions from log file
    try:
        with open("interactions_log.json", "r") as log_file:
            interactions = [json.loads(line) for line in log_file if line.strip()]
        if not interactions:
            raise FileNotFoundError("Empty interactions file")
    except FileNotFoundError:
        print("No interactions_log.json found. Creating dummy training data...")
        interactions = [
            {"user_input": "Hello", "response": "Hi there!"},
            {"user_input": "How are you?", "response": "I'm fine, thank you!"},
            {"user_input": "What's your name?", "response": "I'm NOVA, your AI assistant."},
            {"user_input": "Good morning", "response": "Good morning! How can I help you?"},
            {"user_input": "Thank you", "response": "You're welcome!"},
        ]
    
    print(f"Loaded {len(interactions)} interactions for training")
    
    # Initialize model with proper configuration
    model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(interactions, model, criterion, optimizer)
