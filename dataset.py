"""
Dataset module for testing and training data handling.
"""
import torch
from torch.utils.data import Dataset
import json
import os


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


class TestDataset(Dataset):
    """
    Test dataset for model evaluation with proper text preprocessing.
    """
    def __init__(self, data_file=None, max_length=100):
        """
        Initialize the test dataset.
        
        Args:
            data_file (str): Path to the data file. If None, creates dummy data.
            max_length (int): Maximum sequence length for text preprocessing.
        """
        self.max_length = max_length
        
        if data_file and os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.interactions = [json.loads(line) for line in f if line.strip()]
        else:
            # Create dummy data for testing
            self.interactions = [
                {"user_input": "Hello", "response": "Hi there!"},
                {"user_input": "How are you?", "response": "I'm fine, thank you!"},
                {"user_input": "What's your name?", "response": "I'm NOVA, your AI assistant."},
                {"user_input": "Good morning", "response": "Good morning! How can I help you?"},
                {"user_input": "Thank you", "response": "You're welcome!"},
            ]
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        interaction = self.interactions[idx]
        user_input = interaction["user_input"]
        response = interaction["response"]
        
        # Use the same preprocessing as training
        input_tensor = preprocess_text(user_input, self.max_length)
        
        # For testing, we'll use the same simplified classification approach
        response_class = hash(response) % 10  # OUTPUT_SIZE
        target_tensor = torch.tensor(response_class, dtype=torch.long)
        
        return input_tensor, target_tensor