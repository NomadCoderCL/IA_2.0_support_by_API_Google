import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.model import MyModel
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from utils import calculate_accuracy
from dataset import TestDataset
from torch.utils.data import DataLoader

def test_model(model, test_loader):
    """
    Test the model on the test dataset.
    
    Args:
        model (nn.Module): The model to test
        test_loader (DataLoader): Test data loader
    """
    model.eval()
    total_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            try:
                output = model(data)
                accuracy = calculate_accuracy(output, target)
                total_accuracy += accuracy * data.size(0)
                total_samples += data.size(0)
            except Exception as e:
                print(f"Error during testing: {e}")
                continue
    
    if total_samples > 0:
        average_accuracy = total_accuracy / total_samples
        print(f'Test Accuracy: {average_accuracy * 100:.2f}%')
    else:
        print("No valid samples processed")

if __name__ == "__main__":
    try:
        # Initialize model
        model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        
        # Load trained model if available
        if os.path.exists('model.pth'):
            model.load_state_dict(torch.load('model.pth'))
            print("Loaded trained model")
        else:
            print("No trained model found, using random weights")
        
        # Create test dataset and loader
        test_dataset = TestDataset()
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        
        # Test the model
        test_model(model, test_loader)
        
    except Exception as e:
        print(f"Error running test: {e}")
        sys.exit(1)
