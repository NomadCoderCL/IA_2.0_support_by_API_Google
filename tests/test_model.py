import torch
from models.model import MyModel
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE
from utils import calculate_accuracy
from dataset import TestDataset
from torch.utils.data import DataLoader

def test_model(model, test_loader):
    model.eval()
    total_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            accuracy = calculate_accuracy(output, target)
            total_accuracy += accuracy
    average_accuracy = total_accuracy / len(test_loader)
    print(f'Test Accuracy: {average_accuracy * 100:.2f}%')

if __name__ == "__main__":
    model = MyModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load('model.pth'))
    test_dataset = TestDataset()  
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_model(model, test_loader)
