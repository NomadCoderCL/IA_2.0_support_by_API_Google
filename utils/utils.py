import torch

def calculate_accuracy(predictions, targets):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy
