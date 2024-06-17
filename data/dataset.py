import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data, label = self.data[idx]
        return torch.tensor(input_data), torch.tensor(label)
