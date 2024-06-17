import torch
from torch.utils.data import DataLoader
from dataset import MyDataset

def get_dataloader(data, batch_size):
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
