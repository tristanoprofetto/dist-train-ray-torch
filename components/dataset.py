import torch
import torch.nn as nn
from torchvision import datasets, transforms


def get_data_loaders(batch_size: int):
    """
    Returns train, test dataloaders for a given batch size

    Args:
        batch_size (int):

    Returns:
        train_loader (torch.utils.utils.data.DataLoader): training dataloader
        test_loader (torch.utils.data.DataLoader): testing dataloader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.38081))
    ])
    train = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)
    
    return train_loader, test_loader