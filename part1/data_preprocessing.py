import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


SAMPLE_SIZE = 1000

def load_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    train_indices = torch.randperm(len(full_train_dataset))[:SAMPLE_SIZE]
    test_indices = torch.randperm(len(full_test_dataset))[:SAMPLE_SIZE]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def get_numpy_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    full_train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    
    train_indices = torch.randperm(len(full_train_dataset))[:SAMPLE_SIZE]
    test_indices = torch.randperm(len(full_test_dataset))[:SAMPLE_SIZE]
    
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)
    
    X_train = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).view(-1, 28*28).numpy()
    y_train = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))]).numpy()
    X_test = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).view(-1, 28*28).numpy()
    y_test = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).numpy()
    
    return X_train, y_train, X_test, y_test
