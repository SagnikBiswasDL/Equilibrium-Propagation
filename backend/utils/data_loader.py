import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_mnist_data(batch_size=64, train_size=None, val_size=None, data_dir='../data'):
    """
    Load MNIST dataset for training and validation.
    
    Args:
        batch_size: Batch size for data loaders
        train_size: Number of training samples to use (None for all)
        val_size: Number of validation samples to use (None for all)
        data_dir: Directory to store/load MNIST data
        
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load training data
    train_dataset = datasets.MNIST(
        data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Load validation data
    val_dataset = datasets.MNIST(
        data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Limit dataset sizes if specified
    if train_size is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
    
    if val_size is not None:
        val_dataset = torch.utils.data.Subset(val_dataset, range(val_size))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"MNIST Data Loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


def get_sample_batch(batch_size=16, data_dir='../data'):
    """
    Get a single batch of MNIST data for visualization.
    
    Args:
        batch_size: Size of the batch
        data_dir: Directory containing MNIST data
        
    Returns:
        data, labels: A batch of images and their labels
    """
    train_loader, _ = get_mnist_data(batch_size=batch_size, data_dir=data_dir)
    
    for data, labels in train_loader:
        return data, labels
    
    return None, None
