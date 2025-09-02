import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from ..models.neural_network import StandardNetwork
from ..utils.data_loader import get_mnist_data


class BackpropagationTrainer:
    """
    Trainer for standard backpropagation algorithm.
    """
    
    def __init__(self, model, device='cpu', learning_rate=0.001, optimizer='adam'):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Setup optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.gradient_norms = []
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch using backpropagation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        epoch_gradients = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            data, target = data.to(self.device), target.to(self.device)
            
            # Flatten input for MNIST
            data = data.view(data.size(0), -1)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm before optimization
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            epoch_gradients.append(total_norm)
            
            # Update weights
            self.optimizer.step()
            
            # Compute accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            total_loss += loss.item()
        
        # Record metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.gradient_norms.append(np.mean(epoch_gradients))
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                data = data.view(data.size(0), -1)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs=10):
        """Full training loop"""
        print(f"Starting Backpropagation training for {num_epochs} epochs")
        print(f"Learning rate: {self.learning_rate}, Optimizer: {type(self.optimizer).__name__}")
        
        best_val_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_backprop_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'gradient_norms': self.gradient_norms
        }
    
    def save_model(self, filename):
        """Save the trained model"""
        os.makedirs('../results', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'learning_rate': self.learning_rate,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'gradient_norms': self.gradient_norms
            }
        }, f'../results/{filename}')
    
    def save_training_history(self, filename):
        """Save training history to JSON"""
        os.makedirs('../results', exist_ok=True)
        history = {
            'timestamp': datetime.now().isoformat(),
            'learning_rate': self.learning_rate,
            'optimizer': type(self.optimizer).__name__,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'gradient_norms': self.gradient_norms
        }
        
        with open(f'../results/{filename}', 'w') as f:
            json.dump(history, f, indent=2)


def train_backpropagation(learning_rate=0.001, num_epochs=10, 
                         hidden_sizes=[500, 500], device='cpu', optimizer='adam'):
    """
    Train a model using standard backpropagation.
    
    Args:
        learning_rate: Learning rate for optimization
        num_epochs: Number of training epochs
        hidden_sizes: Hidden layer sizes
        device: Device to train on
        optimizer: Optimizer to use ('adam' or 'sgd')
        
    Returns:
        Training history and trained model
    """
    # Get data
    train_loader, val_loader = get_mnist_data()
    
    # Create model
    model = StandardNetwork(hidden_sizes=hidden_sizes)
    
    # Create trainer
    trainer = BackpropagationTrainer(
        model, device=device, learning_rate=learning_rate, optimizer=optimizer
    )
    
    # Train
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    # Save results
    trainer.save_training_history('backprop_training_history.json')
    
    return history, model
