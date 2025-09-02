import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from ..models.neural_network import EquilibriumNetwork
from ..utils.data_loader import get_mnist_data


class EquilibriumPropagationTrainer:
    """
    Trainer for Equilibrium Propagation algorithm.
    """
    
    def __init__(self, model, device='cpu', beta=0.1, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.beta = beta
        self.learning_rate = learning_rate
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.energies = []
        self.gradient_norms = []
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch using Equilibrium Propagation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        epoch_energies = []
        epoch_gradients = []
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            data, target = data.to(self.device), target.to(self.device)
            
            # Flatten input for MNIST
            data = data.view(data.size(0), -1)
            
            # Convert target to one-hot encoding
            target_one_hot = torch.zeros(target.size(0), 10, device=self.device)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
            
            # Compute equilibrium states and gradients
            equilibrium_info = self.model.compute_equilibrium(data, target_one_hot)
            gradients = self.model.get_equilibrium_gradients(data, target_one_hot, self.beta)
            
            # Update weights using computed gradients
            self._update_weights(gradients)
            
            # Compute loss and accuracy
            with torch.no_grad():
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                total_loss += loss.item()
                epoch_energies.append(equilibrium_info['final_energy'])
                
                # Compute gradient norm
                grad_norm = sum(torch.norm(grad).item() for grad in gradients.values())
                epoch_gradients.append(grad_norm)
        
        # Record metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        self.energies.append(np.mean(epoch_energies))
        self.gradient_norms.append(np.mean(epoch_gradients))
        
        return avg_loss, accuracy
    
    def _update_weights(self, gradients):
        """Update network weights using EP gradients"""
        with torch.no_grad():
            # Update hidden layer weights
            for i, layer in enumerate(self.model.layers):
                if f'layer_{i}' in gradients:
                    # Simple weight update rule
                    # In practice, you might want to use a more sophisticated optimizer
                    layer.weight -= self.learning_rate * gradients[f'layer_{i}'].mean(dim=0, keepdim=True).T
            
            # Update output layer weights
            if 'output' in gradients:
                self.model.output_layer.weight -= self.learning_rate * gradients['output'].mean(dim=0, keepdim=True).T
    
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
                loss = nn.CrossEntropyLoss()(output, target)
                
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
        print(f"Starting Equilibrium Propagation training for {num_epochs} epochs")
        print(f"Beta: {self.beta}, Learning rate: {self.learning_rate}")
        
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
                self.save_model('best_ep_model.pth')
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'energies': self.energies,
            'gradient_norms': self.gradient_norms
        }
    
    def save_model(self, filename):
        """Save the trained model"""
        os.makedirs('../results', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'beta': self.beta,
            'learning_rate': self.learning_rate,
            'training_history': {
                'train_losses': self.train_losses,
                'train_accuracies': self.train_accuracies,
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'energies': self.energies,
                'gradient_norms': self.gradient_norms
            }
        }, f'../results/{filename}')
    
    def save_training_history(self, filename):
        """Save training history to JSON"""
        os.makedirs('../results', exist_ok=True)
        history = {
            'timestamp': datetime.now().isoformat(),
            'beta': self.beta,
            'learning_rate': self.learning_rate,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'energies': self.energies,
            'gradient_norms': self.gradient_norms
        }
        
        with open(f'../results/{filename}', 'w') as f:
            json.dump(history, f, indent=2)


def train_equilibrium_propagation(beta=0.1, learning_rate=0.001, num_epochs=10, 
                                hidden_sizes=[500, 500], device='cpu'):
    """
    Train a model using Equilibrium Propagation.
    
    Args:
        beta: EP parameter
        learning_rate: Learning rate for weight updates
        num_epochs: Number of training epochs
        hidden_sizes: Hidden layer sizes
        device: Device to train on
        
    Returns:
        Training history and trained model
    """
    # Get data
    train_loader, val_loader = get_mnist_data()
    
    # Create model
    model = EquilibriumNetwork(hidden_sizes=hidden_sizes, beta=beta)
    
    # Create trainer
    trainer = EquilibriumPropagationTrainer(
        model, device=device, beta=beta, learning_rate=learning_rate
    )
    
    # Train
    history = trainer.train(train_loader, val_loader, num_epochs)
    
    # Save results
    trainer.save_training_history(f'ep_training_history_beta_{beta}.json')
    
    return history, model
