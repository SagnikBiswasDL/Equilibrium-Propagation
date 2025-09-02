import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EquilibriumNetwork(nn.Module):
    """
    Neural network designed for Equilibrium Propagation training.
    This network maintains internal states and can compute equilibrium states.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[500, 500], output_size=10, beta=0.1):
        super(EquilibriumNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.beta = beta  # Equilibrium propagation parameter
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # Internal states for equilibrium computation
        self.hidden_states = None
        self.output_state = None
        
    def forward(self, x):
        """Standard forward pass for backpropagation"""
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.output_layer(h)
    
    def compute_equilibrium(self, x, target=None, n_iterations=100, learning_rate=0.01):
        """
        Compute equilibrium states for Equilibrium Propagation.
        
        Args:
            x: Input data
            target: Target labels (for supervised learning)
            n_iterations: Number of iterations to reach equilibrium
            learning_rate: Learning rate for state updates
            
        Returns:
            Dictionary containing equilibrium states and energy
        """
        batch_size = x.size(0)
        
        # Initialize hidden states randomly
        if self.hidden_states is None:
            self.hidden_states = [torch.randn(batch_size, hidden_size, device=x.device) 
                                for hidden_size in self.hidden_sizes]
            self.output_state = torch.randn(batch_size, self.output_size, device=x.device)
        
        # Compute equilibrium through iterative updates
        energies = []
        
        for iteration in range(n_iterations):
            # Update hidden states
            new_hidden_states = []
            
            # First hidden layer
            h_input = x
            h_new = self.hidden_states[0] - learning_rate * self._compute_hidden_gradient(
                h_input, self.hidden_states[0], self.layers[0], 0
            )
            new_hidden_states.append(h_new)
            
            # Middle hidden layers
            for i in range(1, len(self.hidden_states)):
                h_input = new_hidden_states[i-1]
                h_new = self.hidden_states[i] - learning_rate * self._compute_hidden_gradient(
                    h_input, self.hidden_states[i], self.layers[i], i
                )
                new_hidden_states.append(h_new)
            
            # Update output state
            h_input = new_hidden_states[-1]
            output_new = self.output_state - learning_rate * self._compute_output_gradient(
                h_input, self.output_state, target
            )
            
            # Update states
            self.hidden_states = new_hidden_states
            self.output_state = output_new
            
            # Compute energy
            energy = self._compute_energy(x, self.hidden_states, self.output_state, target)
            energies.append(energy.item())
            
            # Check convergence
            if iteration > 10 and abs(energies[-1] - energies[-2]) < 1e-6:
                break
        
        return {
            'hidden_states': self.hidden_states,
            'output_state': self.output_state,
            'energies': energies,
            'final_energy': energies[-1] if energies else 0.0
        }
    
    def _compute_hidden_gradient(self, input_data, hidden_state, layer, layer_idx):
        """Compute gradient for hidden layer states"""
        # Energy gradient with respect to hidden state
        grad = layer.weight.T @ (layer(input_data) - hidden_state)
        
        # Add regularization term
        if layer_idx < len(self.hidden_sizes) - 1:
            next_layer = self.layers[layer_idx + 1]
            next_grad = next_layer.weight.T @ (next_layer(hidden_state) - self.hidden_states[layer_idx + 1])
            grad += next_grad
        
        return grad
    
    def _compute_output_gradient(self, input_data, output_state, target):
        """Compute gradient for output layer states"""
        if target is not None:
            # Supervised learning: minimize cross-entropy with target
            return output_state - target
        else:
            # Unsupervised: minimize output magnitude
            return output_state
    
    def _compute_energy(self, x, hidden_states, output_state, target=None):
        """Compute the total energy of the system"""
        energy = 0.0
        
        # Input reconstruction energy
        h = x
        for i, (layer, hidden_state) in enumerate(zip(self.layers, hidden_states)):
            energy += 0.5 * torch.sum((layer(h) - hidden_state) ** 2)
            h = hidden_state
        
        # Output energy
        energy += 0.5 * torch.sum((self.output_layer(h) - output_state) ** 2)
        
        # Target energy (if supervised)
        if target is not None:
            energy += 0.5 * torch.sum((output_state - target) ** 2)
        
        return energy
    
    def get_equilibrium_gradients(self, x, target, beta=0.1):
        """
        Compute gradients using Equilibrium Propagation.
        
        Args:
            x: Input data
            target: Target labels
            beta: EP parameter
            
        Returns:
            Dictionary containing gradients for each layer
        """
        # Compute free phase equilibrium (no target)
        free_eq = self.compute_equilibrium(x, target=None)
        
        # Compute nudged phase equilibrium (with target)
        nudged_eq = self.compute_equilibrium(x, target, learning_rate=0.01)
        
        # Compute gradients using EP formula
        gradients = {}
        
        # Hidden layer gradients
        for i, (layer, hidden_size) in enumerate(zip(self.layers, self.hidden_sizes)):
            free_state = free_eq['hidden_states'][i]
            nudged_state = nudged_eq['hidden_states'][i]
            
            # EP gradient: (nudged_state - free_state) / beta
            gradients[f'layer_{i}'] = (nudged_state - free_state) / beta
        
        # Output layer gradients
        free_output = free_eq['output_state']
        nudged_output = nudged_eq['output_state']
        gradients['output'] = (nudged_output - free_output) / beta
        
        return gradients


class StandardNetwork(nn.Module):
    """
    Standard neural network for backpropagation comparison.
    """
    
    def __init__(self, input_size=784, hidden_sizes=[500, 500], output_size=10):
        super(StandardNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Build layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.output_layer = nn.Linear(prev_size, output_size)
    
    def forward(self, x):
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
        return self.output_layer(h)
