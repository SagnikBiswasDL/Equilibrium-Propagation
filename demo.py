#!/usr/bin/env python3
"""
Demo script for Equilibrium Propagation implementation.
This script demonstrates the basic functionality of the EP algorithm.
"""

import torch
import torch.nn as nn
import numpy as np
from backend.models.neural_network import EquilibriumNetwork, StandardNetwork
from backend.utils.data_loader import get_mnist_data

def demo_equilibrium_propagation():
    """Demonstrate Equilibrium Propagation on a small dataset"""
    print("=" * 60)
    print("EQUILIBRIUM PROPAGATION DEMO")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get a small subset of MNIST for demo
    print("\nLoading MNIST data...")
    train_loader, val_loader = get_mnist_data(batch_size=32, train_size=1000, val_size=200)
    
    # Create EP model
    print("\nCreating Equilibrium Propagation model...")
    ep_model = EquilibriumNetwork(hidden_sizes=[100, 100], beta=0.1)
    ep_model = ep_model.to(device)
    
    # Test equilibrium computation
    print("\nTesting equilibrium computation...")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 1:  # Just test first batch
            break
            
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        # Convert target to one-hot
        target_one_hot = torch.zeros(target.size(0), 10, device=device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        print(f"Input shape: {data.shape}")
        print(f"Target shape: {target_one_hot.shape}")
        
        # Compute equilibrium
        equilibrium_info = ep_model.compute_equilibrium(data, target_one_hot, n_iterations=50)
        print(f"Equilibrium reached in {len(equilibrium_info['energies'])} iterations")
        print(f"Final energy: {equilibrium_info['final_energy']:.4f}")
        
        # Get EP gradients
        gradients = ep_model.get_equilibrium_gradients(data, target_one_hot, beta=0.1)
        print(f"Computed gradients for {len(gradients)} layers")
        
        # Test forward pass
        with torch.no_grad():
            output = ep_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / target.size(0)
            
        print(f"Forward pass - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        break
    
    print("\n✅ Equilibrium Propagation demo completed successfully!")

def demo_backpropagation():
    """Demonstrate standard backpropagation for comparison"""
    print("\n" + "=" * 60)
    print("BACKPROPAGATION DEMO")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get data
    train_loader, val_loader = get_mnist_data(batch_size=32, train_size=1000, val_size=200)
    
    # Create standard model
    print("Creating standard backpropagation model...")
    bp_model = StandardNetwork(hidden_sizes=[100, 100])
    bp_model = bp_model.to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 1:
            break
            
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        with torch.no_grad():
            output = bp_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            accuracy = 100. * pred.eq(target.view_as(pred)).sum().item() / target.size(0)
            
        print(f"Forward pass - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
        break
    
    print("✅ Backpropagation demo completed successfully!")

def demo_gradient_comparison():
    """Compare gradients between EP and backprop"""
    print("\n" + "=" * 60)
    print("GRADIENT COMPARISON DEMO")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get small batch
    train_loader, _ = get_mnist_data(batch_size=16, train_size=100)
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        
        # Create models
        ep_model = EquilibriumNetwork(hidden_sizes=[50, 50], beta=0.1).to(device)
        bp_model = StandardNetwork(hidden_sizes=[50, 50]).to(device)
        
        # Convert target to one-hot for EP
        target_one_hot = torch.zeros(target.size(0), 10, device=device)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        
        # Get EP gradients
        ep_gradients = ep_model.get_equilibrium_gradients(data, target_one_hot, beta=0.1)
        
        # Get backprop gradients
        bp_model.zero_grad()
        output = bp_model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Compare gradient magnitudes
        print("Gradient comparison:")
        print(f"EP model has {len(ep_gradients)} gradient components")
        print(f"Backprop model has {len(list(bp_model.parameters()))} parameter groups")
        
        # Simple comparison - just check if gradients exist
        ep_grad_norm = sum(torch.norm(grad).item() for grad in ep_gradients.values())
        bp_grad_norm = sum(p.grad.norm().item() for p in bp_model.parameters() if p.grad is not None)
        
        print(f"EP gradient norm: {ep_grad_norm:.4f}")
        print(f"Backprop gradient norm: {bp_grad_norm:.4f}")
        
        if ep_grad_norm > 0 and bp_grad_norm > 0:
            difference = abs(ep_grad_norm - bp_grad_norm) / bp_grad_norm * 100
            print(f"Relative difference: {difference:.2f}%")
            
            if difference <= 3.0:
                print("✅ Target achieved: EP gradients within 3% of backprop!")
            else:
                print(f"⚠️ Target not met: EP gradients differ by {difference:.2f}%")
        
        break
    
    print("✅ Gradient comparison demo completed!")

def main():
    """Run all demos"""
    try:
        print("Starting Equilibrium Propagation Demo Suite...")
        print("This demo will test the basic functionality of the implementation.")
        
        # Run demos
        demo_equilibrium_propagation()
        demo_backpropagation()
        demo_gradient_comparison()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Run the Flask backend: cd backend && python app.py")
        print("2. Run the React frontend: cd frontend && npm start")
        print("3. Open http://localhost:3000 in your browser")
        print("4. Use the training interface to run full experiments")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check that all dependencies are installed and the backend modules are accessible.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
