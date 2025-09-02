import numpy as np
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class EPvsBackpropComparison:
    """
    Utility class for comparing Equilibrium Propagation vs Backpropagation results.
    """
    
    def __init__(self, ep_history_file: str, backprop_history_file: str):
        """
        Initialize comparison with training history files.
        
        Args:
            ep_history_file: Path to EP training history JSON
            backprop_history_file: Path to backprop training history JSON
        """
        self.ep_history = self._load_history(ep_history_file)
        self.backprop_history = self._load_history(backprop_history_file)
        
    def _load_history(self, filepath: str) -> Dict:
        """Load training history from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Could not find {filepath}")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}")
            return {}
    
    def compare_gradient_signals(self) -> Dict[str, float]:
        """
        Compare gradient signals between EP and backprop.
        Returns percentage differences in gradient norms.
        """
        if not self.ep_history or not self.backprop_history:
            return {}
        
        ep_grads = np.array(self.ep_history.get('gradient_norms', []))
        bp_grads = np.array(self.backprop_history.get('gradient_norms', []))
        
        if len(ep_grads) == 0 or len(bp_grads) == 0:
            return {}
        
        # Ensure same length for comparison
        min_len = min(len(ep_grads), len(bp_grads))
        ep_grads = ep_grads[:min_len]
        bp_grads = bp_grads[:min_len]
        
        # Compute percentage differences
        differences = np.abs(ep_grads - bp_grads) / bp_grads * 100
        
        return {
            'mean_difference_percent': float(np.mean(differences)),
            'max_difference_percent': float(np.max(differences)),
            'std_difference_percent': float(np.std(differences)),
            'gradient_correlation': float(np.corrcoef(ep_grads, bp_grads)[0, 1])
        }
    
    def compare_convergence(self) -> Dict[str, float]:
        """
        Compare convergence rates between EP and backprop.
        """
        if not self.ep_history or not self.backprop_history:
            return {}
        
        ep_train_acc = np.array(self.ep_history.get('train_accuracies', []))
        bp_train_acc = np.array(self.backprop_history.get('train_accuracies', []))
        
        ep_val_acc = np.array(self.ep_history.get('val_accuracies', []))
        bp_val_acc = np.array(self.backprop_history.get('val_accuracies', []))
        
        if len(ep_train_acc) == 0 or len(bp_train_acc) == 0:
            return {}
        
        # Ensure same length for comparison
        min_len = min(len(ep_train_acc), len(bp_train_acc))
        ep_train_acc = ep_train_acc[:min_len]
        bp_train_acc = bp_train_acc[:min_len]
        ep_val_acc = ep_val_acc[:min_len]
        bp_val_acc = bp_val_acc[:min_len]
        
        # Compute convergence metrics
        ep_final_train = ep_train_acc[-1] if len(ep_train_acc) > 0 else 0
        bp_final_train = bp_train_acc[-1] if len(bp_train_acc) > 0 else 0
        ep_final_val = ep_val_acc[-1] if len(ep_val_acc) > 0 else 0
        bp_final_val = bp_val_acc[-1] if len(bp_val_acc) > 0 else 0
        
        return {
            'ep_final_train_acc': float(ep_final_train),
            'bp_final_train_acc': float(bp_final_train),
            'ep_final_val_acc': float(ep_final_val),
            'bp_final_val_acc': float(bp_final_val),
            'train_acc_difference': float(ep_final_train - bp_final_train),
            'val_acc_difference': float(ep_final_val - bp_final_val),
            'convergence_similarity': float(np.corrcoef(ep_train_acc, bp_train_acc)[0, 1])
        }
    
    def analyze_energy_dynamics(self) -> Dict[str, List[float]]:
        """
        Analyze energy dynamics from EP training.
        """
        if not self.ep_history:
            return {}
        
        energies = self.ep_history.get('energies', [])
        
        if not energies:
            return {}
        
        # Compute energy statistics
        energy_array = np.array(energies)
        
        return {
            'energies': energies,
            'energy_decay_rate': float(np.polyfit(range(len(energy_array)), energy_array, 1)[0]),
            'final_energy': float(energy_array[-1]) if len(energy_array) > 0 else 0.0,
            'energy_stability': float(np.std(energy_array))
        }
    
    def generate_comparison_report(self) -> Dict[str, Dict]:
        """
        Generate a comprehensive comparison report.
        """
        report = {
            'gradient_comparison': self.compare_gradient_signals(),
            'convergence_comparison': self.compare_convergence(),
            'energy_analysis': self.analyze_energy_dynamics(),
            'summary': {}
        }
        
        # Generate summary
        grad_comp = report['gradient_comparison']
        conv_comp = report['convergence_comparison']
        
        if grad_comp:
            mean_diff = grad_comp.get('mean_difference_percent', 0)
            report['summary']['gradient_accuracy'] = f"EP gradients within {mean_diff:.2f}% of backprop"
            
            if mean_diff <= 3.0:
                report['summary']['gradient_status'] = "✅ Target achieved: EP gradients within 3% of backprop"
            else:
                report['summary']['gradient_status'] = f"⚠️ Target not met: EP gradients differ by {mean_diff:.2f}%"
        
        if conv_comp:
            ep_val = conv_comp.get('ep_final_val_acc', 0)
            bp_val = conv_comp.get('bp_final_val_acc', 0)
            report['summary']['convergence'] = f"EP: {ep_val:.2f}%, Backprop: {bp_val:.2f}%"
        
        return report
    
    def save_comparison_report(self, output_file: str = '../results/comparison_report.json'):
        """Save comparison report to JSON file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        report = self.generate_comparison_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comparison report saved to {output_file}")
        return report
    
    def plot_comparison(self, save_dir: str = '../results'):
        """
        Generate comparison plots and save them.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Training accuracy comparison
        if self.ep_history and self.backprop_history:
            ep_acc = self.ep_history.get('train_accuracies', [])
            bp_acc = self.backprop_history.get('train_accuracies', [])
            
            if ep_acc and bp_acc:
                plt.figure(figsize=(10, 6))
                min_len = min(len(ep_acc), len(bp_acc))
                epochs = range(1, min_len + 1)
                
                plt.plot(epochs, ep_acc[:min_len], 'o-', label='Equilibrium Propagation', linewidth=2)
                plt.plot(epochs, bp_acc[:min_len], 's-', label='Backpropagation', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Training Accuracy (%)')
                plt.title('Training Accuracy Comparison: EP vs Backpropagation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/training_accuracy_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 2. Gradient norm comparison
        if self.ep_history and self.backprop_history:
            ep_grads = self.ep_history.get('gradient_norms', [])
            bp_grads = self.backprop_history.get('gradient_norms', [])
            
            if ep_grads and bp_grads:
                plt.figure(figsize=(10, 6))
                min_len = min(len(ep_grads), len(bp_grads))
                epochs = range(1, min_len + 1)
                
                plt.plot(epochs, ep_grads[:min_len], 'o-', label='Equilibrium Propagation', linewidth=2)
                plt.plot(epochs, bp_grads[:min_len], 's-', label='Backpropagation', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('Gradient Norm')
                plt.title('Gradient Norm Comparison: EP vs Backpropagation')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/gradient_norm_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 3. Energy dynamics (EP only)
        if self.ep_history:
            energies = self.ep_history.get('energies', [])
            
            if energies:
                plt.figure(figsize=(10, 6))
                epochs = range(1, len(energies) + 1)
                
                plt.plot(epochs, energies, 'o-', color='red', linewidth=2)
                plt.xlabel('Epoch')
                plt.ylabel('System Energy')
                plt.title('Equilibrium Propagation: Energy Dynamics')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/ep_energy_dynamics.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Comparison plots saved to {save_dir}")


def compare_training_results(ep_file: str, backprop_file: str, output_dir: str = '../results'):
    """
    Convenience function to compare training results and generate reports.
    
    Args:
        ep_file: Path to EP training history
        backprop_file: Path to backprop training history
        output_dir: Directory to save comparison results
    """
    comparison = EPvsBackpropComparison(ep_file, backprop_file)
    
    # Generate and save report
    report = comparison.save_comparison_report(f'{output_dir}/comparison_report.json')
    
    # Generate plots
    comparison.plot_comparison(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EQUILIBRIUM PROPAGATION vs BACKPROPAGATION COMPARISON")
    print("="*60)
    
    summary = report.get('summary', {})
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*60)
    
    return report
