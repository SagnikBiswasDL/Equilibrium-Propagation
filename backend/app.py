from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import torch
import threading
import time
from datetime import datetime

from training.equilibrium_propagation import train_equilibrium_propagation
from training.backpropagation import train_backpropagation
from utils.comparison import compare_training_results

app = Flask(__name__)
CORS(app)

# Global variables to track training status
training_status = {
    'ep_training': False,
    'backprop_training': False,
    'ep_progress': 0,
    'backprop_progress': 0,
    'ep_results': None,
    'backprop_results': None
}

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('../frontend/build', 'index.html')

@app.route('/api/status')
def get_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/train/ep', methods=['POST'])
def start_ep_training():
    """Start Equilibrium Propagation training"""
    if training_status['ep_training']:
        return jsonify({'error': 'EP training already in progress'}), 400
    
    try:
        data = request.get_json()
        beta = data.get('beta', 0.1)
        learning_rate = data.get('learning_rate', 0.001)
        num_epochs = data.get('num_epochs', 10)
        hidden_sizes = data.get('hidden_sizes', [500, 500])
        device = data.get('device', 'cpu')
        
        # Start training in background thread
        def train_ep():
            global training_status
            training_status['ep_training'] = True
            training_status['ep_progress'] = 0
            
            try:
                # Train with smaller dataset for demo
                history, model = train_equilibrium_propagation(
                    beta=beta,
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    hidden_sizes=hidden_sizes,
                    device=device
                )
                
                training_status['ep_results'] = history
                training_status['ep_progress'] = 100
                
            except Exception as e:
                print(f"EP training error: {e}")
                training_status['ep_results'] = {'error': str(e)}
            finally:
                training_status['ep_training'] = False
        
        thread = threading.Thread(target=train_ep)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'EP training started',
            'params': {
                'beta': beta,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'hidden_sizes': hidden_sizes,
                'device': device
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train/backprop', methods=['POST'])
def start_backprop_training():
    """Start backpropagation training"""
    if training_status['backprop_training']:
        return jsonify({'error': 'Backprop training already in progress'}), 400
    
    try:
        data = request.get_json()
        learning_rate = data.get('learning_rate', 0.001)
        num_epochs = data.get('num_epochs', 10)
        hidden_sizes = data.get('hidden_sizes', [500, 500])
        device = data.get('device', 'cpu')
        optimizer = data.get('optimizer', 'adam')
        
        # Start training in background thread
        def train_backprop():
            global training_status
            training_status['backprop_training'] = True
            training_status['backprop_progress'] = 0
            
            try:
                # Train with smaller dataset for demo
                history, model = train_backpropagation(
                    learning_rate=learning_rate,
                    num_epochs=num_epochs,
                    hidden_sizes=hidden_sizes,
                    device=device,
                    optimizer=optimizer
                )
                
                training_status['backprop_results'] = history
                training_status['backprop_progress'] = 100
                
            except Exception as e:
                print(f"Backprop training error: {e}")
                training_status['backprop_results'] = {'error': str(e)}
            finally:
                training_status['backprop_training'] = False
        
        thread = threading.Thread(target=train_backprop)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Backprop training started',
            'params': {
                'learning_rate': learning_rate,
                'num_epochs': num_epochs,
                'hidden_sizes': hidden_sizes,
                'device': device,
                'optimizer': optimizer
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_results():
    """Compare EP vs Backprop results"""
    try:
        # Check if both training results are available
        if not training_status['ep_results'] or not training_status['backprop_results']:
            return jsonify({'error': 'Both EP and Backprop results are required for comparison'}), 400
        
        # Save results to files for comparison
        os.makedirs('../results', exist_ok=True)
        
        ep_file = '../results/ep_training_history.json'
        backprop_file = '../results/backprop_training_history.json'
        
        with open(ep_file, 'w') as f:
            json.dump(training_status['ep_results'], f, indent=2)
        
        with open(backprop_file, 'w') as f:
            json.dump(training_status['backprop_results'], f, indent=2)
        
        # Generate comparison
        comparison_report = compare_training_results(ep_file, backprop_file)
        
        return jsonify({
            'message': 'Comparison completed',
            'comparison': comparison_report
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/ep')
def get_ep_results():
    """Get EP training results"""
    if training_status['ep_results']:
        return jsonify(training_status['ep_results'])
    else:
        return jsonify({'error': 'No EP results available'}), 404

@app.route('/api/results/backprop')
def get_backprop_results():
    """Get backprop training results"""
    if training_status['backprop_results']:
        return jsonify(training_status['backprop_results'])
    else:
        return jsonify({'error': 'No backprop results available'}), 404

@app.route('/api/results/comparison')
def get_comparison_results():
    """Get comparison results if available"""
    try:
        comparison_file = '../results/comparison_report.json'
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                comparison = json.load(f)
            return jsonify(comparison)
        else:
            return jsonify({'error': 'No comparison results available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'training_status': training_status
    })

@app.route('/api/stop/ep', methods=['POST'])
def stop_ep_training():
    """Stop EP training"""
    if training_status['ep_training']:
        training_status['ep_training'] = False
        return jsonify({'message': 'EP training stopped'})
    else:
        return jsonify({'error': 'No EP training in progress'}), 400

@app.route('/api/stop/backprop', methods=['POST'])
def stop_backprop_training():
    """Stop backprop training"""
    if training_status['backprop_training']:
        training_status['backprop_training'] = False
        return jsonify({'message': 'Backprop training stopped'})
    else:
        return jsonify({'error': 'No backprop training in progress'}), 400

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../results', exist_ok=True)
    
    print("Starting Equilibrium Propagation Backend Server...")
    print("API endpoints available at http://localhost:5000/api/")
    print("Health check: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
