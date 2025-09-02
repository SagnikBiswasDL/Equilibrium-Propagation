#!/usr/bin/env python3
"""
Startup script for the Equilibrium Propagation application.
This script will start both the backend and frontend services.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    # Check Python dependencies
    try:
        import torch
        import flask
        import numpy
        print("✅ Python dependencies OK")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False
    
    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js available: {result.stdout.strip()}")
        else:
            print("❌ Node.js not available")
            return False
    except FileNotFoundError:
        print("❌ Node.js not found. Please install Node.js from https://nodejs.org/")
        return False
    
    # Check if npm is available
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm available: {result.stdout.strip()}")
        else:
            print("❌ npm not available")
            return False
    except FileNotFoundError:
        print("❌ npm not found")
        return False
    
    return True

def install_frontend_dependencies():
    """Install frontend dependencies if needed"""
    frontend_dir = Path("frontend")
    node_modules = frontend_dir / "node_modules"
    
    if not node_modules.exists():
        print("Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install frontend dependencies: {e}")
            return False
    else:
        print("✅ Frontend dependencies already installed")
    
    return True

def start_backend():
    """Start the Flask backend server"""
    print("Starting Flask backend...")
    
    backend_dir = Path("backend")
    backend_script = backend_dir / "app.py"
    
    if not backend_script.exists():
        print(f"❌ Backend script not found: {backend_script}")
        return None
    
    try:
        # Start backend in background
        process = subprocess.Popen(
            [sys.executable, "app.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Backend started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Backend failed to start:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the React frontend development server"""
    print("Starting React frontend...")
    
    frontend_dir = Path("frontend")
    
    try:
        # Start frontend in background
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Frontend started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Frontend failed to start:")
            print(f"stdout: {stdout.decode()}")
            print(f"stderr: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("=" * 60)
    print("EQUILIBRIUM PROPAGATION APPLICATION STARTUP")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing dependencies.")
        return 1
    
    # Install frontend dependencies
    if not install_frontend_dependencies():
        print("\n❌ Frontend dependency installation failed.")
        return 1
    
    print("\nStarting services...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return 1
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Stopping backend...")
        backend_process.terminate()
        return 1
    
    print("\n" + "=" * 60)
    print("🎉 APPLICATION STARTED SUCCESSFULLY!")
    print("=" * 60)
    print("\nServices running:")
    print("• Backend API: http://localhost:5000")
    print("• Frontend UI: http://localhost:3000")
    print("• API Health: http://localhost:5000/api/health")
    
    print("\nOpening application in browser...")
    try:
        webbrowser.open('http://localhost:3000')
    except:
        print("Please manually open http://localhost:3000 in your browser")
    
    print("\nPress Ctrl+C to stop all services...")
    
    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        
        # Stop backend
        if backend_process and backend_process.poll() is None:
            print("Stopping backend...")
            backend_process.terminate()
            backend_process.wait()
        
        # Stop frontend
        if frontend_process and frontend_process.poll() is None:
            print("Stopping frontend...")
            frontend_process.terminate()
            frontend_process.wait()
        
        print("✅ All services stopped")
        return 0
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    finally:
        # Cleanup
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
