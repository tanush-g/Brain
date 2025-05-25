#!/usr/bin/env python3
"""
Simple script to run the Brain Tumor Classification Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    # Map pip package names to import names
    required_packages = {
        'streamlit': 'streamlit',
        'tensorflow': 'tensorflow', 
        'keras': 'keras',
        'numpy': 'numpy',
        'pillow': 'PIL',  # Pillow imports as PIL
        'scikit-learn': 'sklearn',  # scikit-learn imports as sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n📦 Please install them with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if model file exists"""
    model_path = Path("model.keras")
    if not model_path.exists():
        print("❌ Model file 'model.keras' not found!")
        print("📝 You need to train the model first:")
        print("python train_model.py")
        return False
    
    print("✅ Model file found!")
    return True

def run_streamlit_app():
    """Run the Streamlit app"""
    print("🚀 Starting Brain Tumor Classification App...")
    print("📱 The app will open in your default web browser")
    print("🔄 Press Ctrl+C to stop the app")
    
    try:
        # First try to import and use our GPU configuration
        try:
            from gpu_config import initialize_gpu
            initialize_gpu()
            print("✅ GPU configuration initialized")
        except ImportError:
            print("⚠️ GPU configuration module not found, using default settings")
        
        # Run the Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True, capture_output=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running app: {e}")
        
        # Check if it's a SIGBUS error (common with Metal)
        if "SIGBUS" in e.stderr.decode() or "Bus error" in e.stderr.decode():
            print("\n❌ GPU error detected (SIGBUS). Retrying with CPU only...")
            
            # Force CPU mode
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            os.environ['TF_DISABLE_MLIR_GRAPH_OPTIMIZATION'] = '1'
            
            try:
                # Second attempt - CPU only
                print("🔄 Restarting with CPU-only mode...")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
            except Exception as e2:
                print(f"❌ Failed to run in CPU mode as well: {e2}")

def main():
    print("🧠 Brain Tumor Classification App Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found in current directory!")
        print("Please run this script from the Brain project directory")
        return
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        return
    
    print("✅ All required packages are installed!")
    
    # Check model file
    print("🔍 Checking model file...")
    if not check_model_file():
        return
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()
