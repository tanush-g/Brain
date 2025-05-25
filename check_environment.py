#!/usr/bin/env python3
"""
Environment Checker for Brain Tumor Classification App

This script helps diagnose environment issues in both development and deployment.
Run this script to get detailed information about your Python environment.
"""

import os
import sys
import platform
import subprocess
import importlib.util


def check_package_installed(package_name):
    """Check if a package is installed"""
    return importlib.util.find_spec(package_name) is not None


def get_package_version(package_name):
    """Get the version of an installed package"""
    if not check_package_installed(package_name):
        return "Not installed"
    
    try:
        package = importlib.import_module(package_name)
        return getattr(package, "__version__", "Unknown version")
    except ImportError:
        return "Error importing"


def check_gpu():
    """Check for GPU availability"""
    if check_package_installed("tensorflow"):
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            
            # Apple Metal specific detection
            is_macos = platform.system() == 'Darwin'
            has_metal = check_package_installed("tensorflow_metal")
            
            if gpus:
                return f"GPU Available: {gpus}"
            elif is_macos and has_metal:
                return "Apple Silicon GPU with Metal plugin available"
            else:
                return "No GPU detected - using CPU"
        except Exception as e:
            return f"Error checking GPU: {e}"
    else:
        return "TensorFlow not installed, cannot check GPU"


def main():
    """Run environment checks"""
    print("\n🧠 Brain Tumor Classification - Environment Check")
    print("=" * 70)
    
    # System information
    print("\n📊 System Information:")
    print(f"  • OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  • Python: {platform.python_version()} ({sys.executable})")
    
    # Key packages
    print("\n📦 Key Packages:")
    packages = [
        "tensorflow", "keras", "numpy", "PIL", "streamlit", 
        "sklearn", "matplotlib", "seaborn", "plotly"
    ]
    
    for package in packages:
        version = get_package_version(package)
        status = "✅" if version not in ["Not installed", "Error importing"] else "❌"
        print(f"  • {status} {package}: {version}")
    
    # Check TensorFlow Metal for macOS
    if platform.system() == 'Darwin' and platform.machine() in ['arm64', 'arm']:
        tf_metal_version = get_package_version("tensorflow_metal")
        status = "✅" if tf_metal_version not in ["Not installed", "Error importing"] else "❌"
        print(f"  • {status} tensorflow_metal: {tf_metal_version}")
    
    # GPU check
    print("\n🖥️ GPU Status:")
    print(f"  • {check_gpu()}")
    
    # Environment variables
    print("\n🔧 Environment Variables:")
    env_vars = [
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", 
        "TF_CPP_MIN_LOG_LEVEL", "TF_METAL_DEVICE_MEMORY_LIMIT"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        print(f"  • {var}: {value}")
    
    # Streamlit Cloud detection
    print("\n☁️ Deployment Platform:")
    is_streamlit_cloud = os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit' or \
                         os.environ.get('IS_STREAMLIT_CLOUD') == 'true'
    
    if is_streamlit_cloud:
        print("  • Running on Streamlit Cloud")
    else:
        print("  • Running locally")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
