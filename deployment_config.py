"""
Deployment configuration for Streamlit Cloud and other environments.
This file helps manage environment-specific settings.
"""

import platform
import os

def is_macos():
    """Check if running on macOS (local development)"""
    return platform.system() == 'Darwin'

def is_streamlit_cloud():
    """Check if running on Streamlit Cloud"""
    return os.environ.get('STREAMLIT_SHARING_MODE') == 'streamlit' or \
           os.environ.get('IS_STREAMLIT_CLOUD') == 'true'

def configure_deployment():
    """Configure the environment based on deployment platform"""
    # Check if running on macOS (likely local development)
    if is_macos():
        try:
            from gpu_config import initialize_gpu
            # Initialize GPU with Metal backend for macOS
            initialize_gpu()
            return "macOS with Metal GPU acceleration"
        except (ImportError, Exception) as e:
            print(f"GPU configuration failed: {e}")
            return "macOS with CPU only"
    
    # For Linux/Streamlit Cloud - force CPU mode
    else:
        # Disable GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_DISABLE_MLIR_GRAPH_OPTIMIZATION'] = '1'
        return "Streamlit Cloud/Linux with CPU"

# Function to safely detect GPU availability across platforms
def get_gpu_info():
    """Get GPU information safely across platforms"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # GPU is available
            return True, f"GPU Available: {len(gpus)} device(s)"
        else:
            # No GPU available
            return False, "No GPU detected - using CPU"
    except Exception as e:
        # Error checking GPU
        return False, f"Error checking GPU: {e}"
