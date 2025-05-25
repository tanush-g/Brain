"""
GPU Configuration for TensorFlow on Apple Silicon

This module provides functions to safely configure TensorFlow to use
Metal GPU acceleration on Apple Silicon Macs, with appropriate error handling
and fallback mechanisms.
"""

import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def configure_gpu_memory():
    """
    Configure GPU memory management to prevent memory errors with Metal plugin.
    """
    try:
        # Set Metal plugin memory limits
        os.environ['TF_METAL_DEVICE_MEMORY_LIMIT'] = '2048'  # Limit to 2GB 
        os.environ['TF_METAL_DEVICE_MEMORY_FRACTION'] = '0.7'  # Use at most 70% of GPU memory
        
        # Limit TensorFlow memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                # Allow memory growth to avoid allocating all memory at once
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"Memory growth enabled for GPU: {gpu}")
                except RuntimeError as e:
                    logger.warning(f"Error setting memory growth for GPU: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error configuring GPU memory: {e}")
        return False

def verify_gpu_availability():
    """
    Check if GPU is available and properly configured.
    
    Returns:
        tuple: (is_gpu_available, gpu_info_str)
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = f"GPU available: {len(gpus)} device(s)"
            logger.info(gpu_info)
            logger.info(f"GPU devices: {gpus}")
            return True, gpu_info
        else:
            logger.warning("No GPU devices detected by TensorFlow")
            return False, "No GPU available"
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False, f"Error checking GPU: {e}"

def initialize_gpu():
    """
    Initialize GPU with all safety measures.
    
    Returns:
        bool: True if GPU is available and properly configured, False otherwise
    """
    try:
        # Set lower TensorFlow logging level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        # Additional Metal-specific configurations that help prevent SIGBUS errors
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Enable memory growth
        os.environ['TF_USE_CPU_ALLOCATOR_FOR_METAL'] = 'true'  # This helps with some M1/M2 specific issues
        
        # Configure memory management
        memory_configured = configure_gpu_memory()
        if not memory_configured:
            logger.warning("Failed to configure GPU memory, continuing with default settings")
        
        # Verify GPU is available
        is_available, _ = verify_gpu_availability()
        
        return is_available
    except Exception as e:
        logger.error(f"Error during GPU initialization: {e}")
        # In case of any exception, better to fall back to CPU
        force_cpu_mode()
        return False

def force_cpu_mode():
    """
    Force TensorFlow to use CPU only, disabling GPU.
    This is used as a fallback when GPU errors occur.
    """
    # Multiple environment variables to ensure GPU is fully disabled
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_DISABLE_MLIR_GRAPH_OPTIMIZATION'] = '1'
    
    # Clear any GPU-specific configuration
    if 'TF_METAL_DEVICE_MEMORY_LIMIT' in os.environ:
        del os.environ['TF_METAL_DEVICE_MEMORY_LIMIT']
    if 'TF_METAL_DEVICE_MEMORY_FRACTION' in os.environ:
        del os.environ['TF_METAL_DEVICE_MEMORY_FRACTION']
    
    # Close TensorFlow's connection to the GPU
    try:
        tf.config.set_visible_devices([], 'GPU')
        logger.info("Successfully disabled GPU devices")
    except Exception as e:
        logger.warning(f"Error while disabling GPU devices: {e}")
    
    logger.info("Forcing CPU-only mode for TensorFlow")
    
    # Clear and reset TensorFlow's device list to force reinitialization
    try:
        tf.keras.backend.clear_session()
        logger.info("Cleared TensorFlow session")
    except Exception as e:
        logger.warning(f"Error while clearing TensorFlow session: {e}")
    
    return True
