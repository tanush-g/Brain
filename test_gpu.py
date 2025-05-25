#!/usr/bin/env python3
"""
Test script to verify that the GPU configuration fixes work correctly
"""

import os
import time
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# First try to safely initialize GPU
try:
    logger.info("Importing GPU configuration...")
    from gpu_config import initialize_gpu, verify_gpu_availability
    
    # Initialize GPU with safety measures
    logger.info("Initializing GPU...")
    gpu_available = initialize_gpu()
    is_available, gpu_info = verify_gpu_availability()
    
    logger.info(f"GPU Available: {is_available}")
    logger.info(f"GPU Info: {gpu_info}")
    
except Exception as e:
    logger.error(f"Error importing GPU configuration: {e}")
    gpu_available = False

# Import TensorFlow after GPU configuration
import tensorflow as tf
import numpy as np
from model_utils import ModelUtils

def test_model_loading():
    """Test loading the model with GPU configuration"""
    logger.info("Testing model loading...")
    
    try:
        # Create a test input
        logger.info("Creating test input...")
        test_input = np.zeros((1, 168, 168, 1), dtype=np.float32)
        
        # Load the model
        logger.info("Loading model...")
        model_utils = ModelUtils()
        model = model_utils.load_trained_model()
        
        # Perform a prediction
        logger.info("Running model prediction...")
        start_time = time.time()
        prediction = model.predict(test_input)
        end_time = time.time()
        
        logger.info(f"Prediction successful! Shape: {prediction.shape}")
        logger.info(f"Prediction time: {(end_time - start_time) * 1000:.2f} ms")
        logger.info(f"Prediction result: {prediction}")
        
        return True
    except Exception as e:
        logger.error(f"Error during model test: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("🧠 Testing GPU Fix for Brain Tumor Classifier")
    logger.info("=" * 50)
    
    # Log TensorFlow version
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Show physical devices
    devices = tf.config.list_physical_devices()
    logger.info(f"Physical devices: {devices}")
    
    # Check GPU specifically
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU devices: {gpus}")
    else:
        logger.info("No GPU devices detected")
    
    # Test model loading and prediction
    success = test_model_loading()
    
    if success:
        logger.info("\n✅ Test completed successfully!")
        logger.info("The GPU configuration appears to be working correctly.")
    else:
        logger.error("\n❌ Test failed!")
        logger.error("There may still be issues with the GPU configuration.")
