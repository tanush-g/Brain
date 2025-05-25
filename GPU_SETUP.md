# GPU Acceleration Setup for Brain Tumor Classification

This document provides comprehensive information about setting up and using GPU acceleration with the Brain Tumor Classification project on Apple Silicon Macs.

## Overview

The Brain Tumor Classification application now includes safeguards and optimizations to prevent SIGBUS errors and other issues when using TensorFlow with Metal GPU acceleration on Apple Silicon Macs. This document explains the problem, solution, and setup instructions.

## SIGBUS Error on Apple Silicon: The Problem

SIGBUS (Bus error) errors are a common issue when running TensorFlow models on Apple Silicon Macs using the Metal backend. These errors typically occur due to:

1. Memory allocation/deallocation issues in the Metal plugin
2. Misaligned memory access
3. Aggressive memory usage by TensorFlow when running on Metal
4. Insufficient memory management for GPU operations

These errors are more likely to occur when:

- Processing larger images or batches
- Using complex models with many layers
- Running for extended periods of time
- Performing multiple inference passes

## Implemented Solutions

The application now includes the following safeguards:

1. **Memory Management**:
   - Limited Metal device memory to prevent OOM errors
   - Enabled memory growth to avoid allocating all GPU memory at once
   - Set memory fraction to limit peak memory usage

2. **Error Handling**:
   - Added SIGBUS detection and automatic fallback to CPU mode
   - Graceful handling of Metal plugin initialization failures
   - Proper cleanup of GPU resources

3. **Configuration Settings**:
   - Safe default settings in `gpu_config.py`
   - Added status indicators for GPU/CPU mode in the Streamlit UI
   - Fallback mechanisms for various error scenarios

## Setting Up Metal GPU Acceleration

### Requirements

- Apple Silicon Mac (M1, M2, or newer)
- macOS 12.0+ (Monterey or newer recommended)
- Python 3.9, 3.10, 3.11, or 3.12
- TensorFlow 2.15+ and tensorflow-metal

### Step 1: Install TensorFlow and Metal Plugin

Create and activate a virtual environment (recommended):

```bash
python -m venv brain-env
source brain-env/bin/activate
```

Install the required packages:

```bash
# For Python 3.9-3.11
pip install tensorflow==2.15.0 tensorflow-metal==1.1.0

# For Python 3.12
pip install tensorflow==2.16.1 tensorflow-metal==1.2.0
```

### Step 2: Install Project Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

Run the test script to verify your GPU is properly configured:

```bash
python test_gpu.py
```

You should see output similar to:

```text
Physical devices: [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
GPU is available and recognized by TensorFlow.
GPU Device: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

### Step 4: Test the Fixed App

Run the application using the safer launcher:

```bash
python run_app.py
```

## Manual Configuration (Advanced)

If you need to manually configure GPU settings, you can modify `gpu_config.py`:

```python
# Key settings you can adjust:
os.environ['TF_METAL_DEVICE_MEMORY_LIMIT'] = '2048'  # Limit to 2GB 
os.environ['TF_METAL_DEVICE_MEMORY_FRACTION'] = '0.7'  # Use at most 70% of GPU memory
```

### Testing GPU Performance

To test the actual performance gain from GPU:

```bash
python test_gpu.py
```

This will show you if the model loads and runs correctly with GPU acceleration.

## Advanced Configuration

### Environment Variables

These can be set before running Python or added to your shell profile:

```bash
# Force CPU-only mode if needed
export CUDA_VISIBLE_DEVICES="-1"

# Memory management settings
export TF_METAL_DEVICE_MEMORY_LIMIT="2048"
export TF_METAL_DEVICE_MEMORY_FRACTION="0.7"
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Debugging options
export TF_CPP_MIN_LOG_LEVEL="1"  # 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR
```

## Performance Notes

- Metal GPU acceleration works best for batch processing
- First inference may be slower due to compilation overhead
- Metal performance improves with larger matrices and operations
- For very small operations, CPU might actually be faster due to GPU transfer overhead

## Additional Resources

- [Apple's TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [TensorFlow Metal Documentation](https://developer.apple.com/documentation/tensorflow-metal)
- [TensorFlow Guide for Apple Silicon](https://developer.apple.com/metal/tensorflow-plugin/)

## Running with GPU Acceleration

The app has been updated to better handle GPU acceleration with proper error handling. Use the following scripts to run the application:

- `run_app.py`: Runs the app with improved GPU error handling
- `test_gpu.py`: Tests GPU acceleration with the brain tumor model

## GPU Memory Management

The Metal backend has specific memory management requirements. The fixed app includes:

1. Memory growth settings to avoid OOM errors
2. Metal memory limits to prevent crashes
3. Fallback to CPU if GPU encounters errors

## Troubleshooting

If you encounter GPU-related errors:

1. **SIGBUS errors**: These can occur with Metal. The fixed app should automatically fall back to CPU.
2. **Symbol not found errors**: These indicate version incompatibility. Make sure you're using exactly TensorFlow 2.16.1 with tensorflow-metal 1.2.0.
3. **Memory errors**: Try reducing the batch size or matrix dimensions if you encounter memory issues.

## Additional Information

For more details on TensorFlow Metal support, see:

- [Apple's TensorFlow Metal Plugin](https://developer.apple.com/metal/tensorflow-plugin/)
- [TensorFlow Metal Documentation](https://developer.apple.com/documentation/tensorflow-metal)

## Migration from TensorFlow.Keras

The project has been updated to use standalone Keras 3.x instead of the integrated tensorflow.keras. Key changes:

1. Imports changed from `tensorflow.keras` to `keras`
2. Replaced deprecated `ImageDataGenerator` with modern data augmentation layers
3. Updated model saving and loading code for Keras 3.x compatibility

Refer to the KERAS3_UPGRADE.md file for more details on these changes.
