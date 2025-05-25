# GPU Acceleration Setup for Brain Tumor Classification

This document provides information about setting up and using GPU acceleration with the Brain Tumor Classification project on Apple Silicon Macs.

## Overview

The Brain Tumor Classification project now supports GPU acceleration using Apple's Metal API via the tensorflow-metal plugin. This allows for faster model inference and training on Apple Silicon hardware.

## Requirements

- Apple Silicon Mac (M1, M2, or newer)
- macOS 11.0+
- Python 3.9 - 3.12
- TensorFlow 2.16.1 (specific version constraint)
- tensorflow-metal 1.2.0

## Installation

We've updated the `requirements.txt` file with specific version constraints to ensure compatibility:

```
keras>=3.6.0
tensorflow==2.16.1
tensorflow-metal==1.2.0
numpy<2.0.0,>=1.26.0
```

Install these dependencies using:

```bash
pip install -r requirements.txt
```

## Version Compatibility Notes

Important compatibility constraints:

1. **TensorFlow Version**: We're using TensorFlow 2.16.1 specifically as it's compatible with tensorflow-metal 1.2.0 and Python 3.12.
2. **Keras Version**: We're using standalone Keras 3.6.0+ (not tensorflow.keras).
3. **Python Version**: Python 3.12 is supported by these specific versions.

## Running with GPU Acceleration

The app has been updated to better handle GPU acceleration with proper error handling. Use the following scripts to run the application:

- `run_fixed_app.py`: Runs the app with improved GPU error handling
- `test_gpu_inference_fixed.py`: Tests GPU acceleration with the brain tumor model

## GPU Memory Management

The Metal backend has specific memory management requirements. The fixed app includes:

1. Memory growth settings to avoid OOM errors
2. Metal memory limits to prevent crashes
3. Fallback to CPU if GPU encounters errors

## Performance Notes

- GPU acceleration provides significant speedups for batch processing but may have higher overhead for single-image inference
- Larger matrix operations show more dramatic speedups on GPU vs CPU
- First-time GPU operations may be slower due to compilation overhead

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
