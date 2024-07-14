# Description: This script is used to test the availability of GPU on the system.
import tensorflow as tf

# List all physical devices
physical_devices = tf.config.list_physical_devices()
print("Physical devices:", physical_devices)

# Check for GPU availability
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU is available and recognized by TensorFlow.")
    for gpu in gpu_devices:
        print("GPU Device:", gpu)
else:
    print("GPU is not available.")