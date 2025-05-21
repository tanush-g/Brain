import io
import numpy as np
from typing import Dict, Tuple
import keras
from PIL import Image

def load_trained_model(model_path: str):
    """
    Load and return a TensorFlow/Keras model from the given file path.
    """
    return keras.models.load_model(model_path)


def preprocess_image_from_bytes(image_bytes: bytes, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert raw image bytes into a normalized numpy array ready for model prediction.

    - Reads bytes, converts to grayscale, resizes to target_size.
    - Scales pixel values to [0, 1].
    - Returns shape (1, height, width, 1).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)  # add channel dim
    arr = np.expand_dims(arr, axis=0)   # add batch dim
    return arr


def predict(model, img_array: np.ndarray, class_mappings: Dict[str, int]) -> Tuple[str, np.ndarray]:
    """
    Run model prediction on preprocessed image array.

    Returns the predicted class name and the probability vector.
    """
    preds = model.predict(img_array)
    probs = preds[0]
    # invert mapping: index -> class name
    inv_map = {idx: name for name, idx in class_mappings.items()}
    pred_idx = int(np.argmax(probs))
    pred_class = inv_map.get(pred_idx, 'Unknown')
    return pred_class, probs
