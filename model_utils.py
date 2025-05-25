import io
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageEnhance

from config import (
    IMAGE_SIZE, IMAGE_SHAPE, INVERSE_CLASS_MAPPINGS,
    PERFORMANCE_THRESHOLDS, MODEL_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility class for model operations including loading, preprocessing, and prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ModelUtils with optional model path."""
        self.model_path = model_path or str(MODEL_PATH)
        self.model = None
        self.is_model_loaded = False
    
    def load_trained_model(self, model_path: Optional[str] = None):
        """
        Load and return a TensorFlow/Keras model from the given file path.
        
        Args:
            model_path: Path to the model file. If None, uses the default path.
            
        Returns:
            Loaded Keras model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        path_to_use = model_path or self.model_path
        
        if not Path(path_to_use).exists():
            raise FileNotFoundError(f"Model file not found: {path_to_use}")
        
        try:
            logger.info(f"Loading model from: {path_to_use}")
            self.model = keras.models.load_model(path_to_use)
            self.is_model_loaded = True
            logger.info("Model loaded successfully")
            return self.model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise Exception(f"Error loading model: {e}")
    
    def preprocess_image_from_bytes(
        self, 
        image_bytes: bytes, 
        target_size: Optional[Tuple[int, int]] = None,
        enhance_contrast: bool = False
    ) -> np.ndarray:
        """
        Convert raw image bytes into a normalized numpy array ready for model prediction.

        Args:
            image_bytes: Raw image bytes
            target_size: Target size for resizing. If None, uses config IMAGE_SIZE
            enhance_contrast: Whether to enhance image contrast
            
        Returns:
            Preprocessed image array with shape (1, height, width, 1)
            
        Raises:
            ValueError: If image cannot be processed
        """
        try:
            target_size = target_size or IMAGE_SIZE
            
            # Open and convert to grayscale
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB first if image has transparency (RGBA) or is in palette mode
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Convert to grayscale
            img = img.convert('L')
            
            # Optional contrast enhancement
            if enhance_contrast:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Resize to target size
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            arr = np.array(img, dtype=np.float32) / 255.0
            
            # Add channel dimension for grayscale
            arr = np.expand_dims(arr, axis=-1)
            
            # Add batch dimension
            arr = np.expand_dims(arr, axis=0)
            
            logger.debug(f"Preprocessed image shape: {arr.shape}")
            return arr
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}")
    
    def preprocess_image_from_path(
        self, 
        image_path: Union[str, Path], 
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Load and preprocess an image from file path.
        
        Args:
            image_path: Path to the image file
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return self.preprocess_image_from_bytes(image_bytes, target_size)
        except Exception as e:
            logger.error(f"Error loading image from {image_path}: {e}")
            raise ValueError(f"Failed to load image from {image_path}: {e}")
    
    def predict(
        self, 
        img_array: np.ndarray, 
        model = None,
        return_probabilities: bool = True
    ) -> Tuple[str, Optional[np.ndarray], float]:
        """
        Run model prediction on preprocessed image array.

        Args:
            img_array: Preprocessed image array
            model: Keras model. If None, uses the loaded model
            return_probabilities: Whether to return full probability array
            
        Returns:
            Tuple of (predicted_class_name, probabilities, confidence_score)
            
        Raises:
            ValueError: If no model is available or prediction fails
        """
        model_to_use = model or self.model
        
        if model_to_use is None:
            raise ValueError("No model available. Please load a model first.")
        
        try:
            # Ensure input shape is correct
            if img_array.shape[1:] != IMAGE_SHAPE:
                logger.warning(f"Input shape {img_array.shape} doesn't match expected {IMAGE_SHAPE}")
            
            # Make prediction
            preds = model_to_use.predict(img_array, verbose=0)
            probs = preds[0]
            
            # Get predicted class
            pred_idx = int(np.argmax(probs))
            pred_class = INVERSE_CLASS_MAPPINGS.get(pred_idx, 'Unknown')
            
            # Calculate confidence score
            confidence = float(np.max(probs))
            
            logger.info(f"Prediction: {pred_class} (confidence: {confidence:.3f})")
            
            return pred_class, probs if return_probabilities else None, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def predict_with_confidence_analysis(
        self, 
        img_array: np.ndarray, 
        model = None
    ) -> Dict:
        """
        Perform prediction with detailed confidence analysis.
        
        Args:
            img_array: Preprocessed image array
            model: Keras model to use
            
        Returns:
            Dictionary with prediction results and confidence analysis
        """
        pred_class, probs, confidence = self.predict(img_array, model, return_probabilities=True)
        
        # Ensure probs is not None for confidence analysis
        if probs is None:
            raise ValueError("Probabilities are required for confidence analysis")
        
        # Confidence analysis
        high_confidence = confidence >= PERFORMANCE_THRESHOLDS['high_confidence_threshold']
        acceptable_confidence = confidence >= PERFORMANCE_THRESHOLDS['confidence_threshold']
        
        # Get top 2 predictions for uncertainty analysis
        top_2_indices = np.argsort(probs)[-2:][::-1]
        top_2_probs = probs[top_2_indices]
        uncertainty = float(top_2_probs[0] - top_2_probs[1])  # Difference between top 2
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'high_confidence': high_confidence,
            'acceptable_confidence': acceptable_confidence,
            'uncertainty': uncertainty,
            'all_probabilities': {
                INVERSE_CLASS_MAPPINGS[i]: float(prob) 
                for i, prob in enumerate(probs)
            },
            'top_2_classes': [
                INVERSE_CLASS_MAPPINGS[idx] for idx in top_2_indices
            ]
        }
    
    def batch_predict(
        self, 
        image_arrays: list, 
        model = None
    ) -> list:
        """
        Perform batch prediction on multiple images.
        
        Args:
            image_arrays: List of preprocessed image arrays
            model: Keras model to use
            
        Returns:
            List of prediction results
        """
        model_to_use = model or self.model
        
        if model_to_use is None:
            raise ValueError("No model available. Please load a model first.")
        
        results = []
        for img_array in image_arrays:
            try:
                result = self.predict_with_confidence_analysis(img_array, model_to_use)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction failed for one image: {e}")
                results.append({
                    'error': str(e),
                    'predicted_class': 'Error',
                    'confidence': 0.0
                })
        
        return results

# Convenience functions for backward compatibility
def load_trained_model(model_path: str):
    """Load a trained model (backward compatibility function)."""
    utils = ModelUtils()
    return utils.load_trained_model(model_path)

def preprocess_image_from_bytes(image_bytes: bytes, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image from bytes (backward compatibility function)."""
    utils = ModelUtils()
    return utils.preprocess_image_from_bytes(image_bytes, target_size)

def predict(model, img_array: np.ndarray, class_mappings: Dict[str, int]) -> Tuple[str, np.ndarray]:
    """Make prediction (backward compatibility function)."""
    utils = ModelUtils()
    pred_class, probs, _ = utils.predict(img_array, model)
    return pred_class, probs
