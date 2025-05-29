"""
Configuration file for Brain Tumor Classification Project
Contains all hyperparameters, paths, and settings used across training and inference
"""

from pathlib import Path
from typing import Dict, Tuple
import logging

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "brain-tumor-mri-dataset"
RESOURCES_DIR = BASE_DIR / "resources"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

def initialize_directories():
    """Ensure required directories exist."""
    RESOURCES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

# Data paths
TRAIN_DATA_PATH = DATA_DIR / "Training"
TEST_DATA_PATH = DATA_DIR / "Testing"
# Model paths
MODEL_PATH = BASE_DIR / "model.keras"
BEST_MODEL_PATH = MODELS_DIR / "best_model.keras"
CHECKPOINT_PATH = MODELS_DIR / "checkpoint.keras"

# Image processing parameters
IMAGE_SIZE = (168, 168)
IMAGE_CHANNELS = 1  # Grayscale
IMAGE_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)

# Class mappings - Fixed to match dataset structure
CLASS_MAPPINGS = {
    'glioma': 0, 
    'meningioma': 1, 
    'notumor': 2, 
    'pituitary': 3
}

# Display names for UI (capitalized versions)
CLASS_DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma', 
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary'
}

INVERSE_CLASS_MAPPINGS = {v: k for k, v in CLASS_MAPPINGS.items()}
CLASS_NAMES = list(CLASS_MAPPINGS.keys())
NUM_CLASSES = len(CLASS_MAPPINGS)

# Class descriptions for UI
CLASS_DESCRIPTIONS = {
    'Glioma': 'Cancerous brain tumors that develop in glial cells. These tumors grow from the supportive tissue of the brain and are often aggressive.',
    'Meningioma': 'Usually non-cancerous tumors originating from the meninges (protective membranes covering the brain and spinal cord).',
    'No Tumor': 'Normal brain scan without detectable tumors or abnormal growth patterns.',
    'Pituitary': 'Tumors affecting the pituitary gland (can be cancerous or non-cancerous). The pituitary gland controls hormone production.'
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'beta_1': 0.85,
    'beta_2': 0.9925,
    'validation_split': 0.2,
    'random_seed': 111,
    'verbose': 1
}

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation_range': 0.02,
    'contrast_range': 0.1,
    'zoom_height_factor': 0.01,
    'zoom_width_factor': 0.05,
    'translation_height_factor': 0.0015,
    'translation_width_factor': 0.0015,
    'fill_mode': 'constant',
    'contrast_enhancement_factor': 1.2
}

# Model architecture parameters
MODEL_CONFIG = {
    'conv_layers': [
        {'filters': 64, 'kernel_size': (5, 5), 'pool_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (5, 5), 'pool_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (4, 4), 'pool_size': (2, 2), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (4, 4), 'pool_size': (2, 2), 'activation': 'relu'}
    ],
    'dense_units': 512,
    'dropout_rate': 0.3,
    'final_activation': 'softmax',
    'loss_function': 'categorical_crossentropy',
    'metrics': ['accuracy']
}

# Callback parameters
CALLBACK_CONFIG = {
    'reduce_lr_factor': 0.8,
    'reduce_lr_patience': 4,
    'reduce_lr_min_lr': 1e-4,
    'reduce_lr_monitor': 'val_loss',
    'early_stopping_patience': 10,
    'early_stopping_monitor': 'val_accuracy',
    'early_stopping_restore_best_weights': True,
    'model_checkpoint_monitor': 'val_accuracy',
    'model_checkpoint_mode': 'max',
    'model_checkpoint_save_best_only': True,
    'custom_lr_thresholds': [0.96, 0.99, 0.9935],
    'custom_lr_factor': 0.75
}

# Streamlit app configuration
STREAMLIT_CONFIG = {
    'page_title': "Brain Tumor MRI Classifier",
    'page_icon': "🧠",
    'layout': "wide",
    'initial_sidebar_state': "expanded",
    'max_file_size': 10,  # MB
    'allowed_extensions': ["jpg", "jpeg", "png", "bmp", "tiff", "dcm", "tif"],
}

# Visualization parameters
VISUALIZATION_CONFIG = {
    'figsize_default': (12, 8),
    'figsize_large': (15, 10),
    'figsize_small': (8, 6),
    'color_palette': ['#FAC500', '#0BFA00', '#0066FA', '#FA0000'],
    'dpi': 100,
    'style': 'default',
    'confusion_matrix_cmap': 'Blues',
    'plot_style': 'seaborn-v0_8'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': LOGS_DIR / 'brain_tumor_classifier.log',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'confidence_threshold': 0.7,
    'high_confidence_threshold': 0.9,
    'acceptable_accuracy': 0.85,
    'excellent_accuracy': 0.95
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'normalize': True,
    'normalization_range': (0, 1),
    'resize_method': 'lanczos',
    'grayscale_conversion': True,
    'contrast_enhancement': False,
    'histogram_equalization': False
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision', 
    'recall',
    'f1_score',
    'confusion_matrix',
    'classification_report'
]

def get_data_paths() -> Tuple[str, str]:
    """Get training and testing data paths"""
    return str(TRAIN_DATA_PATH), str(TEST_DATA_PATH)

def get_model_save_path() -> str:
    """Get the path where the model should be saved"""
    return str(MODEL_PATH)

def get_class_info() -> Dict:
    """Get comprehensive class information"""
    return {
        'mappings': CLASS_MAPPINGS,
        'inverse_mappings': INVERSE_CLASS_MAPPINGS,
        'names': CLASS_NAMES,
        'display_names': CLASS_DISPLAY_NAMES,
        'descriptions': CLASS_DESCRIPTIONS,
        'num_classes': NUM_CLASSES
    }

def get_training_config() -> Dict:
    """Get complete training configuration"""
    return {
        **TRAINING_CONFIG,
        'model_config': MODEL_CONFIG,
        'augmentation_config': AUGMENTATION_CONFIG,
        'callback_config': CALLBACK_CONFIG
    }

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=LOGGING_CONFIG['level'],
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['log_file']),
            logging.StreamHandler()
        ]
    )

def validate_config():
    """Validate configuration parameters"""
    # Check if data directories exist
    if not TRAIN_DATA_PATH.exists():
        print(f"Warning: Training data directory not found: {TRAIN_DATA_PATH}")
    
    if not TEST_DATA_PATH.exists():
        print(f"Warning: Testing data directory not found: {TEST_DATA_PATH}")
    
    # Validate image parameters
    if len(IMAGE_SIZE) != 2:
        raise ValueError("IMAGE_SIZE must be a tuple of (height, width)")
    
    if IMAGE_CHANNELS not in [1, 3]:
        raise ValueError("IMAGE_CHANNELS must be 1 (grayscale) or 3 (RGB)")
    
    # Validate class mappings
    if len(set(CLASS_MAPPINGS.values())) != len(CLASS_MAPPINGS):
        raise ValueError("Duplicate values in CLASS_MAPPINGS")
    
    if set(CLASS_MAPPINGS.values()) != set(range(NUM_CLASSES)):
        raise ValueError("CLASS_MAPPINGS values must be consecutive integers starting from 0")
    
    return True

if __name__ == "__main__":
    # Validate configuration when run directly
    try:
        validate_config()
        print("✅ Configuration validation passed!")
        print(f"📁 Data directory: {DATA_DIR}")
        print(f"🎯 Number of classes: {NUM_CLASSES}")
        print(f"🖼️ Image shape: {IMAGE_SHAPE}")
        print(f"🔧 Training epochs: {TRAINING_CONFIG['epochs']}")
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
