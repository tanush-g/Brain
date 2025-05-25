#!/usr/bin/env python3
"""
Brain Tumor Classification Model Training Script

This script provides a clean, production-ready training pipeline for the brain tumor
classification CNN model. It replaces the Jupyter notebook with structured, reusable code.

Usage:
    python train_model.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--evaluate-only]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle

# Keras imports
import keras
from keras.models import Sequential, load_model
from keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Input, 
                          RandomRotation, RandomContrast, RandomZoom, 
                          RandomFlip, RandomTranslation, Dropout)
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

# Local imports
from config import (
    TRAINING_CONFIG, MODEL_CONFIG, AUGMENTATION_CONFIG, CALLBACK_CONFIG,
    IMAGE_SIZE, IMAGE_SHAPE, CLASS_MAPPINGS, INVERSE_CLASS_MAPPINGS,
    get_data_paths, get_model_save_path, get_class_info, setup_logging,
    VISUALIZATION_CONFIG, PERFORMANCE_THRESHOLDS, RESOURCES_DIR,
    CHECKPOINT_PATH, BEST_MODEL_PATH
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class BrainTumorTrainer:
    """Main training class for brain tumor classification model"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize trainer with configuration"""
        self.config = TRAINING_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        
        self.model = None
        self.history = None
        self.train_ds = None
        self.test_ds = None
        
        # Set random seed for reproducibility
        tf.random.set_seed(self.config['random_seed'])
        np.random.seed(self.config['random_seed'])
        
        logger.info(f"Initialized BrainTumorTrainer with config: {self.config}")
    
    def check_gpu_availability(self):
        """Check and log GPU availability"""
        logger.info(f'TensorFlow Version: {tf.__version__}')
        
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            logger.info(f"GPU Available: {gpu_devices}")
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("GPU not available. Training will use CPU.")
            logger.info("For M1 MacBooks, ensure TensorFlow-Metal is installed.")
    
    def get_data_labels(self, directory: str, shuffle_data: bool = True) -> Tuple[List[str], List[int]]:
        """
        Extract image paths and labels from directory structure
        
        Args:
            directory: Path to data directory
            shuffle_data: Whether to shuffle the data
            
        Returns:
            Tuple of (image_paths, labels)
        """
        data_paths = []
        data_labels = []
        
        if not os.path.isdir(directory):
            logger.error(f"Directory '{directory}' not found")
            return data_paths, data_labels
        
        # Get class directories
        class_dirs = [d for d in os.listdir(directory) 
                     if os.path.isdir(os.path.join(directory, d))]
        
        if not class_dirs:
            logger.error(f"No class directories found in '{directory}'")
            return data_paths, data_labels
        
        # Create label mapping for discovered classes
        discovered_classes = sorted(class_dirs)
        logger.info(f"Discovered classes: {discovered_classes}")
        
        for class_name in discovered_classes:
            if class_name.lower() in CLASS_MAPPINGS:
                class_idx = CLASS_MAPPINGS[class_name.lower()]
                class_dir = os.path.join(directory, class_name)
                
                # Get all image files in class directory
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        image_path = os.path.join(class_dir, filename)
                        data_paths.append(image_path)
                        data_labels.append(class_idx)
            else:
                logger.warning(f"Unknown class '{class_name}' found in directory")
        
        logger.info(f"Found {len(data_paths)} images across {len(set(data_labels))} classes")
        
        if shuffle_data:
            data_paths, data_labels = shuffle(data_paths, data_labels, 
                                            random_state=self.config['random_seed'])
        
        return data_paths, data_labels
    
    def parse_function(self, filename: str, label: int) -> Tuple[tf.Tensor, int]:
        """Parse function for tf.data pipeline"""
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_image(image_string, channels=1, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        return image, label
    
    def create_dataset(self, paths: List[str], labels: List[int]) -> tf.data.Dataset:
        """Create TensorFlow dataset from paths and labels"""
        path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        dataset = path_ds.map(
            lambda path, label: self.parse_function(path, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        return dataset.batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
    
    def create_data_augmentation(self) -> Sequential:
        """Create data augmentation pipeline"""
        aug_config = AUGMENTATION_CONFIG
        
        layers = []
        
        if aug_config.get('horizontal_flip', False):
            layers.append(RandomFlip("horizontal"))
        
        if aug_config.get('vertical_flip', False):
            layers.append(RandomFlip("vertical"))
        
        if aug_config.get('rotation_range', 0) > 0:
            layers.append(RandomRotation(
                aug_config['rotation_range'], 
                fill_mode=aug_config.get('fill_mode', 'constant')
            ))
        
        if aug_config.get('contrast_range', 0) > 0:
            layers.append(RandomContrast(aug_config['contrast_range']))
        
        zoom_height = aug_config.get('zoom_height_factor', 0)
        zoom_width = aug_config.get('zoom_width_factor', 0)
        if zoom_height > 0 or zoom_width > 0:
            layers.append(RandomZoom(
                height_factor=zoom_height,
                width_factor=zoom_width
            ))
        
        trans_height = aug_config.get('translation_height_factor', 0)
        trans_width = aug_config.get('translation_width_factor', 0)
        if trans_height > 0 or trans_width > 0:
            layers.append(RandomTranslation(
                height_factor=trans_height,
                width_factor=trans_width,
                fill_mode=aug_config.get('fill_mode', 'constant')
            ))
        
        return Sequential(layers, name="data_augmentation")
    
    def preprocess_data(self, image, label, is_training=True):
        """Preprocess data with augmentation for training"""
        if is_training and hasattr(self, 'data_augmentation'):
            image = self.data_augmentation(image)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Convert label to one-hot encoding
        label = tf.one_hot(label, depth=len(CLASS_MAPPINGS))
        
        return image, label
    
    def build_model(self) -> Sequential:
        """Build the CNN model architecture"""
        model_config = MODEL_CONFIG
        
        model = Sequential([
            Input(shape=IMAGE_SHAPE, name="input_layer")
        ])
        
        # Add convolutional layers
        for i, layer_config in enumerate(model_config['conv_layers']):
            model.add(Conv2D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config.get('activation', 'relu'),
                name=f"conv2d_{i+1}"
            ))
            model.add(MaxPooling2D(
                pool_size=layer_config['pool_size'],
                name=f"maxpool_{i+1}"
            ))
        
        # Flatten and add dense layers
        model.add(Flatten(name="flatten"))
        
        # Add dropout if specified
        if model_config.get('dropout_rate', 0) > 0:
            model.add(Dropout(model_config['dropout_rate'], name="dropout"))
        
        # Dense hidden layer
        model.add(Dense(
            model_config['dense_units'], 
            activation='relu',
            name="dense_hidden"
        ))
        
        # Output layer
        model.add(Dense(
            len(CLASS_MAPPINGS),
            activation=model_config.get('final_activation', 'softmax'),
            name="output_layer"
        ))
        
        # Compile model
        optimizer = Adam(
            learning_rate=self.config['learning_rate'],
            beta_1=self.config['beta_1'],
            beta_2=self.config['beta_2']
        )
        
        model.compile(
            optimizer=optimizer,
            loss=model_config.get('loss_function', 'categorical_crossentropy'),
            metrics=model_config.get('metrics', ['accuracy'])
        )
        
        logger.info("Model built and compiled successfully")
        return model
    
    def setup_callbacks(self) -> List:
        """Setup training callbacks"""
        callback_config = CALLBACK_CONFIG
        callbacks = []
        
        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(
            monitor=callback_config.get('reduce_lr_monitor', 'val_loss'),
            factor=callback_config.get('reduce_lr_factor', 0.8),
            patience=callback_config.get('reduce_lr_patience', 4),
            min_lr=callback_config.get('reduce_lr_min_lr', 1e-4),
            verbose=self.config.get('verbose', 1)
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            str(CHECKPOINT_PATH),
            monitor=callback_config.get('model_checkpoint_monitor', 'val_accuracy'),
            mode=callback_config.get('model_checkpoint_mode', 'max'),
            save_best_only=callback_config.get('model_checkpoint_save_best_only', True),
            verbose=self.config.get('verbose', 1)
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        if callback_config.get('early_stopping_patience', 0) > 0:
            early_stop = EarlyStopping(
                monitor=callback_config.get('early_stopping_monitor', 'val_accuracy'),
                patience=callback_config.get('early_stopping_patience', 10),
                restore_best_weights=callback_config.get('early_stopping_restore_best_weights', True),
                verbose=self.config.get('verbose', 1)
            )
            callbacks.append(early_stop)
        
        return callbacks
    
    def load_and_prepare_data(self):
        """Load and prepare training and testing data"""
        train_path, test_path = get_data_paths()
        
        logger.info("Loading training data...")
        train_paths, train_labels = self.get_data_labels(train_path)
        
        logger.info("Loading testing data...")
        test_paths, test_labels = self.get_data_labels(test_path)
        
        logger.info(f"Training samples: {len(train_paths)}")
        logger.info(f"Testing samples: {len(test_paths)}")
        
        # Create datasets
        self.train_ds = self.create_dataset(train_paths, train_labels)
        self.test_ds = self.create_dataset(test_paths, test_labels)
        
        # Setup data augmentation
        self.data_augmentation = self.create_data_augmentation()
        
        # Apply preprocessing
        self.train_ds = self.train_ds.map(
            lambda x, y: self.preprocess_data(x, y, is_training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        self.test_ds = self.test_ds.map(
            lambda x, y: self.preprocess_data(x, y, is_training=False),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        logger.info("Data loading and preprocessing completed")
    
    def train(self):
        """Train the model"""
        if self.train_ds is None or self.test_ds is None:
            raise ValueError("Data not loaded. Call load_and_prepare_data() first.")
        
        # Build model
        self.model = self.build_model()
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        logger.info(f"Starting training for {self.config['epochs']} epochs...")
        self.history = self.model.fit(
            self.train_ds,
            epochs=self.config['epochs'],
            validation_data=self.test_ds,
            callbacks=callbacks,
            verbose=self.config.get('verbose', 1)
        )
        
        # Save final model
        final_model_path = get_model_save_path()
        self.model.save(final_model_path)
        logger.info(f"Model saved to {final_model_path}")
        
        return self.history
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict:
        """Evaluate trained model"""
        if model_path:
            self.model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        if self.model is None:
            raise ValueError("No model available for evaluation")
        
        if self.test_ds is None:
            logger.warning("Test data not loaded, loading now...")
            self.load_and_prepare_data()
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(self.test_ds, verbose=0)
        
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        
        # Generate predictions for detailed metrics
        y_true = []
        y_pred = []
        
        for images, labels in self.test_ds.unbatch():
            true_label = np.argmax(labels.numpy())
            pred = self.model.predict(tf.expand_dims(images, 0), verbose=0)
            pred_label = np.argmax(pred)
            
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        # Calculate detailed metrics
        cm = confusion_matrix(y_true, y_pred)
        class_names = [INVERSE_CLASS_MAPPINGS[i] for i in range(len(CLASS_MAPPINGS))]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        if self.history is None:
            logger.error("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=VISUALIZATION_CONFIG['figsize_default'])
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', marker='o')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', marker='o')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', marker='o')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', marker='o')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, class_names, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=VISUALIZATION_CONFIG['figsize_small'])
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap=VISUALIZATION_CONFIG['confusion_matrix_cmap'],
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True
        )
        
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                        help=f'Number of training epochs (default: {TRAINING_CONFIG["epochs"]})')
    parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'],
                        help=f'Batch size (default: {TRAINING_CONFIG["batch_size"]})')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate existing model without training')
    parser.add_argument('--model-path', type=str,
                        help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config_override = {
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    # Initialize trainer
    trainer = BrainTumorTrainer(config_override)
    trainer.check_gpu_availability()
    
    if args.evaluate_only:
        # Evaluation only mode
        model_path = args.model_path or get_model_save_path()
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)
        
        trainer.load_and_prepare_data()
        results = trainer.evaluate_model(model_path)
        
        # Plot results
        class_names = [INVERSE_CLASS_MAPPINGS[i] for i in range(len(CLASS_MAPPINGS))]
        trainer.plot_confusion_matrix(
            results['confusion_matrix'], 
            class_names,
            save_path=RESOURCES_DIR / "confusion_matrix_evaluation.png"
        )
        
    else:
        # Full training mode
        trainer.load_and_prepare_data()
        trainer.train()
        
        # Plot training history
        trainer.plot_training_history(
            save_path=RESOURCES_DIR / "training_history.png"
        )
        
        # Evaluate and plot results
        results = trainer.evaluate_model()
        class_names = [INVERSE_CLASS_MAPPINGS[i] for i in range(len(CLASS_MAPPINGS))]
        trainer.plot_confusion_matrix(
            results['confusion_matrix'], 
            class_names,
            save_path=RESOURCES_DIR / "confusion_matrix.png"
        )
        
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
