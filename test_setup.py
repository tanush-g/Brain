#!/usr/bin/env python3
"""
Test script to verify the brain tumor classification setup
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully")
        
        # Check GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"🚀 GPU available: {len(gpu_devices)} device(s)")
        else:
            print("💻 Using CPU (no GPU detected)")
            
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow (PIL) imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration file"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import (
            IMAGE_SIZE, CLASS_MAPPINGS, TRAINING_CONFIG,
            get_data_paths, validate_config
        )
        print("✅ Configuration imported successfully")
        
        # Validate config
        validate_config()
        print("✅ Configuration validation passed")
        
        print(f"📊 Image size: {IMAGE_SIZE}")
        print(f"🎯 Number of classes: {len(CLASS_MAPPINGS)}")
        print(f"🔢 Classes: {list(CLASS_MAPPINGS.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_utils():
    """Test model utilities"""
    print("\n🧪 Testing model utilities...")
    
    try:
        from model_utils import ModelUtils
        print("✅ ModelUtils imported successfully")
        
        # Test initialization
        utils = ModelUtils()
        print("✅ ModelUtils initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model utils test failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "app.py",
        "config.py", 
        "model_utils.py",
        "train_model.py",
        "requirements.txt"
    ]
    
    required_dirs = [
        "brain-tumor-mri-dataset",
        "resources",
        "logs",
        "models"
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} missing")
            all_good = False
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/ directory found")
        else:
            print(f"⚠️  {dir_name}/ directory missing (will be created)")
    
    # Check for model file
    if Path("model.keras").exists():
        print("✅ model.keras found (ready to run app)")
    else:
        print("⚠️  model.keras not found (need to train model first)")
    
    return all_good

def test_dataset():
    """Test dataset structure"""
    print("\n🧪 Testing dataset...")
    
    dataset_path = Path("brain-tumor-mri-dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not found")
        return False
    
    train_path = dataset_path / "Training"
    test_path = dataset_path / "Testing"
    
    if train_path.exists():
        print("✅ Training directory found")
        # Count training images
        train_count = 0
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                class_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                print(f"  📁 {class_dir.name}: {class_count} images")
                train_count += class_count
        print(f"📊 Total training images: {train_count}")
    else:
        print("❌ Training directory not found")
        return False
    
    if test_path.exists():
        print("✅ Testing directory found")
        # Count test images
        test_count = 0
        for class_dir in test_path.iterdir():
            if class_dir.is_dir():
                class_count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
                test_count += class_count
        print(f"📊 Total testing images: {test_count}")
    else:
        print("❌ Testing directory not found")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧠 Brain Tumor Classification - System Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Model Utils Test", test_model_utils),
        ("File Structure Test", test_file_structure),
        ("Dataset Test", test_dataset)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with error: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        if Path("model.keras").exists():
            print("🚀 You can now run: python run_app.py")
        else:
            print("📝 Train the model first: python train_model.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
