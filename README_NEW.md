# 🧠 Brain Tumor Classification AI

A sophisticated deep learning application for brain tumor classification from MRI scans using Convolutional Neural Networks (CNN). This project provides both a training pipeline and a user-friendly web interface for real-time tumor detection and classification.

## 🎯 Features

- **AI-Powered Classification**: Advanced CNN model for accurate brain tumor detection
- **4 Tumor Types**: Glioma, Meningioma, No Tumor, Pituitary tumors
- **Web Interface**: Beautiful Streamlit app with real-time predictions
- **Confidence Analysis**: Detailed uncertainty and confidence metrics
- **Interactive Charts**: Plotly visualizations for probability distributions
- **Medical-Grade UI**: Professional interface with medical disclaimers
- **Training Pipeline**: Complete training script with data augmentation
- **Performance Metrics**: Comprehensive evaluation and visualization tools

## 🏗️ Project Structure

```plaintext
Brain/
├── app.py                          # Streamlit web application
├── train_model.py                  # Training pipeline
├── model_utils.py                  # Model utilities and preprocessing
├── config.py                       # Configuration management
├── run_app.py                      # App launcher script
├── requirements.txt                # Python dependencies
├── model.keras                     # Trained model (generated after training)
├── brain-tumor-mri-dataset/        # Main dataset
│   ├── Training/                   # Training images
│   └── Testing/                    # Testing images
├── resources/                      # Generated plots and visualizations
└── logs/                           # Training logs
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd Brain
pip install -r requirements.txt
```

### 2. Extract Dataset

Make sure your `brain-tumor-mri-dataset.zip` is extracted to `brain-tumor-mri-dataset/`

### 3. Train the Model (if needed)

```bash
python train_model.py
```

### 4. Run the Web App

```bash
python run_app.py
# or directly:
streamlit run app.py
```

## 📊 Model Architecture

- **Type**: Convolutional Neural Network (CNN)
- **Input**: 168×168 grayscale MRI images
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam with adaptive learning rate
- **Accuracy**: ~95%+ on test data
- **Classes**: 4 (Glioma, Meningioma, No Tumor, Pituitary)

### Architecture Details

```
Input Layer (168x168x1)
    ↓
Conv2D + MaxPooling (64 filters, 5x5 kernel)
    ↓
Conv2D + MaxPooling (64 filters, 5x5 kernel)
    ↓
Conv2D + MaxPooling (128 filters, 4x4 kernel)
    ↓
Conv2D + MaxPooling (128 filters, 4x4 kernel)
    ↓
Flatten + Dropout (0.3)
    ↓
Dense (512 units, ReLU)
    ↓
Output (4 units, Softmax)
```

## 🎛️ Usage

### Training a New Model

```bash
# Basic training
python train_model.py

# Custom parameters
python train_model.py --epochs 100 --batch-size 64

# Evaluation only
python train_model.py --evaluate-only --model-path model.keras
```

### Running the Web App

1. Launch the app: `python run_app.py`
2. Upload an MRI image using the sidebar
3. Adjust analysis settings if needed
4. View real-time predictions with confidence scores
5. Analyze detailed probability distributions

### Using Model Utils Programmatically

```python
from model_utils import ModelUtils

# Initialize
utils = ModelUtils()
utils.load_trained_model()

# Predict from image bytes
with open('mri_scan.jpg', 'rb') as f:
    image_bytes = f.read()

img_array = utils.preprocess_image_from_bytes(image_bytes)
result = utils.predict_with_confidence_analysis(img_array)

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📋 Configuration

All settings are centralized in `config.py`:

- **Training Parameters**: Epochs, batch size, learning rate
- **Model Architecture**: Layer configurations, dropout rates
- **Data Augmentation**: Rotation, contrast, zoom settings
- **UI Settings**: Colors, thresholds, styling
- **File Paths**: Model locations, data directories

## 🔧 Data Augmentation

The training pipeline includes comprehensive data augmentation:

- Horizontal flipping
- Rotation (±2%)
- Contrast adjustment (±10%)
- Zoom (±1-5%)
- Translation (±0.15%)

## 📈 Performance Metrics

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~95%
- **Test Accuracy**: ~95%
- **Confidence Thresholds**:
  - High Confidence: ≥90%
  - Medium Confidence: ≥70%
  - Low Confidence: <70%

## 🎨 Web Interface Features

- **Real-time Processing**: Instant MRI analysis
- **Progress Indicators**: Visual feedback during processing
- **Interactive Charts**: Plotly-powered probability visualizations
- **Confidence Analysis**: Detailed uncertainty metrics
- **Medical Disclaimer**: Professional medical warnings
- **Responsive Design**: Works on desktop and mobile
- **Error Handling**: Robust error management and user feedback

## 🧪 Testing

Test the application with sample images:

1. Use images from the `brain-tumor-mri-dataset/Testing/` folder
2. Try different tumor types to see classification accuracy
3. Test with various image qualities and formats

## 📦 Dependencies

```
streamlit>=1.28.0
tensorflow>=2.13.0
keras>=2.13.0
numpy>=1.24.0
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
```

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is for educational and research purposes only. It should NOT be used for actual medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice.

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**:

   ```bash
   python train_model.py  # Train the model first
   ```

2. **Import errors**:

   ```bash
   pip install -r requirements.txt
   ```

3. **GPU issues on macOS**:

   ```bash
   pip install tensorflow-metal  # For M1/M2 Macs
   ```

4. **Memory issues**:
   - Reduce batch size in `config.py`
   - Use CPU instead of GPU for training

### Performance Issues

- Ensure adequate system RAM (8GB+ recommended)
- Use SSD storage for faster data loading
- Consider GPU acceleration for training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset source: Kaggle Brain Tumor MRI Dataset
- TensorFlow team for the amazing framework
- Streamlit for the beautiful web interface
- Medical imaging community for research insights

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the configuration in `config.py`
3. Ensure all dependencies are installed correctly
4. Open an issue with detailed error messages

---

**Built with ❤️ using TensorFlow, Keras, and Streamlit**
