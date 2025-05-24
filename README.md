# Deep Learning based Brain Tumour Classification using CNNs and MR Imaging

This project aims to classify brain MRI images into four categories: Glioma, Meningioma, No tumor, and Pituitary tumor. It utilizes Keras (with TensorFlow backend) to build and train a convolutional neural network (CNN). An interactive web application built with Streamlit allows users to upload MRI images and receive instant classification results.

## Features

- **CNN Model**: A Keras-based Convolutional Neural Network trained for classifying brain tumors.
- **Interactive Web App**: A Streamlit application (`app.py`) for easy interaction with the model. Users can upload MRI images and view predictions.
- **Data Preprocessing**: Includes scripts and functions for preparing MRI image data for training and prediction.
- **Model Evaluation**: Jupyter notebook (`tumorClassification.py`) detailing the model training, evaluation, and visualization of results (e.g., confusion matrix, accuracy/loss plots).
- **Dev Environment**: Pre-configured development environment using Dev Containers for consistent setup.
- **Code Quality**: Includes a GitHub Actions workflow for CodeQL to analyze code for security vulnerabilities.

## Understanding Brain Tumors and Detection Methods

A brain tumor represents an abnormal mass or growth of cells in the brain, existing within the confined space of the skull. This condition can lead to serious complications due to increased intracranial pressure and potential brain damage. Tumors can be benign (noncancerous) or malignant (cancerous), and their early detection and accurate classification are critical for effective treatment planning, underscoring the importance of advancements in medical imaging.

Deep learning, particularly in the realm of healthcare, has brought significant improvements in diagnosing various conditions, including brain tumors. The World Health Organization emphasizes the importance of accurate brain tumor diagnosis, which includes detecting the presence of a tumor, pinpointing its location, and classifying its type and grade. This notebook explores the use of Convolutional Neural Networks (CNNs) in a multi-task approach for the detection, classification, and location identification of brain tumors using MRI images, showcasing the potential of these models to revolutionize diagnostics in neurology.

## Motivation

Brain tumors pose a major medical challenge with life-altering consequences. Misdiagnosis may delay treatment and lead to unsuitable interventions, potentially diminishing survival rates. This project seeks to enhance neuro-oncology diagnostic tools by leveraging AI to address existing shortcomings in brain tumor detection and classification.

Improving machine learning models for tumor classification is pivotal in advancing neurology research and patient care. These models can boost diagnostic accuracy, inform more effective treatment strategies, and deepen our understanding of tumor behavior. Ultimately, by integrating robust machine learning techniques, we aim to drive progress in medical research and deliver better healthcare outcomes for individuals affected by brain tumors.

## Scope of Work

This project encompasses the end-to-end development of a CNN-based model for brain tumor classification from MRI images. The key phases include:

1. **Data Collection**: Utilizing a dataset of 7,023 human brain MRI images categorized into four classes: Glioma, Meningioma, No Tumor, and Pituitary.
2. **Data Preprocessing**: Standardizing image dimensions, normalizing pixel values, encoding labels, splitting the dataset, and applying data augmentation techniques.
3. **Model Design**: Developing a sequential CNN model with four convolutional layers, max pooling layers, flattening layers, and dense layers for classification.
4. **Model Training**: Using the Adam optimizer, categorical cross-entropy loss function, and implementing training callbacks such as ReduceLROnPlateau and ModelCheckpoint.
5. **Model Evaluation**: Splitting the dataset into training and testing sets, evaluating model performance using accuracy, precision, recall, F1-score, and confusion matrix.
6. **Deployment**: Creating a user-friendly interface for uploading MRI images and receiving classification results.
7. **Documentation**: Documenting the development process, including data processing, model building, and evaluation.

## About the Dataset

This dataset is a compilation of two primary datasets: figshare and Br35H. The dataset comprises a total of `7023` human **brain MRI images**, categorized into four distinct classes. The dataset focuses on brain tumors and their classification. The four classes are as follows:

- **Glioma**: Cancerous brain tumors in glial cells.
- **Meningioma**: Non-cancerous tumors originating from the meninges.
- **No Tumor**: Normal brain scans without detectable tumors.
- **Pituitary**: Tumors affecting the pituitary gland, which can be cancerous or non-cancerous.

The "No Tumor" class images were obtained from the `Br35H dataset`.

The data link and complete description here [`Brain Tumor Data on Kaggle`](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Methodology

### Data Collection

The dataset consists of 7,023 MRI images categorized as Glioma, Meningioma, No Tumor, and Pituitary. The images are sourced from figshare and Br35H Datasets.

![Dataset Split](datasplit.png "Dataset Split")

### Data Preprocessing

- **Image Resizing**: All images are resized to 168x168 pixels.
- **Normalization**: Pixel values are normalized to the range [0, 1].
- **Label Encoding**: Labels are encoded into one-hot vectors.
- **Train/Test Splitting**: The dataset is split into training and testing sets.
- **Data Augmentation**: Techniques include random horizontal flipping, rotation, contrast adjustment, zoom, and slight translations.

![Augmented and Pre-Processed Images](augmentedtumors.png "Augmented and Pre-Processed Images")

### Model Design

The CNN model architecture includes:

- Four convolutional layers with increasing filter sizes (64 to 128).
- Max pooling layers for spatial dimension reduction.
- A flattening layer to convert 2D feature maps to a 1D vector.
- Two dense layers with SoftMax activation for multi-class classification.

![Model Architecture Summary](model_architecture.png "Model Architecture Summary")

![Model Layers](model.png "Model Layers")

### Model Training

- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Loss Function**: Categorical cross-entropy.
- **Training Callbacks**: ReduceLROnPlateau and ModelCheckpoint.
- **Hyperparameter Tuning**: Batch size of 32 and 50 epochs. Early stopping and learning rate scheduling are used to prevent overfitting.

![Training Curves](trainingepochs.png "Training Curves")

### Model Evaluation

**Class-wise metrics:**

**Class: Glioma**
Precision: 0.9966
Recall: 0.9900
F1-Score: 0.9933

**Class: Meninigioma**
Precision: 0.9837
Recall: 0.9869
F1-Score: 0.9853

**Class: Notumor**
Precision: 0.9926
Recall: 1.0000
F1-Score: 0.9963

**Class: Pituitary**
Precision: 0.9933
Recall: 0.9867
F1-Score: 0.9900

Overall Accuracy: 0.9916

**Confusion Matrix**: Visualizing model performance across all classes.

![Confusion Matrix](confusionmatrix.png "Confusion Matrix")

### Deployment

The trained model is saved in Keras format and integrated into a user-friendly interface for clinical use.

## Results and Discussion

The results indicate that the CNN model performs well in classifying brain tumors from MRI images, achieving high accuracy (99+%) and reliability. The implementation demonstrates the potential of deep learning in enhancing diagnostic accuracy and efficiency in neuro-oncology.

![Testing Samples](testing.png "Testing Samples")

## Conclusion and Future Work

This project showcases the application of CNNs in medical imaging, providing a valuable tool for brain tumor classification. Future work may involve further refining the model, expanding the dataset, and integrating the system into clinical workflows to assist healthcare professionals.

## Project Structure

```plaintext
├── .devcontainer/
│   └── devcontainer.json  # Configuration for VS Code Dev Containers
├── .github/
│   └── workflows/
│       └── codeql.yml     # GitHub Actions workflow for CodeQL
├── brain-tumor-mri-dataset/ # Dataset (not committed, download separately)
│   ├── Testing/
│   └── Training/
├── Data/                    # Sample data for testing (not committed)
├── __pycache__/
├── app.py                   # Main Streamlit application file
├── model_utils.py           # Utility functions for model loading and preprocessing
├── tumorClassification.py   # Python script version of the Jupyter Notebook for model training and evaluation
├── model.keras              # Trained Keras model file
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── ... (other images, notebooks, etc.)
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package installer)
- Git

### Installation and Setup

1. **Clone the repository:**

    ```bash
    # if using HTTPS
    git clone https://github.com/tanush-g/Brain.git
    # if using SSH
    git clone git@github.com:tanush-g/Brain.git
    # if using GitHub CLI
    gh repo clone tanush-g/Brain

    cd Brain
    ```

2. **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset:**
    The dataset is available on Kaggle: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). Download it and place the `Training` and `Testing` folders into the `brain-tumor-mri-dataset` directory at the root of the project.

5. **Train the model (Optional):**
    If you want to retrain the model or explore the training process, run the `tumorClassification.py` script or the original Jupyter notebook. This will generate the `model.keras` file.

    ```bash
    python tumorClassification.py
    ```

    *Note: Ensure you have the dataset downloaded and paths are correctly set in the script if you choose to retrain.*

### Running the Streamlit Application

Once the `model.keras` file is present (either by training or by using a pre-trained one provided with the project), you can run the Streamlit application:

```bash
streamlit run app.py
```

This will start a local web server, and you can interact with the application by navigating to the URL provided in your terminal (usually `http://localhost:8501`).

### Using the Dev Container (VS Code)

This project is configured to use VS Code Dev Containers, which provides a consistent development environment.

1. Ensure you have Docker Desktop installed and running.
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in VS Code.
3. When you open the project folder in VS Code, you should be prompted to "Reopen in Container". Click it.
4. VS Code will build the container based on `.devcontainer/devcontainer.json`. The Streamlit app will automatically start as defined in the `postAttachCommand`.

## Use Cases

- **Educational Tool**: Understand how CNNs can be applied to medical image analysis.
- **Research Prototype**: A starting point for researchers working on brain tumor classification.
- **Clinical Assistant (Proof-of-Concept)**: Demonstrates the potential for AI tools to assist radiologists in diagnosing brain tumors. ***This is not a production-ready medical device and should not be used for actual medical diagnosis without further validation and regulatory approval.***

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
