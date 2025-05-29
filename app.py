# Configure environment based on deployment platform
try:
    from deployment_config import configure_deployment, get_gpu_info
    deployment_platform = configure_deployment()
except ImportError:
    # Fallback if deployment_config is not available
    deployment_platform = "Unknown (using default configuration)"
    def get_gpu_info():
        return False, "GPU status unknown"

import streamlit as st
import pandas as pd
from PIL import Image
import io
import time


from model_utils import ModelUtils
from config import (
    STREAMLIT_CONFIG, CLASS_DESCRIPTIONS,
    PERFORMANCE_THRESHOLDS, IMAGE_SIZE, NUM_CLASSES
)

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    # initial_sidebar_state="expanded"
)

@st.cache_resource
def get_model_utils():
    """Load and cache the ModelUtils instance."""
    with st.spinner("🧠 Loading AI model..."):
        utils = ModelUtils()
        try:
            utils.load_trained_model()
            st.success("✅ AI model loaded successfully!")
            return utils
        except FileNotFoundError as e:
            st.error(f"❌ Model file not found: {e}")
            st.info("Make sure the model.keras file exists in the project directory.")
            return None
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.info("There was an error loading the model. This might be due to compatibility issues.")
            return None

def add_gpu_status_to_sidebar():
    """Add GPU status information to the sidebar"""
    # Display deployment platform
    st.sidebar.info(f"🚀 Deployment: {deployment_platform}")
    
    # Check GPU status using our safe detection function
    try:
        is_gpu_available, gpu_info = get_gpu_info()
        if is_gpu_available:
            st.sidebar.success("✅ Using GPU acceleration")
            st.sidebar.info(f"GPU info: {gpu_info}")
        else:
            st.sidebar.warning("⚠️ Running in CPU mode")
            st.sidebar.info(gpu_info)
    except Exception as e:
        st.sidebar.error(f"⚠️ Error checking GPU status: {str(e)}")
        st.sidebar.warning("⚠️ Running in CPU mode due to error")

def display_prediction_results(result, image_bytes):
    """Display prediction results with enhanced visualization."""
    
    # Main prediction display
    predicted_class = result['predicted_class']
    confidence = result['confidence']
    
    # Determine confidence level for styling and text
    if confidence >= PERFORMANCE_THRESHOLDS['high_confidence_threshold']:
        confidence_level = "high"
        confidence_icon = "✅"
        confidence_text = "High confidence in this prediction."
    elif confidence >= PERFORMANCE_THRESHOLDS['confidence_threshold']:
        confidence_level = "medium"
        confidence_icon = "⚠️"
        confidence_text = "Moderate confidence. Consider further review."
    else:
        confidence_level = "low"
        confidence_icon = "❗"
        confidence_text = "Low confidence. Results may be less reliable."

    # Display main prediction
    st.header(f"Prediction: {predicted_class}")
    st.subheader(f"Confidence: {confidence:.1%}")

    if confidence_level == "high":
        st.success(f"{confidence_icon} {confidence_text}")
    elif confidence_level == "medium":
        st.warning(f"{confidence_icon} {confidence_text}")
    else:
        st.error(f"{confidence_icon} {confidence_text}")
    
    # Display class description
    if predicted_class in CLASS_DESCRIPTIONS:
        st.info(f"**About {predicted_class}:** {CLASS_DESCRIPTIONS[predicted_class]}")
    
    # Create two columns for image and probability chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Analyzed Image")
        # Display the uploaded image
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, use_container_width=True)
        
        # Image info
        st.markdown(f"""
            **Image Information:**
            - Size: {image.size}
            - Mode: {image.mode}
            - Processed to: {IMAGE_SIZE}
        """)
    
    with col2:
        st.subheader("📊 Prediction Probabilities")
        
        # Create probability chart
        probs_df = pd.DataFrame([
            {'Class': class_name, 'Probability': prob, 'Is_Predicted': class_name == predicted_class}
            for class_name, prob in result['all_probabilities'].items()
        ]).sort_values('Probability', ascending=False)
        
        # Always use Streamlit's bar_chart
        st.bar_chart(probs_df.set_index('Class')['Probability'])
        
        # Uncertainty analysis
        uncertainty = result['uncertainty']
        st.markdown(f"""
            **Uncertainty Analysis:**
            - Prediction uncertainty: {uncertainty:.1%}
            - Top 2 classes: {', '.join(result['top_2_classes'])}
            - Model certainty: {'High' if uncertainty > 0.3 else 'Medium' if uncertainty > 0.1 else 'Low'}
        """)

def display_model_info():
    """Display information about the model and dataset."""
    with st.expander("🧠 About the AI Model", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🏗️ Model Architecture
            
            - **Type**: Convolutional Neural Network (CNN)
            - **Framework**: TensorFlow/Keras
            - **Input**: 168×168 grayscale MRI images
            - **Classes**: 4 tumor types
            - **Layers**: Multiple Conv2D + MaxPooling layers
            - **Optimizer**: Adam with adaptive learning rate
            
            ### 📊 Performance Metrics
            
            - **Accuracy**: ~95%+ on test data
            - **Training**: Data augmentation applied
            - **Validation**: Cross-validation used
            """)
        
        with col2:
            st.markdown("""
            ### 📚 Training Dataset
            - **Source**: [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
            - **Size**: 7,023 brain MRI images
            - **Classes**: Glioma, Meningioma, No Tumor, Pituitary
            - **Augmentation**: Rotation, flipping, contrast enhancement
            
            ### 🎯 Confidence Thresholds
            - **High Confidence**: ≥90% prediction certainty
            - **Medium Confidence**: ≥70% prediction certainty  
            - **Low Confidence**: <70% prediction certainty
            - **Uncertainty**: Measured by probability spread
            """)

def add_demo_section():
    """Add demonstration section with sample images."""
    st.subheader("🎭 Try with Sample Images")
    
    st.info("📝 **Note**: You can use sample images from your dataset for testing. Upload your own MRI scans above to get predictions.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Sample MRI Types
        - **Glioma**: Malignant brain tumor affecting glial cells
        - **Meningioma**: Usually benign tumor of the meninges
        - **No Tumor**: Normal healthy brain tissue
        - **Pituitary**: Tumor affecting the pituitary gland
        """)
    
    with col2:
        st.markdown("""
        #### Upload Guidelines
        - Clear MRI brain scans work best
        - Supported formats: JPG, PNG, BMP, TIFF
        - Axial, sagittal, or coronal views
        - T1 or T2 weighted images preferred
        """)

def display_medical_disclaimer():
    """Display medical disclaimer."""
    st.warning("""
    **Medical Disclaimer:** This tool is for informational and educational purposes only and does not constitute medical advice. 
    The predictions made by this AI model should not be used as a sole basis for making any medical decisions. 
    Always consult with a qualified healthcare professional for any medical concerns or before making any decisions related to your health.
    """)

def sidebar_info():
    """Display sidebar information and controls."""
    st.sidebar.markdown("""
    # 🧠 Brain Tumor AI Classifier
    
    Upload a brain MRI scan to get an AI-powered classification.
    """)
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📁 Upload MRI Image")
    
    with st.sidebar.expander("📋 Upload Guidelines", expanded=False):
        st.markdown("""
        **Supported formats:** JPG, PNG, BMP, TIFF
        **Recommended:** 
        - Clear MRI brain scans
        - Good contrast and resolution
        - Max file size: 10MB
        
        **Best results with:**
        - Axial, sagittal, or coronal views
        - T1 or T2 weighted images
        """)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an MRI image...",
        type=STREAMLIT_CONFIG['allowed_extensions'],
        help=f"Max file size: {STREAMLIT_CONFIG['max_file_size']}MB"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("⚙️ Analysis Settings")
    
    enhance_contrast = st.sidebar.checkbox(
        "🎛️ Enhance image contrast",
        value=False,
        help="Apply contrast enhancement during preprocessing to improve image clarity"
    )
    
    show_confidence_details = st.sidebar.checkbox(
        "📊 Show detailed confidence analysis",
        value=True,
        help="Display additional confidence metrics and uncertainty analysis"
    )
    
    show_class_info = st.sidebar.checkbox(
        "📚 Show class information",
        value=True,
        help="Display detailed information about the predicted tumor type"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📈 Model Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown(f"**Classes:** {NUM_CLASSES}")
        st.markdown(f"**Input Size:** {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    
    with col2:
        st.markdown("**Model Type:** CNN")
        st.markdown("**Accuracy:** ~95%")
    
    with st.sidebar.expander("ℹ️ Quick Info", expanded=False):
        st.markdown("""
        **Tumor Types:**
        - 🔴 **Glioma**: Most common brain tumor
        - 🟡 **Meningioma**: Usually benign tumor
        - 🟢 **No Tumor**: Healthy brain tissue
        - 🔵 **Pituitary**: Affects pituitary gland
        """)
    
    add_gpu_status_to_sidebar()
    
    return uploaded_file, enhance_contrast, show_confidence_details, show_class_info

def main():
    st.title("🧠 Brain Tumor AI Classifier")
    st.subheader("Advanced AI-powered brain tumor detection and classification from MRI scans")
    
    uploaded_file, enhance_contrast, show_confidence_details, show_class_info = sidebar_info()
    
    display_medical_disclaimer()
    
    display_model_info()
    
    if not uploaded_file:
        st.info("👈 Please upload an MRI image using the sidebar to start the analysis.")
        
        add_demo_section()
        
        st.subheader("📋 How to Use This Application")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### Step 1: Upload
            - Use the sidebar to upload your MRI image
            - Supported: JPG, PNG, BMP, TIFF
            - Max size: 10MB
            - Best: Clear brain scans
            """)
        
        with col2:
            st.markdown("""
            #### Step 2: Analyze
            - AI processes your image automatically
            - Advanced CNN analysis
            - Real-time processing
            - Multiple confidence metrics
            """)
        
        with col3:
            st.markdown("""
            #### Step 3: Review
            - Get detailed results and insights
            - Prediction with confidence
            - Probability breakdown
            - Medical information
            """)
        
        st.markdown("---")
        st.subheader("🎯 What This Tool Can Detect")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔴 Glioma**
            - Most common primary brain tumor
            - Arises from glial cells
            - Can be malignant or benign
            
            **🟡 Meningioma** 
            - Usually benign tumor
            - Arises from meninges
            - Often treatable with surgery
            """)
        
        with col2:
            st.markdown("""
            **🟢 No Tumor**
            - Healthy brain tissue
            - Normal MRI appearance
            - No abnormal growths detected
            
            **🔵 Pituitary Tumor**
            - Affects pituitary gland
            - Can affect hormone production
            - Often requires specialized treatment
            """)
        
        add_footer()
        return
    
    if uploaded_file.size > STREAMLIT_CONFIG['max_file_size'] * 1024 * 1024:
        st.error(f"File size too large. Maximum allowed: {STREAMLIT_CONFIG['max_file_size']}MB")
        return
    
    model_utils = get_model_utils()
    if model_utils is None:
        st.error("Failed to load the AI model. Please check if the model file exists.")
        return
    
    try:
        image_bytes = uploaded_file.read()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing your MRI image..."):
            progress_bar.progress(20)
            status_text.text("Loading image...")
            time.sleep(0.5)
            
            progress_bar.progress(50)
            status_text.text("Preprocessing image...")
            
            img_array = model_utils.preprocess_image_from_bytes(
                image_bytes, 
                enhance_contrast=enhance_contrast
            )
            
            progress_bar.progress(80)
            status_text.text("Running AI analysis...")
            time.sleep(0.5)
            
            result = model_utils.predict_with_confidence_analysis(img_array)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.3)
        
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Analysis complete!")
        display_prediction_results(result, image_bytes)
        
        if show_confidence_details:
            with st.expander("🔍 Detailed Confidence Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction Confidence", f"{result['confidence']:.1%}")
                    st.metric("Prediction Uncertainty", f"{result['uncertainty']:.1%}")
                
                with col2:
                    st.metric("High Confidence?", "Yes" if result['high_confidence'] else "No")
                    st.metric("Acceptable Confidence?", "Yes" if result['acceptable_confidence'] else "No")
                
                prob_df = pd.DataFrame([
                    {'Class': class_name, 'Probability': f"{prob:.4f}", 'Percentage': f"{prob:.1%}"}
                    for class_name, prob in result['all_probabilities'].items()
                ]).sort_values('Probability', ascending=False, key=lambda x: x.astype(float))
                
                st.dataframe(prob_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please try uploading a different image or contact support if the problem persists.")

def add_footer():
    """Add footer with additional information."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🔬 About the Technology
        This AI model uses deep learning (CNN) to analyze brain MRI scans and classify different types of tumors with high accuracy.
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Model Performance
        - **Accuracy**: ~95% on test data (refer to model training for exact metrics)
        - **Training**: On a dataset of 7,000+ MRI images.
        - **Classes**: Detects 4 categories (Glioma, Meningioma, No Tumor, Pituitary).
        """)
    
    with col3:
        st.markdown("""
        #### ⚡ Quick Tips
        - Use clear, high-quality MRI scans.
        - Ensure good contrast in images if possible.
        - Always check prediction confidence scores.
        """)
    
    st.markdown("---")
    st.caption("🧠 Brain Tumor AI Classifier | Built with ❤️ & TensorFlow & Streamlit. For educational and research purposes only. Not for medical diagnosis.")

if __name__ == "__main__":
    main()
