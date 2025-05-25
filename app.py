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
import tensorflow as tf

# Check for plotly availability
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("📊 Plotly not available. Install with: pip install plotly")

from model_utils import ModelUtils
from config import (
    STREAMLIT_CONFIG, CLASS_DESCRIPTIONS,
    PERFORMANCE_THRESHOLDS, IMAGE_SIZE
)

# Configure Streamlit page
st.set_page_config(
    page_title=STREAMLIT_CONFIG['page_title'],
    page_icon=STREAMLIT_CONFIG['page_icon'],
    layout=STREAMLIT_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    .confidence-high {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }
    .confidence-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
    }
    .confidence-low {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
    }
    .info-box {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
        padding: 1.5rem;
        border-left: 4px solid #FF6B6B;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .upload-section {
        border: 2px dashed #FF6B6B;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #fafafa;
        margin: 1rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #FF6B6B;
    }
    .disclaimer-box {
        background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(244, 67, 54, 0.1);
    }
    .stats-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Determine confidence level for styling
    if confidence >= PERFORMANCE_THRESHOLDS['high_confidence_threshold']:
        confidence_class = "confidence-high"
        confidence_text = "High Confidence"
        confidence_icon = "✅"
    elif confidence >= PERFORMANCE_THRESHOLDS['confidence_threshold']:
        confidence_class = "confidence-medium"
        confidence_text = "Medium Confidence"
        confidence_icon = "⚠️"
    else:
        confidence_class = "confidence-low"
        confidence_text = "Low Confidence"
        confidence_icon = "❌"
    
    # Display main prediction
    st.markdown(f"""
        <div class="prediction-box {confidence_class}">
            <h2>{confidence_icon} Prediction: {predicted_class}</h2>
            <h3>Confidence: {confidence:.1%}</h3>
            <p>{confidence_text}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display class description
    if predicted_class in CLASS_DESCRIPTIONS:
        st.markdown(f"""
            <div class="info-box">
                <strong>About {predicted_class}:</strong><br>
                {CLASS_DESCRIPTIONS[predicted_class]}
            </div>
        """, unsafe_allow_html=True)
    
    # Create two columns for image and probability chart
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Analyzed Image")
        # Display the uploaded image
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, use_container_width=True)
        
        # Image info
        st.markdown(f"""
            <div class="metric-container">
                <strong>Image Information:</strong><br>
                • Size: {image.size}<br>
                • Mode: {image.mode}<br>
                • Processed to: {IMAGE_SIZE}
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📊 Prediction Probabilities")
        
        # Create probability chart
        probs_df = pd.DataFrame([
            {'Class': class_name, 'Probability': prob, 'Is_Predicted': class_name == predicted_class}
            for class_name, prob in result['all_probabilities'].items()
        ]).sort_values('Probability', ascending=False)
        
        if PLOTLY_AVAILABLE:
            # Create bar chart with Plotly
            import plotly.express as px
            fig = px.bar(
                probs_df, 
                x='Probability', 
                y='Class',
                orientation='h',
                color='Is_Predicted',
                color_discrete_map={True: '#FF6B6B', False: '#E0E0E0'},
                title="Class Probabilities"
            )
            
            fig.update_layout(
                showlegend=False,
                height=300,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            fig.update_traces(
                texttemplate='%{x:.1%}',
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to simple bar chart with streamlit
            st.bar_chart(probs_df.set_index('Class')['Probability'])
        
        # Uncertainty analysis
        uncertainty = result['uncertainty']
        st.markdown(f"""
            <div class="metric-container">
                <strong>Uncertainty Analysis:</strong><br>
                • Prediction uncertainty: {uncertainty:.1%}<br>
                • Top 2 classes: {', '.join(result['top_2_classes'])}<br>
                • Model certainty: {'High' if uncertainty > 0.3 else 'Medium' if uncertainty > 0.1 else 'Low'}
            </div>
        """, unsafe_allow_html=True)

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
            - **Source**: Kaggle Brain Tumor MRI Dataset
            - **Size**: 7,022+ brain MRI images
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
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🧠 Sample MRI Types</h4>
                <p><strong>Glioma:</strong> Malignant brain tumor affecting glial cells</p>
                <p><strong>Meningioma:</strong> Usually benign tumor of the meninges</p>
                <p><strong>No Tumor:</strong> Normal healthy brain tissue</p>
                <p><strong>Pituitary:</strong> Tumor affecting the pituitary gland</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>📋 Upload Guidelines</h4>
                <p>• Clear MRI brain scans work best</p>
                <p>• Supported formats: JPG, PNG, BMP, TIFF</p>
                <p>• Axial, sagittal, or coronal views</p>
                <p>• T1 or T2 weighted images preferred</p>
            </div>
        """, unsafe_allow_html=True)

def display_medical_disclaimer():
    """Display medical disclaimer."""
    st.markdown("""
        <div style="background-color: #ffebee; padding: 1rem; border-radius: 5px; border-left: 4px solid #f44336; margin: 2rem 0;">
            <h4 style="color: #c62828; margin-top: 0;">⚠️ Medical Disclaimer</h4>
            <p style="margin-bottom: 0;">
                This AI tool is for <strong>educational and research purposes only</strong>. 
                It should not be used for actual medical diagnosis or treatment decisions. 
                Always consult with qualified healthcare professionals for medical advice.
            </p>
        </div>
    """, unsafe_allow_html=True)

def sidebar_info():
    """Display sidebar information and controls."""
    st.sidebar.markdown("""
    # 🧠 Brain Tumor AI Classifier
    
    Upload a brain MRI scan to get an AI-powered classification.
    """)
    
    st.sidebar.markdown("---")
    
    # File upload section
    st.sidebar.subheader("📁 Upload MRI Image")
    
    # Add upload guidelines
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
    
    # Analysis Settings
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
    
    # Model Statistics
    st.sidebar.subheader("📈 Model Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.markdown('<div class="stats-card"><strong>Classes</strong><br>4</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-card"><strong>Input Size</strong><br>168×168</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-card"><strong>Model Type</strong><br>CNN</div>', unsafe_allow_html=True)
        st.markdown('<div class="stats-card"><strong>Accuracy</strong><br>~95%</div>', unsafe_allow_html=True)
    
    # Quick Info
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
    # Header
    st.markdown('<h1 class="main-header">🧠 Brain Tumor AI Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-powered brain tumor detection and classification from MRI scans</p>', unsafe_allow_html=True)
    
    # Sidebar
    uploaded_file, enhance_contrast, show_confidence_details, show_class_info = sidebar_info()
    
    # Medical disclaimer
    display_medical_disclaimer()
    
    # Model information
    display_model_info()
    
    # Main content
    if not uploaded_file:
        st.info("👆 Please upload an MRI image using the sidebar to start the analysis.")
        
        # Demo section
        add_demo_section()
        
        # How to use section
        st.subheader("📋 How to Use This Application")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="feature-card">
                    <h4>📁 Step 1: Upload</h4>
                    <p>Use the sidebar to upload your MRI image</p>
                    <ul>
                        <li>Supported: JPG, PNG, BMP, TIFF</li>
                        <li>Max size: 10MB</li>
                        <li>Best: Clear brain scans</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="feature-card">
                    <h4>🔍 Step 2: Analyze</h4>
                    <p>AI processes your image automatically</p>
                    <ul>
                        <li>Advanced CNN analysis</li>
                        <li>Real-time processing</li>
                        <li>Multiple confidence metrics</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="feature-card">
                    <h4>📊 Step 3: Review</h4>
                    <p>Get detailed results and insights</p>
                    <ul>
                        <li>Prediction with confidence</li>
                        <li>Probability breakdown</li>
                        <li>Medical information</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Additional information
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
        
        # Add footer
        add_footer()
        return
    
    # File size check
    if uploaded_file.size > STREAMLIT_CONFIG['max_file_size'] * 1024 * 1024:
        st.error(f"File size too large. Maximum allowed: {STREAMLIT_CONFIG['max_file_size']}MB")
        return
    
    # Load model
    model_utils = get_model_utils()
    if model_utils is None:
        st.error("Failed to load the AI model. Please check if the model file exists.")
        return
    
    # Process the uploaded image
    try:
        image_bytes = uploaded_file.read()
        
        # Show processing status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("🔍 Analyzing your MRI image..."):
            # Add a small delay for better UX
            progress_bar.progress(20)
            status_text.text("Loading image...")
            time.sleep(0.5)
            
            progress_bar.progress(50)
            status_text.text("Preprocessing image...")
            
            # Preprocess image
            img_array = model_utils.preprocess_image_from_bytes(
                image_bytes, 
                enhance_contrast=enhance_contrast
            )
            
            progress_bar.progress(80)
            status_text.text("Running AI analysis...")
            time.sleep(0.5)
            
            # Make prediction with confidence analysis
            result = model_utils.predict_with_confidence_analysis(img_array)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            time.sleep(0.3)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success("✅ Analysis complete!")
        display_prediction_results(result, image_bytes)
        
        # Additional confidence details
        if show_confidence_details:
            with st.expander("🔍 Detailed Confidence Analysis", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Prediction Confidence", f"{result['confidence']:.1%}")
                    st.metric("Prediction Uncertainty", f"{result['uncertainty']:.1%}")
                
                with col2:
                    st.metric("High Confidence?", "Yes" if result['high_confidence'] else "No")
                    st.metric("Acceptable Confidence?", "Yes" if result['acceptable_confidence'] else "No")
                
                # Raw probabilities table
                st.subheader("Raw Probability Scores")
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
    
    # Footer content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            ### 🔬 About the Technology
            This AI model uses deep learning (CNN) to analyze brain MRI scans and classify different types of tumors with high accuracy.
        """)
    
    with col2:
        st.markdown("""
            ### 📊 Model Performance
            - **Accuracy**: ~95% on test data
            - **Training**: 7,000+ MRI images
            - **Classes**: 4 tumor types
        """)
    
    with col3:
        st.markdown("""
            ### ⚡ Quick Tips
            - Use clear, high-quality MRI scans
            - Ensure good contrast in images
            - Check prediction confidence scores
        """)
    
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #eee; margin-top: 2rem;">
            <p>🧠 Brain Tumor AI Classifier | Built with TensorFlow & Streamlit</p>
            <p style="font-size: 0.9em;">For educational and research purposes only. Not for medical diagnosis.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
