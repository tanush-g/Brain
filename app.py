import streamlit as st
from model_utils import load_trained_model, preprocess_image_from_bytes, predict

# Configuration
MODEL_PATH = "model.keras"
IMAGE_SIZE = (168, 168)
# Mapping class names to indices (must match training)
CLASS_MAPPINGS = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}

@st.cache_resource
def get_model():
    """Load and cache the trained model."""
    return load_trained_model(MODEL_PATH)


def main():
    st.title("Brain Tumor MRI Classifier")
    st.write("Upload a brain MRI (grayscale) to classify into one of 4 categories.")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    if not uploaded_file:
        return

    # Read and preprocess
    img_bytes = uploaded_file.read()
    img_array = preprocess_image_from_bytes(img_bytes, target_size=IMAGE_SIZE)

    # Load model and predict
    model = get_model()
    pred_class, probs = predict(model, img_array, CLASS_MAPPINGS)

    # Display results
    st.image(img_bytes, caption="Uploaded MRI") # use_column_width=True
    st.markdown(f"**Prediction:** {pred_class}")
    st.markdown("**Probabilities:**")
    for class_name, idx in CLASS_MAPPINGS.items():
        st.write(f"- {class_name}: {probs[idx]:.2%}")


if __name__ == "__main__":
    main()
