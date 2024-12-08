import streamlit as st
from PIL import Image
from PIL import ImageOps
import numpy as np
from tensorflow.keras.models import load_model

MAX_IMAGE_SIZE = (1024, 1024)
model_conv = load_model('model.h5')
model_vgg = load_model('model2.h5')

# Sidebar for user navifation
st.sidebar.title("Navigation")
model_type = st.sidebar.radio("Select a model:", ["Basic Convolutional", "Pretrained VGG16"])
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
predict_button = st.sidebar.button("Predict")

# Main screen
st.markdown("<h1 style='font-size:32px; text-align: center;'>Clothing and Footwear Classification</h1>", unsafe_allow_html=True)
st.write("Welcome! Select a model and upload an image for classification.")

# Initialize state
if "image" not in st.session_state:
    st.session_state["image"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = None

# Update state when an image is uploaded
if uploaded_file is not None:
    try:
        # Open and verify the image
        image = Image.open(uploaded_file)
        image.verify()  # Verify the image format
        image = Image.open(uploaded_file)  # Reopen after verification to avoid issues

        # Resize if the image is too large
        if image.size[0] > MAX_IMAGE_SIZE[0] or image.size[1] > MAX_IMAGE_SIZE[1]:
            st.warning(f"Image size too large! Resizing to {MAX_IMAGE_SIZE}.")
            image.thumbnail(MAX_IMAGE_SIZE)  # Resize while maintaining aspect ratio

        st.session_state["image"] = image
        st.session_state["predictions"] = None  # Reset predictions when a new image is uploaded

    except Exception:
        st.error("The uploaded file is not a valid image. Please upload a PNG, JPG, or JPEG file.")


# Display uploaded image
if st.session_state["image"] is not None:
    st.image(st.session_state["image"], caption="Uploaded Image", use_container_width=True)

# function to process image for basic convolutional model
def preprocess_image_basic(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    image_array = np.array(image).astype("float32") / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, 28, 28, 1)
    return image_array

# function to process image for the model with pretrained base
def preprocess_image_vgg(image):
    # Ensure image is in RGB format
    image = image.convert("RGB")
    # Resize to 32x32
    image = image.resize((32, 32))
    # Convert to numpy array
    image_array = np.array(image).astype("float32") / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 32, 32, 3)
    return image_array

# class labels
labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Predict when the sidebar button is clicked
if predict_button:
    if st.session_state["image"] is not None:
        # Process the image and generate predictions 
        if model_type == "Basic Convolutional":
            model = model_conv
            processed_image = preprocess_image_basic(st.session_state["image"])
        elif model_type == "Pretrained VGG16":
            model = model_vgg
            processed_image = preprocess_image_vgg(st.session_state["image"])

        predictions = model.predict(processed_image)[0]  # Get probabilities for each class
        predicted_class_index = np.argmax(predictions)  # Find the index of the highest probability
        predicted_class_label = labels[predicted_class_index]  # Get the corresponding class name
        
        # Save predictions to session state for display
        st.session_state["predictions"] = [
            {"Class": label, "Probability": f"{prob * 100:.2f}%"}  # Formating to %
            for label, prob in zip(labels, predictions)
        ]
        st.session_state["predicted_class_label"] = predicted_class_label  # Save predicted label

    else:
        # Notify the user to upload an image
        st.error("Please upload an image first.")

# Display classification results or info message
if st.session_state["predictions"] is None:
    st.info("Upload an image and press 'Predict' to see results.")
else:
    st.markdown(f"### Predicted Class: **{st.session_state['predicted_class_label']}**")
    st.write("**Classification Results:**")
    st.table(st.session_state["predictions"])

