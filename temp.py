import streamlit as st
from PIL import Image
import numpy as np
import cv2  # For any OpenCV processing (optional)  
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title = "Image Processing",
    options = ["Process Image", "Anime IRL"],
    default_index =0,
    orientation = "horizontal",
)

if selected == "Process Image":
    st.title("Image processing")
if selected == "Anime IRL":
    st.title("Anime IRL")





# Load your AI model here (e.g., using TensorFlow or PyTorch)
# model = ...

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Convert image to a format suitable for your AI model (e.g., NumPy array)
    image_np = np.array(image)

    # Example processing using OpenCV (you can replace this with your AI model)
    # Here, converting the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # If you are using a model, you would pass `image_np` to the model for prediction
    # result = model.predict(image_np) or similar

    # Convert processed result back to an image (Pillow Image format)
    processed_image = Image.fromarray(gray_image)

    # Display the processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)
