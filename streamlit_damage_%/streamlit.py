import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Title and introduction
st.title("Leaf Damage Analysis App")
st.write("Upload an image and adjust settings to analyze leaf damage.")

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    # Sidebar for HUE settings
    st.sidebar.subheader("HUE Settings")
    leaf_hue = st.sidebar.slider("Leaf HUE", 0, 179, 60, 1)
    damage_hue = st.sidebar.slider("Damage HUE", 0, 179, 120, 1)

    # Sidebar for display options
    st.sidebar.subheader("Display Options")
    display_option = st.sidebar.radio(
        "Choose output display:",
        ("Original + Processed Images", "Original Only", "Processed Only")
    )

    # Process the image based on selected HUE values
    def process_image(img, leaf_hue, damage_hue):
        hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Define ranges for leaf and damage
        leaf_lower = np.array([leaf_hue - 10, 50, 50])
        leaf_upper = np.array([leaf_hue + 10, 255, 255])
        
        damage_lower = np.array([damage_hue - 10, 50, 50])
        damage_upper = np.array([damage_hue + 10, 255, 255])
        
        # Masks
        leaf_mask = cv2.inRange(hsv_image, leaf_lower, leaf_upper)
        damage_mask = cv2.inRange(hsv_image, damage_lower, damage_upper)
        
        # Combine masks with the original image
        leaf_area = cv2.bitwise_and(img, img, mask=leaf_mask)
        damage_area = cv2.bitwise_and(img, img, mask=damage_mask)
        return leaf_area, damage_area

    processed_leaf, processed_damage = process_image(image_np, leaf_hue, damage_hue)

    # Display results
    if display_option == "Original + Processed Images":
        st.image([image, processed_leaf, processed_damage], caption=["Original", "Leaf Area", "Damage Area"], use_column_width=True)
    elif display_option == "Original Only":
        st.image(image, caption="Original Image", use_column_width=True)
    elif display_option == "Processed Only":
        st.image([processed_leaf, processed_damage], caption=["Leaf Area", "Damage Area"], use_column_width=True)
