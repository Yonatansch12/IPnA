import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to process the image
def process_image(uploaded_image):
    # Convert the image to OpenCV format
    image = np.array(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV color range for veins (greenish-yellow)
    lower_veins = np.array([30, 40, 40])  # Adjust based on your image
    upper_veins = np.array([60, 255, 255])

    # Create a binary mask for veins
    veins_mask = cv2.inRange(hsv_image, lower_veins, upper_veins)

    # Define HSV color range for damaged areas (yellowish-brownish range)
    lower_damage = np.array([10, 50, 50])
    upper_damage = np.array([35, 255, 255])

    # Create a binary mask for damaged areas
    damage_mask = cv2.inRange(hsv_image, lower_damage, upper_damage)

    # Remove the veins from the damage mask
    damage_mask_no_veins = cv2.bitwise_and(damage_mask, cv2.bitwise_not(veins_mask))

    # Apply morphological operations to clean up the new mask
    kernel = np.ones((5, 5), np.uint8)
    damage_mask_no_veins = cv2.morphologyEx(damage_mask_no_veins, cv2.MORPH_CLOSE, kernel)
    damage_mask_no_veins = cv2.morphologyEx(damage_mask_no_veins, cv2.MORPH_OPEN, kernel)

    # Calculate the total leaf area in pixels
    total_leaf_area_pixels = cv2.countNonZero(damage_mask)

    # Calculate the refined damaged area in pixels
    refined_damaged_area_pixels = cv2.countNonZero(damage_mask_no_veins)

    # Highlight the original damaged areas on the original image
    highlighted_damage = cv2.bitwise_and(image, image, mask=damage_mask)

    # Highlight the refined damaged areas on the original image
    highlighted_refined_damage = cv2.bitwise_and(image, image, mask=damage_mask_no_veins)

    return image, highlighted_damage, highlighted_refined_damage, total_leaf_area_pixels, refined_damaged_area_pixels

# Streamlit UI
st.title("Leaf Damage Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Process the image
    original_image, highlighted_damage, highlighted_refined_damage, total_pixels, refined_pixels = process_image(uploaded_image)

    # Display results
    st.image([original_image, highlighted_damage, highlighted_refined_damage], caption=["Original Image", "Original Damage", "Refined Damage"], use_column_width=True)

    # Show calculated results
    st.write(f"Original Damaged Area (pixels): {total_pixels}")
    st.write(f"Refined Damaged Area (pixels): {refined_pixels}")
